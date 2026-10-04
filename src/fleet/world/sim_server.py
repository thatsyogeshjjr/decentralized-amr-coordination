"""MuJoCo world server: physics, emulated sensors, world events, ground-truth metrics, visual viewer.

Role boundary (design doc §7.1, FR-03): this process is the *physical world and hardware clock*.  It integrates
physics, produces each robot's own sensor readings (pose/odometry, 360-degree range ring, battery), applies the
wheel-speed commands each robot sends, and executes scheduled world events (WMS order releases, aisle
blockages, robot crashes).  It never plans routes, sets priorities or allocates tasks.

Lockstep: every control period the server sends SENSE(k) to each live robot and waits for CMD(k).  CMD(k)
carries counters of peer messages the robot sent during tick k; SENSE(k+1) forwards those counters so each
robot can wait until the corresponding peer traffic has physically arrived (reproducible delivery barrier).
The barrier never carries message content.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import zmq

from fleet.common.config import config_hash, ports, resolve
from fleet.world.mapgrid import WarehouseMap
from fleet.world.mjcf_builder import BLOCK_PARK, ROBOT_COLORS, build_world_xml, robot_name
from fleet.world.robot_mjcf import MIR
from fleet.world.viewer_thread import ViewerThread


def rc_node(m: WarehouseMap, rc) -> int:
    r, c = int(rc[0]), int(rc[1])
    n = r * m.W + c
    if n not in m.pos:
        raise ValueError(f"cell {rc} is not traversable")
    return n


# ----------------------------------------------------------------------------------------------- physics world
class World:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.map = WarehouseMap.load(resolve(cfg["world"]["map"]), cfg["world"]["cell"])
        self.n = int(cfg["fleet"]["n_robots"])
        missing = [i for i in range(self.n) if i not in self.map.starts]
        if missing:
            raise ValueError(f"map has no start cell for robots {missing}")
        yaw = float(cfg["fleet"]["start_yaw"])
        self.names = [robot_name(i) for i in range(self.n)]
        xml = build_world_xml(self.map, [(self.map.starts[i], yaw) for i in range(self.n)],
                              n_blocks=int(cfg["world"]["n_blocks"]), seed=int(cfg["run"]["seed"]))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        M = self.model
        oid = lambda kind, nm: mujoco.mj_name2id(M, kind, nm)
        self.body = [oid(mujoco.mjtObj.mjOBJ_BODY, nm) for nm in self.names]
        self.qadr = [M.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_free")] for nm in self.names]
        self.vadr = [M.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_free")] for nm in self.names]
        self.act = [(oid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{nm}_wl"), oid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{nm}_wr"))
                    for nm in self.names]
        self.wheel_dof = [(M.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_wl")],
                           M.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_wr")]) for nm in self.names]
        ang = np.array(MIR.ray_angles())
        self._ray_dirs_body = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
        self._ray_groups = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)   # collision geoms only (groups 0, 3)
        self._ray_geomid = np.zeros(MIR.range_rays, dtype=np.int32)
        self._ray_dist = np.zeros(MIR.range_rays, dtype=np.float64)
        self.chassis = {oid(mujoco.mjtObj.mjOBJ_GEOM, f"{nm}_chassis"): i for i, nm in enumerate(self.names)}
        self.cargo_geom = [oid(mujoco.mjtObj.mjOBJ_GEOM, f"{nm}_cargo") for nm in self.names]
        self.stripe_geom = [oid(mujoco.mjtObj.mjOBJ_GEOM, f"{nm}_stripe") for nm in self.names]
        self.block_mocap = [M.body_mocapid[oid(mujoco.mjtObj.mjOBJ_BODY, f"block{k}")] for k in range(int(cfg["world"]["n_blocks"]))]
        self.static_geoms = {g for g in range(M.ngeom) if M.geom_contype[g] == 4}
        self.dt = 1.0 / float(cfg["run"]["control_hz"])
        self.nsub = max(1, int(round(self.dt / M.opt.timestep)))
        b = cfg["battery"]
        init = list(b["initial"])
        self.battery = np.array([float(init[i % len(init)]) for i in range(self.n)])
        self.alive = [True] * self.n
        self.rng = np.random.default_rng(int(cfg["run"]["seed"]) + 991)
        mujoco.mj_forward(M, self.data)

    # -- state accessors ------------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return float(self.data.time)

    def pose(self, i: int) -> tuple[float, float, float]:
        q = self.data.qpos[self.qadr[i]: self.qadr[i] + 7]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), yaw

    def twist(self, i: int) -> tuple[float, float]:
        qv = self.data.qvel[self.vadr[i]: self.vadr[i] + 6]
        _, _, yaw = self.pose(i)
        return float(qv[0] * math.cos(yaw) + qv[1] * math.sin(yaw)), float(qv[5])

    def chassis_center(self, i: int) -> np.ndarray:
        x, y, yaw = self.pose(i)
        return np.array([x + MIR.wheel_dx * math.cos(yaw), y + MIR.wheel_dx * math.sin(yaw)])

    def ranges(self, i: int) -> list[float]:
        """360-degree range ring from base_link at scanner height; -1 = nothing within cutoff."""
        b = self.body[i]
        R = self.data.xmat[b].reshape(3, 3)
        origin = self.data.xpos[b] + R @ np.array([0.0, 0.0, MIR.range_height])
        dirs = (self._ray_dirs_body @ R.T).flatten()
        mujoco.mj_multiRay(self.model, self.data, origin, dirs, self._ray_groups, 1, b,
                           self._ray_geomid, self._ray_dist, None, MIR.range_rays, MIR.range_cutoff)
        return [round(float(r), 3) if 0 <= r <= MIR.range_cutoff else -1.0 for r in self._ray_dist]

    def sensed_pose(self, i: int) -> list[float]:
        x, y, yaw = self.pose(i)
        pn, yn = float(self.cfg["world"]["pose_noise"]), float(self.cfg["world"]["yaw_noise"])
        if pn > 0:
            x += self.rng.normal(0, pn)
            y += self.rng.normal(0, pn)
        if yn > 0:
            yaw += self.rng.normal(0, yn)
        return [round(x, 4), round(y, 4), round(yaw, 5)]

    # -- actuation & integration ----------------------------------------------------------------------------
    def set_wheels(self, i: int, wl: float, wr: float) -> None:
        al, ar = self.act[i]
        self.data.ctrl[al] = wl
        self.data.ctrl[ar] = wr

    def step(self, metrics: "Metrics") -> None:
        for _ in range(self.nsub):
            mujoco.mj_step(self.model, self.data)
            metrics.contacts(self)
        self._battery()

    def _battery(self) -> None:
        b = self.cfg["battery"]
        cap_j = float(b["capacity_wh"]) * 3600.0
        chargers = [self.map.pos[n] for n in self.map.stations("charger")]
        for i in range(self.n):
            if not self.alive[i]:
                continue
            p = float(b["idle_w"])
            for a, dof in zip(self.act[i], self.wheel_dof[i]):
                p += abs(float(self.data.actuator_force[a]) * float(self.data.qvel[dof]))
            self.battery[i] -= p * self.dt * float(b["drain_scale"]) / cap_j
            x, y, _ = self.pose(i)
            v, _ = self.twist(i)
            if abs(v) < 0.05 and any(math.hypot(x - cx, y - cy) < 0.35 for cx, cy in chargers):
                self.battery[i] += float(b["charge_w"]) * self.dt * float(b["drain_scale"]) / cap_j
            self.battery[i] = float(np.clip(self.battery[i], 0.0, 1.0))
            if self.battery[i] <= 0.0:
                self.set_wheels(i, 0.0, 0.0)

    # -- world events ----------------------------------------------------------------------------------------
    def node_clear(self, node: int, radius: float = 1.3) -> bool:
        x, y = self.map.pos[node]
        return all(math.hypot(*(self.chassis_center(i) - (x, y))) > radius for i in range(self.n))

    def place_block(self, k: int, node: int) -> None:
        x, y = self.map.pos[node]
        self.data.mocap_pos[self.block_mocap[k]] = [x, y, 0.0]

    def remove_block(self, k: int) -> None:
        self.data.mocap_pos[self.block_mocap[k]] = [BLOCK_PARK[0] - 3 * k, BLOCK_PARK[1], 0.0]

    def show_cargo(self, i: int, on: bool) -> None:
        self.model.geom_rgba[self.cargo_geom[i], 3] = 1.0 if on else 0.0


# ----------------------------------------------------------------------------------------------- ground truth
@dataclass
class Metrics:
    n: int
    d_safe: float
    eq8_ticks: int = 0                     # sum_t sum_{i<j} I(||p_i - p_j|| < d_safe)  (eq. 8)
    eq8_events: int = 0                    # rising edges of the above per pair
    robot_contacts: int = 0                # MuJoCo chassis-chassis contact onsets
    static_contacts: int = 0               # chassis-rack/wall/pallet contact onsets
    min_pair_dist: float = float("inf")
    path_len: list = field(default_factory=list)
    _close: set = field(default_factory=set)
    _touch: set = field(default_factory=set)
    _stouch: set = field(default_factory=set)
    _last_xy: list = field(default_factory=list)
    contact_log: list = field(default_factory=list)

    def __post_init__(self):
        self.path_len = [0.0] * self.n
        self._last_xy = [None] * self.n

    def contacts(self, w: World) -> None:
        touch, stouch = set(), set()
        d = w.data
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            i1, i2 = w.chassis.get(g1), w.chassis.get(g2)
            if i1 is not None and i2 is not None:
                touch.add((min(i1, i2), max(i1, i2)))
            elif i1 is not None and g2 in w.static_geoms:
                stouch.add(i1)
            elif i2 is not None and g1 in w.static_geoms:
                stouch.add(i2)
        new = touch - self._touch
        if new:
            self.robot_contacts += len(new)
            self.contact_log += [{"t": round(w.t, 3), "pair": list(p)} for p in new]
        snew = stouch - self._stouch
        if snew:
            self.static_contacts += len(snew)
            self.contact_log += [{"t": round(w.t, 3), "static": i} for i in snew]
        self._touch, self._stouch = touch, stouch

    def tick(self, w: World) -> None:
        cs = [w.chassis_center(i) for i in range(self.n)]
        close = set()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = float(np.linalg.norm(cs[i] - cs[j]))
                self.min_pair_dist = min(self.min_pair_dist, dist)
                if dist < self.d_safe:
                    close.add((i, j))
        self.eq8_ticks += len(close)
        self.eq8_events += len(close - self._close)
        self._close = close
        for i in range(self.n):
            x, y, _ = w.pose(i)
            if self._last_xy[i] is not None:
                self.path_len[i] += math.hypot(x - self._last_xy[i][0], y - self._last_xy[i][1])
            self._last_xy[i] = (x, y)

    def summary(self) -> dict:
        return {
            "eq8_ticks": self.eq8_ticks, "eq8_events": self.eq8_events,
            "robot_contacts": self.robot_contacts, "static_contacts": self.static_contacts,
            "collisions": self.eq8_events + self.robot_contacts,
            "min_pair_dist": None if math.isinf(self.min_pair_dist) else round(self.min_pair_dist, 4),
            "path_len": [round(p, 3) for p in self.path_len],
            "contact_log": self.contact_log[:200],
        }


# ----------------------------------------------------------------------------------------------- task book (WMS)
@dataclass
class Order:
    task_id: str
    release: float
    pickup: int
    drop: int
    announced: bool = False
    picked_by: int | None = None
    picked_t: float | None = None
    delivered_t: float | None = None


class TaskBook:
    """Warehouse management system order stream.  Releases orders at their release time (broadcast to all
    robots like a WMS terminal) and verifies pickups/deliveries against ground-truth robot positions."""

    def __init__(self, cfg: dict, m: WarehouseMap):
        self.cfg, self.map = cfg, m
        t = cfg["tasks"]
        rng = np.random.default_rng(int(cfg["run"]["seed"]) + 17)
        self.orders: list[Order] = []
        if t["list"]:
            for k, (rel, p, d) in enumerate(t["list"]):
                self.orders.append(Order(f"T{k:03d}", float(rel), rc_node(m, p), rc_node(m, d)))
        else:
            picks, drops = m.stations("pickup"), m.stations("drop")
            rel = 0.0
            for k in range(int(t["count"])):
                if k > 0 and float(t["rate"]) > 0:
                    rel += float(rng.exponential(1.0 / float(t["rate"])))
                self.orders.append(Order(f"T{k:03d}", round(rel, 3), int(rng.choice(picks)), int(rng.choice(drops))))
        self.rejected: list[dict] = []

    def due(self, now: float) -> list[Order]:
        out = []
        for o in self.orders:
            if not o.announced and o.release <= now + 1e-9:
                o.announced = True
                out.append(o)
        return out

    def report(self, w: World, i: int, ev: dict) -> bool:
        """Verify a robot's pickup/delivery claim against ground truth.  Returns True if accepted."""
        o = next((o for o in self.orders if o.task_id == ev.get("task_id")), None)
        if o is None:
            return False
        x, y, _ = w.pose(i)
        node = o.pickup if ev["phase"] == "picked" else o.drop
        nx, ny = self.map.pos[node]
        ok = math.hypot(x - nx, y - ny) < 0.35 and abs(w.twist(i)[0]) < 0.1
        if ev["phase"] == "picked":
            ok = ok and o.picked_by is None and o.delivered_t is None
            if ok:
                o.picked_by, o.picked_t = i, w.t
                w.show_cargo(i, True)
        elif ev["phase"] == "delivered":
            ok = ok and o.picked_by == i and o.delivered_t is None
            if ok:
                o.delivered_t = w.t
                w.show_cargo(i, False)
        if not ok:
            self.rejected.append({"t": round(w.t, 3), "robot": i, **ev})
        return ok

    def carrying(self, i: int) -> bool:
        """Robot i's payload sensor: it physically carries an order (picked and not yet delivered)."""
        return any(o.picked_by == i and o.delivered_t is None for o in self.orders)

    def drop_carrier(self, i: int, w: World) -> None:
        """A crashed robot loses its load claim: the order must be picked up again (cargo is re-staged)."""
        for o in self.orders:
            if o.picked_by == i and o.delivered_t is None:
                o.picked_by, o.picked_t = None, None
                w.show_cargo(i, False)

    @property
    def all_done(self) -> bool:
        return bool(self.orders) and all(o.delivered_t is not None for o in self.orders)

    def summary(self) -> dict:
        done = [o for o in self.orders if o.delivered_t is not None]
        return {
            "n_tasks": len(self.orders), "n_delivered": len(done),
            "makespan": max((o.delivered_t for o in done), default=None) if len(done) == len(self.orders) and done else None,
            "last_delivery": max((o.delivered_t for o in done), default=None),
            "orders": [{"id": o.task_id, "release": o.release, "pickup": o.pickup, "drop": o.drop,
                        "picked_t": o.picked_t, "delivered_t": o.delivered_t} for o in self.orders],
            "rejected_reports": self.rejected,
        }


def random_blockage_events(cfg: dict, m: WarehouseMap) -> list[dict]:
    """Experiment matrix 'blockage rate': `slots` candidate dynamic events at fixed times; each becomes a pallet on a
    random plain aisle node (never a station or start cell) with probability `rate`.  Seeded; the candidate nodes
    and coin flips are identical across rates (common random numbers), so higher rates add blockages."""
    rb = cfg["faults"].get("random_blockages", {})
    rate = float(rb.get("rate", 0.0))
    if rate <= 0:
        return []
    rng = np.random.default_rng(int(cfg["run"]["seed"]) + 4242)
    aisle = [n for n in m.nodes if m.kind[n] == "aisle" and n not in m.starts.values()]
    out = []
    for k in range(int(rb.get("slots", 10))):
        u = rng.random()
        node = int(rng.choice(aisle))
        if u < rate:
            r, c = m.rc(node)
            out.append({"t": float(rb.get("start", 20.0)) + k * float(rb.get("spacing", 25.0)), "type": "block",
                        "node": [r, c], "duration": float(rb.get("duration", 40.0)), "random": True})
    return out


# ----------------------------------------------------------------------------------------------- server
class SimServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.world = World(cfg)
        self.metrics = Metrics(self.world.n, float(cfg["world"]["d_safe"]))
        self.tasks = TaskBook(cfg, self.world.map)
        self.p = ports(cfg)
        self.ctx = zmq.Context.instance()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(f"tcp://127.0.0.1:{self.p['sim']}")
        self.world_pub = self.ctx.socket(zmq.PUB)
        self.world_pub.setsockopt(zmq.LINGER, 0)
        self.world_pub.bind(f"tcp://127.0.0.1:{self.p['world_pub']}")
        self.events = sorted(list(cfg["faults"]["events"]) + random_blockage_events(cfg, self.world.map),
                             key=lambda e: float(e["t"]))
        self.event_log: list[dict] = []
        self.active_blocks: dict[int, tuple[int, float]] = {}   # block slot -> (node, until)
        self.pending_blocks: list[dict] = []
        self.status: dict[int, dict] = {}
        self.sent: dict[int, dict] = {}
        self.cmd_tick = [-1] * self.world.n
        self.last_cmd_t = [0.0] * self.world.n
        self.traj: list[list[float]] = []
        self.done_flags = [False] * self.world.n
        self.kill_pending: set[int] = set()
        self.agent_stats: dict[int, dict] = {}
        # benchmark-only: the centralized reference planner joins the lockstep clock like a robot (no body)
        self.extra = ["central"] if cfg["fleet"]["planner"] == "central" else []
        self.extra_sent: dict[str, dict] = {}
        self.prof = {"collect": 0.0, "physics": 0.0, "render": 0.0}
        self.on_event = None
        # TEST HOOK ONLY (control-boundary test): extra fields pushed into every robot sense frame
        self.inject_sense = dict(cfg["run"].get("inject_sense", {}))

    # -- agent link ------------------------------------------------------------------------------------------
    def _send(self, i, msg: dict) -> None:
        ident = i if isinstance(i, str) else self.world.names[i]
        self.router.send_multipart([ident.encode(), json.dumps(msg, separators=(",", ":")).encode()])

    def wait_hello(self, timeout: float = 300.0) -> None:
        seen = set()
        t0 = time.time()
        while len(seen) < self.world.n + len(self.extra):
            if time.time() - t0 > timeout:
                raise TimeoutError(f"agents not connected: {set(self.world.names) - seen}")
            if self.router.poll(200):
                ident, raw = self.router.recv_multipart()
                msg = json.loads(raw)
                if msg.get("type") == "hello":
                    nm = ident.decode()
                    seen.add(nm)
                    if nm in self.extra:
                        self.extra_sent[nm] = msg.get("sent", {})
                    else:
                        self.sent[self.world.names.index(nm)] = msg.get("sent", {})

    def _collect(self, tick: int) -> dict:
        want = {i for i in range(self.world.n) if self.world.alive[i]} | set(self.extra)
        got: dict[int, dict] = {}
        timeout = float(self.cfg["run"]["cmd_timeout"])
        t0 = time.time()
        while want - set(got):
            left = timeout - (time.time() - t0)
            if left <= 0:
                break
            if self.router.poll(int(max(1, left * 1000))):
                ident, raw = self.router.recv_multipart()
                msg = json.loads(raw)
                nm = ident.decode()
                i = nm if nm in self.extra else self.world.names.index(nm)
                if msg.get("type") == "cmd" and msg.get("tick") == tick:
                    got[i] = msg
        for i in want - set(got):
            if isinstance(i, str):
                raise RuntimeError(f"lockstep participant {i} stopped responding")
            self._declare_dead(i, "silent")
        return got

    def _declare_dead(self, i: int, why: str) -> None:
        if not self.world.alive[i]:
            return
        self.world.alive[i] = False
        self.world.set_wheels(i, 0.0, 0.0)
        self.tasks.drop_carrier(i, self.world)
        self.world.model.geom_rgba[self.world.stripe_geom[i]] = [0.15, 0.15, 0.15, 1.0]
        self.event_log.append({"t": round(self.world.t, 3), "event": "robot_dead", "robot": i, "why": why})

    # -- events ------------------------------------------------------------------------------------------------
    def _run_events(self) -> list[dict]:
        w = self.world
        out = []
        while self.events and float(self.events[0]["t"]) <= w.t + 1e-9:
            ev = self.events.pop(0)
            if ev["type"] == "block":
                self.pending_blocks.append(ev)
            elif ev["type"] == "kill":
                self.kill_pending.add(int(ev["robot"]))
            elif self.on_event is not None:
                self.on_event(ev)                 # process-level faults owned by the launcher (e.g. dashboard)
            self.event_log.append({"t": round(w.t, 3), "event": "scheduled", **ev})
        still = []
        for ev in self.pending_blocks:
            node = rc_node(w.map, ev["node"])
            free = [k for k in range(len(w.block_mocap)) if k not in self.active_blocks]
            if free and w.node_clear(node):
                k = free[0]
                w.place_block(k, node)
                self.active_blocks[k] = (node, w.t + float(ev.get("duration", 1e9)))
                self.event_log.append({"t": round(w.t, 3), "event": "block_placed", "node": node, "slot": k})
            else:
                still.append(ev)
        self.pending_blocks = still
        for k, (node, until) in list(self.active_blocks.items()):
            if w.t >= until:
                w.remove_block(k)
                del self.active_blocks[k]
                self.event_log.append({"t": round(w.t, 3), "event": "block_removed", "node": node, "slot": k})
        for o in self.tasks.due(w.t):
            out.append({"task_id": o.task_id, "pickup": o.pickup, "dropoff": o.drop, "release": o.release})
            self.event_log.append({"t": round(w.t, 3), "event": "order", "task_id": o.task_id})
        return out

    # -- viewer overlays ---------------------------------------------------------------------------------------
    def _snapshot(self, rtf: float) -> dict:
        """Everything the viewer thread needs, copied on the sim thread (no shared mutable state)."""
        w = self.world
        left = [f"t = {w.t:7.1f} s   mode = {self.cfg['run']['mode']}   x{rtf:4.1f}",
                f"planner = {self.cfg['fleet']['planner']}"
                + ("  +learned" if self.cfg["fleet"].get("learned_priority") else ""),
                f"delivered {sum(o.delivered_t is not None for o in self.tasks.orders)}/{len(self.tasks.orders)}"
                f"   collisions {self.metrics.eq8_events + self.metrics.robot_contacts}"]
        viz = []
        for i in range(w.n):
            st = self.status.get(i, {})
            alive = "" if w.alive[i] else " [DEAD]"
            left.append(f"{w.names[i]}{alive}: {st.get('hud', '')}  bat {100 * w.battery[i]:.0f}%")
            if w.alive[i]:
                viz.append((i, list(st.get("viz_path") or []), list(st.get("viz_held") or [])))
        return {"qpos": w.data.qpos.copy(), "mocap": w.data.mocap_pos.copy(), "time": w.t,
                "text": "\n".join(left), "viz": viz}

    def _publish_world(self) -> None:
        w = self.world
        msg = {"type": "world", "t": round(w.t, 3), "alive": w.alive,
               "poses": [list(w.pose(i)) for i in range(w.n)], "battery": [round(float(b), 4) for b in w.battery],
               "blocks": [n for n, _ in self.active_blocks.values()],
               "tasks": {"delivered": sum(o.delivered_t is not None for o in self.tasks.orders), "total": len(self.tasks.orders)},
               "orders": [{"task_id": o.task_id, "pickup": o.pickup, "dropoff": o.drop, "release": o.release,
                           "picked": o.picked_by is not None, "delivered": o.delivered_t is not None}
                          for o in self.tasks.orders if o.announced],
               "metrics": {"collisions": self.metrics.eq8_events + self.metrics.robot_contacts,
                           "min_pair_dist": None if math.isinf(self.metrics.min_pair_dist) else round(self.metrics.min_pair_dist, 3)}}
        self.world_pub.send(json.dumps(msg).encode(), zmq.NOBLOCK)

    # -- main loop ---------------------------------------------------------------------------------------------
    def run(self) -> dict:
        cfg, w = self.cfg, self.world
        demo = cfg["run"]["mode"] == "demo"
        duration = float(cfg["run"]["duration"])
        self.wait_hello()
        wall0 = time.time()
        last_sync = 0.0
        tick = 0
        aborted = False
        sync_dt = 1.0 / float(cfg["run"]["viewer_fps"])
        d_vis = mujoco.MjData(w.model)
        d_vis.qpos[:] = w.data.qpos
        mujoco.mj_kinematics(w.model, d_vis)
        with mujoco.viewer.launch_passive(w.model, d_vis) as viewer:
            vt = ViewerThread(w.model, viewer, d_vis)
            viewer.cam.lookat[:] = [*(np.array(w.map.extent()) / 2), 0.0]
            viewer.cam.distance = max(w.map.extent()) * 1.15
            viewer.cam.elevation = -60
            viewer.cam.azimuth = 90
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[4] = 0
            while True:
                if not viewer.is_running():
                    aborted = True
                    break
                orders = self._run_events()
                for i in sorted(self.kill_pending):
                    if w.alive[i]:
                        self._send(i, {"type": "kill"})
                        self._declare_dead(i, "fault_injection")
                self.kill_pending.clear()
                expect = {w.names[j]: self.sent[j] for j in self.sent if w.alive[j]}
                expect.update(self.extra_sent)
                for nm in self.extra:
                    self._send(nm, {"type": "sense", "tick": tick, "t": round(w.t, 4), "orders": orders,
                                    "expect": {k: v for k, v in expect.items() if k != nm},
                                    "alive": [w.names[j] for j in range(w.n) if w.alive[j]]})
                for i in range(w.n):
                    if not w.alive[i]:
                        continue
                    v, om = w.twist(i)
                    self._send(i, {"type": "sense", "tick": tick, "t": round(w.t, 4), "pose": w.sensed_pose(i),
                                   "vel": [round(v, 4), round(om, 4)], "ranges": w.ranges(i),
                                   "battery": round(float(w.battery[i]), 5), "orders": orders,
                                   "load": self.tasks.carrying(i),
                                   "expect": {k: v for k, v in expect.items() if k != w.names[i]},
                                   **self.inject_sense})
                tp0 = time.perf_counter()
                cmds = self._collect(tick)
                tp1 = time.perf_counter()
                self.prof["collect"] += tp1 - tp0
                self.sent = {}
                for nm in self.extra:
                    self.extra_sent[nm] = cmds.pop(nm).get("sent", {})
                for i, c in cmds.items():
                    if w.battery[i] > 0:
                        w.set_wheels(i, float(c["wl"]), float(c["wr"]))
                    self.sent[i] = c.get("sent", {})
                    self.status[i] = c.get("status", {})
                    self.done_flags[i] = bool(c.get("done", False))
                    if c.get("stats"):
                        self.agent_stats[i] = c["stats"]
                    for ev in c.get("task_events", []):
                        ok = self.tasks.report(w, i, ev)
                        self.event_log.append({"t": round(w.t, 3), "event": "task_report", "robot": i, "ok": ok, **ev})
                tp2 = time.perf_counter()
                w.step(self.metrics)
                self.metrics.tick(w)
                self.prof["physics"] += time.perf_counter() - tp2
                if tick % max(1, int(cfg["run"]["control_hz"]) // 10) == 0:
                    self.traj.append([round(w.t, 3)] + [round(v, 4) for i in range(w.n) for v in w.pose(i)])
                    self._publish_world()
                tick += 1
                wall = time.time() - wall0
                if demo:
                    lag = w.t - wall
                    if lag > 0:
                        time.sleep(lag)
                if demo or wall - last_sync >= sync_dt:
                    tp3 = time.perf_counter()
                    vt.post(self._snapshot(w.t / max(wall, 1e-6)))
                    self.prof["render"] += time.perf_counter() - tp3
                    last_sync = wall
                alive_ids = [i for i in range(w.n) if w.alive[i]]
                if w.t >= duration:
                    break
                if cfg["run"]["stop_when_done"]:
                    if self.tasks.orders and self.tasks.all_done:
                        break
                    if not self.tasks.orders and alive_ids and all(self.done_flags[i] for i in alive_ids):
                        break
            # final frame
            vt.post(self._snapshot(w.t / max(time.time() - wall0, 1e-6)))
            time.sleep(0.3)
            vt.stop()
            self.prof["frames"] = vt.frames
        for i in range(w.n):
            if w.alive[i]:
                self._send(i, {"type": "stop"})
        for nm in self.extra:
            self._send(nm, {"type": "stop"})
        wall = time.time() - wall0
        result = {
            "run": cfg["run"]["name"], "seed": cfg["run"]["seed"], "config_hash": config_hash(cfg),
            "planner": cfg["fleet"]["planner"], "learned_priority": bool(cfg["fleet"].get("learned_priority")),
            "n_robots": w.n, "sim_time": round(w.t, 3), "wall_time": round(wall, 2),
            "real_time_factor": round(w.t / max(wall, 1e-6), 2), "aborted": aborted, "ticks": tick,
            "server_profile_s": {k: round(v, 3) for k, v in self.prof.items()},
            "alive": w.alive, "battery": [round(float(b), 4) for b in w.battery],
            "done_flags": self.done_flags,
            "final_poses": [list(map(lambda v: round(v, 4), w.pose(i))) for i in range(w.n)],
            "safety": self.metrics.summary(), "tasks": self.tasks.summary(),
            "agents": {w.names[i]: s for i, s in self.agent_stats.items()},
            "events": self.event_log,
        }
        self._save(result)
        self.router.close()
        self.world_pub.close()
        return result

    def _save(self, result: dict) -> None:
        out = resolve(self.cfg["run"]["results_dir"]) / self.cfg["run"]["name"] / f"seed_{self.cfg['run']['seed']}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(self.cfg, indent=2))
        (out / "result.json").write_text(json.dumps(result, indent=2))
        np.save(out / "traj.npy", np.array(self.traj, dtype=np.float64))
        result["out_dir"] = str(out)
