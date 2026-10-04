"""Per-robot agent brain (design doc Fig. 3): comms adapter -> peer table -> task layer -> planner (A*/PIBT)
-> (learning advisor) -> safety shield -> motion controller.  Everything here runs inside one robot's process
and uses only its own sensors plus peer messages.

Execution loop per control tick (design doc §10.6):
  1. build the local view of peer state (delivery barrier + fault injection happen in PeerNode.sync)
  2. predict occupancy conflicts (held / claimed / intended nodes)
  3. derive priority (task age, deadlock escalation, optional learned advisor)
  4. PIBT-style claim / push / backtrack  (or stop-and-wait baseline)
  5. safety shield
  6. one control interval of wheel commands
  7. publish state and intent (5-10 Hz)
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import zmq

from fleet.agent.coord import PUSH_ACKED_TIMEOUT, PUSH_TIMEOUT, LockManager, PeerView, PIBTPolicy, StopWaitPolicy, rank, straight_bonus
from fleet.agent.blockage import BlockageDetector
from fleet.agent.deadlock import BOOST, DeadlockMonitor
from fleet.agent.goals import GoalSequence
from fleet.agent.isolation import IsolationMonitor
from fleet.agent.motion import MotionController, MotionParams, twist_to_wheels
from fleet.agent.safety_shield import SafetyParams, SafetyShield
from fleet.common.config import ports, resolve
from fleet.comms.peer_node import PeerNode
from fleet.comms.peer_table import LOST, PeerTable
from fleet.world.mapgrid import WarehouseMap
from fleet.world.mjcf_builder import robot_name
from fleet.world.robot_mjcf import MIR

STATE_TTL = 1.0


class FleetBrain:
    def __init__(self, cfg: dict, idx: int, name: str, wmap: WarehouseMap, ctx: zmq.Context):
        self.cfg, self.idx, self.name, self.map = cfg, idx, name, wmap
        n = int(cfg["fleet"]["n_robots"])
        names = [robot_name(i) for i in range(n)]
        pp = ports(cfg)
        c = cfg["comms"]
        pubs, rtrs, net_names = pp["robot_pub"], pp["robot_router"], names
        if cfg["fleet"]["planner"] == "central":      # benchmark only: also listen to the central planner
            from fleet.benchmark.central import central_names
            net_names, pubs, rtrs = central_names(cfg)
        trace = None
        if c.get("trace"):
            out = resolve(cfg["run"]["results_dir"]) / cfg["run"]["name"] / f"seed_{cfg['run']['seed']}"
            out.mkdir(parents=True, exist_ok=True)
            trace = str(out / f"comms_{name}.csv")
        self.net = PeerNode(ctx, name, net_names, pubs, rtrs, latency=float(c["latency"]),
                            jitter=float(c["jitter"]), loss=float(c["loss"]), seed=int(cfg["run"]["seed"]),
                            burst=c.get("burst") or None, disconnect=c.get("disconnect") or None,
                            reorder_pub=bool(c.get("reorder_pub", False)), trace_path=trace)
        self.table = PeerTable([p for p in names if p != name], float(c["heartbeat_timeout"]), float(c["lost_timeout"]))
        self.central_plan: dict | None = None
        self.cstep = 0
        self._central_target: int | None = None
        self.view = PeerView(self.table)
        self.iso = IsolationMonitor(timeout=float(c.get("isolation_timeout", 30.0)), hold=float(c["heartbeat_timeout"]))
        self.load: bool | None = None       # own payload sensor (None: not provided by the world)
        self.lock = LockManager(name, self._send, self._broadcast,
                                bypass=bool(cfg["faults"].get("bypass_claims", False)))
        self.planner = cfg["fleet"]["planner"]
        rng = np.random.default_rng(int(cfg["run"]["seed"]) * 101 + idx)
        tb = rng.random(max(wmap.pos) + 1)
        self.pibt = PIBTPolicy(wmap, name, lambda v: float(tb[v]))
        self.sw = StopWaitPolicy(wmap, name, float(cfg["planning"]["stopwait_timeout"]))
        self.dt = 1.0 / float(cfg["run"]["control_hz"])
        self.motion = MotionController(MotionParams.from_cfg(cfg["motion"]), self.dt)
        self.shield = SafetyShield(SafetyParams.from_cfg(cfg["safety"]))
        pl = cfg["planning"]
        self.dl = DeadlockMonitor(window=float(pl["deadlock_window"]), eps=0.5, boost_time=8.0)
        self.state_every = max(1, int(round(float(cfg["run"]["control_hz"]) / float(cfg["run"]["state_hz"]))))
        self.barrier_timeout = float(cfg["run"]["barrier_timeout"])
        self.goals = self._make_goal_source()
        self.advisor = None
        if cfg["fleet"].get("learned_priority"):
            from fleet.agent.advisor import LearnedPriorityAdvisor
            self.advisor = LearnedPriorityAdvisor(cfg)
        self.optb = None                              # v3 Option B congestion-aware advisor (one per robot)
        if cfg["fleet"].get("priority_advisor", "none") != "none":
            from fleet.agent.optb_advisor import CongestionAdvisor
            self.optb = CongestionAdvisor(cfg, cfg["fleet"]["priority_advisor"])
        self.last_acquire_t = 0.0                     # Option B features (instrumentation, no effect on decisions)
        self.conflict_times: list[float] = []
        self._feat_log = None
        if cfg["fleet"].get("log_features"):
            out = resolve(cfg["run"]["results_dir"]) / cfg["run"]["name"] / f"seed_{cfg['run']['seed']}"
            out.mkdir(parents=True, exist_ok=True)
            self._feat_log = open(out / f"optb_{name}.jsonl", "w", buffering=1 << 16)
        # kinematic / coordination state
        self.node_at = wmap.starts[idx]
        self.lock.held = {self.node_at}
        self.prev: int | None = None
        self.move: dict | None = None
        self.dwell_until: float | None = None
        self.pushes: dict[str, dict] = {}
        self.waiting_push: dict | None = None
        self.goal_since = 0.0
        self._goal_key = None
        self.blocked: dict[int, float] = {}          # node -> expiry (blockage events, route invalidation)
        self.blockage = BlockageDetector(wmap)
        self.blockage_ttl = float(cfg["planning"]["blockage_ttl"])
        self.my_blockages: dict[int, float] = {}     # nodes I detected -> expiry (re-published periodically)
        self._blk_seq = 0
        self._blk_last_pub = -1e9
        self.blocked_ever: dict[int, str] = {}       # node -> how we learned it ("sensed" | peer name)
        self._dist_cache: dict = {}
        self.tick = 0
        self.now = 0.0
        self.pose = (0.0, 0.0, 0.0)
        self.last_reason = ""
        self.prio = -1.0
        self.task_events: list[dict] = []
        self.decide_time = 0.0
        self.decide_max = 0.0
        self.publish_count = 0
        self.wait_time = 0.0
        self._last_claim_node: int | None = None
        self.compute_samples: list[float] = []
        self._battery = 1.0

    # ------------------------------------------------------------------------------------------------ plumbing
    def _make_goal_source(self):
        if self.cfg["fleet"]["planner"] == "central":
            from fleet.agent.goals import CentralGoals
            return CentralGoals(self.cfg, self.name, self.map)
        if self.cfg.get("goals"):
            return GoalSequence(self.cfg, self.name, self.map)
        from fleet.agent.task_manager import TaskManager
        return TaskManager(self.cfg, self.name, self.map, self)

    def _send(self, peer: str, mtype: str, ttl: float, **body) -> None:
        self.net.send(peer, mtype, self.now, ttl, **body)

    def _broadcast(self, peers: list[str], mtype: str, ttl: float, **body) -> None:
        for p in peers:
            self.net.send(p, mtype, self.now, ttl, **body)

    def join_step(self) -> bool:
        return self.net.join_step(20)

    def live_peers(self) -> list[str]:
        return self.table.live_names()

    def close(self) -> None:
        self.net.close()
        if self._feat_log is not None:
            self._feat_log.close()

    # ------------------------------------------------------------------------------------------------ helpers
    def blocked_set(self) -> set[int]:
        b = {v for v, until in self.blocked.items() if until > self.now}
        b |= self.view.lost_nodes()
        b.discard(self.node_at)
        return b

    def dist_field(self, goal: int | None) -> dict[int, int] | None:
        if goal is None:
            return None
        key = (goal, frozenset(self.blocked_set()))
        d = self._dist_cache.get(key)
        if d is None:
            if len(self._dist_cache) > 64:
                self._dist_cache.clear()
            d = self.map.bfs_dist(goal, key[1])
            if not d:                       # goal itself blocked: keep the static field so we can wait nearby
                d = self.map.bfs_dist(goal)
            self._dist_cache[key] = d
        return d

    def preview_path(self, start: int, dist: dict[int, int] | None, k: int = 4) -> list[int]:
        if dist is None:
            return []
        out, u = [], start
        for _ in range(k):
            nb = [v for v in self.map.adj[u] if dist.get(v, 1e9) < dist.get(u, 1e9)]
            if not nb:
                break
            u = min(nb, key=lambda v: (dist[v], v))
            out.append(u)
        return out

    def hops_to(self, a: int, b: int) -> int:
        if a == b:
            return 0
        if b in self.map.adj[a]:
            return 1
        return 2 if any(b in self.map.adj[x] for x in self.map.adj[a]) else 3

    def peer_nodes(self) -> dict[str, tuple[int, dict]]:
        return {e.name: (e.state.get("node"), e.state) for e in self.table.known()}

    # ------------------------------------------------------------------------------------------------ messages
    def _dispatch(self, m: dict) -> None:
        t = m["type"]
        if t in ("RobotState", "RobotIntent"):
            if self.table.update(m, self.now) and t == "RobotState":
                self.lock.on_peer_state(m["src"], m.get("held", []))
                if hasattr(self.goals, "on_peer_done"):
                    self.goals.on_peer_done(m.get("done", []), self.now)
            return
        self.table.heard(m["src"], self.now)
        if t == "CentralPlan":
            if self.central_plan is None or m["step"] > self.central_plan["step"]:
                self.central_plan = m
            return
        if t == "Claim":
            busy = set()
            if self.move is not None:
                busy.add(self.move["to"])
                if self.move.get("next") is not None:
                    busy.add(self.move["next"])
            self.lock.on_claim(m, self.claim_prio(), self.now, busy)
        elif t == "ClaimReply":
            self.lock.on_reply(m)
        elif t == "Release":
            self.lock.on_release(m)
        elif t == "Push":
            self._on_push(m)
        elif t == "PushReply":
            wp = self.waiting_push
            if wp is not None and m["push_id"] == wp["push_id"]:
                if m["ok"]:
                    wp["acked"] = True
                else:
                    self._note_conflict()
                    self.pibt.mark_failed(wp["node"], self.now)
                    self.waiting_push = None
        elif t == "BlockageEvent":
            self._on_blockage(m)
        else:
            self.goals.handle(m, self.now)

    def _on_push(self, m: dict) -> None:
        v = m["node"]
        reply = None
        if self.dwell_until is not None:
            reply = False
        elif self.move is not None and self.move["frm"] == v and self.move["to"] != m["from_node"]:
            reply = True                                  # already leaving the node
        elif v != self.node_at and v not in self.lock.held:
            reply = True                                  # not there (stale view)
        if reply is not None:
            self._send(m["src"], "PushReply", PUSH_TIMEOUT, push_id=m["push_id"], ok=reply)
            return
        if m["push_id"] not in self.pushes:
            # acknowledge at once: the pusher then waits for the (possibly chained) evasion instead of timing out
            self._send(m["src"], "PushReply", PUSH_TIMEOUT, push_id=m["push_id"], ok=True)
        self.pushes[m["push_id"]] = {**m, "deadline": self.now + PUSH_ACKED_TIMEOUT}

    def _answer_pushes(self, ok: bool) -> None:
        for pid, p in self.pushes.items():
            self._send(p["src"], "PushReply", PUSH_TIMEOUT, push_id=pid, ok=ok)
        self.pushes.clear()

    def _sense_blockages(self, ranges: list[float]) -> None:
        active = {v for v, until in self.blocked.items() if until > self.now}
        centres = []
        for e in self.table.known():
            px, py, pth = e.state["pose"]
            centres.append((px + MIR.wheel_dx * math.cos(pth), py + MIR.wheel_dx * math.sin(pth)))
        new_b, new_c = self.blockage.update(self.pose, ranges, centres, active, self.now)
        for n in new_b:
            self.blocked_ever.setdefault(n, "sensed")
            self.blocked[n] = self.now + self.blockage_ttl
            self.my_blockages[n] = self.now + self.blockage_ttl
            self._pub_blockage(n, True)
        for n in new_c:
            self.blocked.pop(n, None)
            self.my_blockages.pop(n, None)
            self._pub_blockage(n, False)
        if self.now - self._blk_last_pub >= 2.0:          # repeat own detections (robust to message loss)
            self._blk_last_pub = self.now
            self.my_blockages = {n: u for n, u in self.my_blockages.items() if u > self.now}
            for n, until in self.my_blockages.items():
                self._pub_blockage(n, True, until - self.now)
        # a move whose target became blocked is aborted: drive back into the node we still hold
        mv = self.move
        if mv is not None and mv["to"] in self.blocked_set() and mv.get("aborted") is None:
            self.move = {"frm": mv["to"], "to": mv["frm"], "start": tuple(self.pose[:2]), "next": None, "aborted": True}
            if mv.get("next") is not None:
                self.lock.release([mv["next"]])

    def _pub_blockage(self, node: int, blocked: bool, ttl: float | None = None) -> None:
        self._blk_seq += 1
        self.net.publish("BlockageEvent", self.now, self.blockage_ttl if ttl is None else ttl, node=node,
                         source=self.name, severity=1.0, blocked=blocked, event_id=f"{self.name}:{self._blk_seq}")

    def _on_blockage(self, m: dict) -> None:
        if m["blocked"]:
            self.blocked_ever.setdefault(m["node"], m["src"])
            self.blocked[m["node"]] = max(self.blocked.get(m["node"], 0.0), m["ts"] + m["ttl"])
        else:
            self.blocked.pop(m["node"], None)

    # ------------------------------------------------------------------------------------------------ priority
    def base_priority(self, goal: int | None, dist: dict | None) -> float:
        """PIBT priority key.  Time-invariant while the goal is unchanged so that every robot compares identical
        numbers regardless of message age: earlier goal assignment = higher priority (equivalent to PIBT's
        'elapsed time since last goal').  Idle robots rank below every robot with a goal."""
        if goal is None:
            return -1e9
        p = -self.goal_since
        if self.advisor is not None and dist is not None:
            p += self.advisor.score(self, goal, dist)
        if self.optb is not None and dist is not None:
            p += self.optb.score(self, goal, dist)      # bounded: 0 <= adjustment <= ADJ_MAX
        return p

    def claim_prio(self) -> float:
        inh = max((float(p["priority"]) for p in self.pushes.values()), default=-1e18)
        return max(self.prio, inh)

    # ------------------------------------------------------------------------------------------------ main step
    def step(self, sense: dict) -> dict:
        self.now = float(sense["t"])
        self.pose = tuple(sense["pose"])
        self._battery = float(sense.get("battery", 1.0))
        self.load = sense.get("load")
        msgs = self.net.sync(self.now, sense.get("expect"), self.barrier_timeout)
        t_start = time.perf_counter()          # compute budget excludes waiting on the network barrier
        for m in msgs:
            self._dispatch(m)
        newly_lost = self.table.refresh(self.now)
        if self.planner != "central":
            act = self.iso.update(self.now, self.table)
            if act == "enter":
                self._enter_isolation()
            elif act == "resume":
                self._resume_after_isolation()
        for name in newly_lost:
            self.goals.on_peer_lost(name, self.now)
        for pid in [k for k, p in self.pushes.items() if p["deadline"] < self.now]:
            del self.pushes[pid]
        if hasattr(self.goals, "on_orders"):
            self.goals.on_orders(sense.get("orders", []), self.now)
        self._sense_blockages(sense["ranges"])
        self.goals.tick(self.now, self)
        self._advance_move()
        goal = self.goals.current()
        if self.goals.goal_key() != self._goal_key:
            self._goal_key = self.goals.goal_key()
            self.goal_since = self.now
            self.last_acquire_t = self.now
            self.dl.reset_goal(self.now, self._goal_key)
            self.pibt.failed_until.clear()
        dist = self.dist_field(goal)
        self.prio = self.base_priority(goal, dist)
        # deadlock monitor (only while stationary-capable decisions matter)
        near = [st for nm, (nd, st) in self.peer_nodes().items() if nd is not None and self.hops_to(self.node_at, nd) <= 2]
        higher_near = any(rank(float(st.get("prio", -1e9)), st["src"]) > rank(self.prio, self.name) and st.get("stalled")
                          for st in near)
        boosted_near = any(st.get("boosted") for st in near)
        busy = (self.move is not None and self.move.get("yield")) or (self.waiting_push or {}).get("acked")
        if busy:
            self.dl.last_improve = self.now
        if self.dl.update(self.now, None if goal is None or self.node_at == goal else float(dist.get(self.node_at, 1e6)),
                          higher_near, boosted_near) and self.dl.wants_invalidation:
            self._invalidate_route(goal, dist)
        if self.dl.boosted(self.now):
            self.prio += BOOST
        self._dwell_logic(goal)
        self._coordinate(goal, dist)
        v0, w0 = self._motion(goal)
        v, w, self._shield_why = self.shield.filter(v0, w0, sense["ranges"], self._peers_body())
        self._cmd_dbg = (round(v0, 2), round(w0, 2), round(v, 2), round(w, 2))
        wl, wr = twist_to_wheels(v, w)
        if self.tick % self.state_every == 0:
            self._publish(goal, dist)
        el = time.perf_counter() - t_start
        self.decide_time += el
        self.decide_max = max(self.decide_max, el)
        self.compute_samples.append(el)
        if self._feat_log is not None and int(sense["tick"]) % 10 == 0:
            self._log_features(int(sense["tick"]), goal, dist)    # outside the measured compute budget
        if self.move is None and goal is not None and self.dwell_until is None and self.node_at != goal:
            self.wait_time += self.dt
        self.tick += 1
        if self.cfg["run"].get("trace") and self.tick % int(self.cfg["run"].get("trace_every", 10)) == 0:
            mv = self.move
            print(f"t={self.now:7.2f} at={self.map.rc(self.node_at)} mv={None if mv is None else (self.map.rc(mv['frm']), self.map.rc(mv['to']), mv.get('next') and self.map.rc(mv['next']))} "
                  f"held={[self.map.rc(v) for v in sorted(self.lock.held)]} pend={self.lock.pending and self.map.rc(self.lock.pending.node)} "
                  f"wp={self.waiting_push and (self.waiting_push['peer'], self.map.rc(self.waiting_push['node']))} pushes={len(self.pushes)} "
                  f"prio={self.prio:.2f} goal={goal and self.map.rc(goal)} why={self.last_reason} cmd={self._cmd_dbg} sh={self._shield_why} pose={[round(p, 2) for p in self.pose]} "
                  f"peers={[(nm, self.map.rc(nd) if nd is not None else None, round(float(st.get('prio', 0)), 1)) for nm, (nd, st) in self.peer_nodes().items()]}",
                  flush=True)
        events, self.task_events = self.task_events, []
        out = {"wheels": (wl, wr), "sent": self.net.sent_counters(), "task_events": events,
               "done": self.goals.finished and self.move is None and self.dwell_until is None,
               "status": self._status(goal, dist)}
        if self.tick % 20 == 0 or out["done"] or events or self.dl._ep is not None or self.iso.blocks():
            out["stats"] = self.stats()
        return out

    # ------------------------------------------------------------------------------------------------ motion state
    def _advance_move(self) -> None:
        mv = self.move
        if mv is None:
            return
        x, y, _ = self.pose
        tx, ty = self.map.pos[mv["to"]]
        sx, sy = mv["start"]
        L = math.hypot(tx - sx, ty - sy)
        prog = ((x - sx) * (tx - sx) + (y - sy) * (ty - sy)) / max(L, 1e-9)
        through = mv.get("next") is not None and straight_bonus(self.map, mv["frm"], mv["to"], mv["next"]) == 0
        arrived = (through and prog >= L - 0.15) or mv.get("arrived", False)
        if not arrived:
            return
        self.lock.release([mv["frm"]])
        self.prev, self.node_at = mv["frm"], mv["to"]
        for v in mv.get("yield", []):
            self.pibt.mark_failed(v, self.now, 5.0)
        if mv.get("next") is not None:
            self.move = {"frm": mv["to"], "to": mv["next"], "start": self.map.pos[mv["to"]], "next": None}
            self._answer_pushes(True)
        else:
            self.move = None
            if self._central_target is not None:
                self.cstep, self._central_target = self._central_target, None

    def _start_move(self, v: int) -> None:
        self.move = {"frm": self.node_at, "to": v, "start": self.map.pos[self.node_at], "next": None}
        self.waiting_push = None
        if self.pushes:
            # asynchronous counterpart of PIBT's "pusher moves into the vacated node in the same step": the
            # vacated node is left to the pusher (not re-claimed by us) until 5 s after we reach the evasion node
            self.move["yield"] = [p["node"] for p in self.pushes.values()]
            for v in self.move["yield"]:
                self.pibt.mark_failed(v, self.now, 60.0)
            self._answer_pushes(True)

    def _dwell_logic(self, goal: int | None) -> None:
        if self.dwell_until is not None:
            if self.now >= self.dwell_until:
                self.dwell_until = None
                self.task_events += self.goals.on_dwell_done(self.now)
            return
        if goal is not None and self.move is None and self.node_at == goal and self.lock.pending is None:
            x, y, _ = self.pose
            gx, gy = self.map.pos[goal]
            if math.hypot(x - gx, y - gy) < 0.2 and abs(self.motion.v) < 0.05:
                self.dwell_until = self.now + self.goals.on_reached(self.now)
                self.task_events += getattr(self.goals, "drain_events", lambda: [])()
                if self.pushes:
                    self._answer_pushes(False)

    def _note_conflict(self) -> None:
        self.conflict_times.append(self.now)
        if len(self.conflict_times) > 64:
            del self.conflict_times[:-32]

    def _log_features(self, tick: int, goal: int | None, dist: dict | None) -> None:
        """Option B dataset row (2 Hz): decision-time features + label ingredients.  Read-only w.r.t. decisions."""
        from fleet.learning.congestion import N_FEATURES, features_from_snapshot, snapshot
        mv = self.move
        path = self.preview_path(self.node_at if mv is None else mv["to"], dist) if dist is not None else []
        if mv is not None:
            path = [mv["to"]] + path
        has_goal = goal is not None and dist is not None
        x = features_from_snapshot(snapshot(self, dist)) if has_goal else [0.0] * N_FEATURES
        row = {"tick": tick, "t": round(self.now, 3), "node": self.node_at, "held": sorted(self.lock.held),
               "waiting": bool(mv is None and goal is not None and self.dwell_until is None and self.node_at != goal),
               "path": path[:4], "has_goal": has_goal, "goal_since": round(self.goal_since, 3),
               "prio": round(self.prio, 4), "x": [round(float(v), 5) for v in x]}
        self._feat_log.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _enter_isolation(self) -> None:
        """All peers went silent at once: assume our own link is down (preflight P4)."""
        self.lock.cancel()
        self.waiting_push = None
        if self.pushes:
            self._answer_pushes(False)

    def _resume_after_isolation(self) -> None:
        """Reconnected and held one heartbeat period: release stale nodes, reconcile tasks, resume."""
        keep = {self.node_at}
        if self.move is not None:
            keep |= {self.move["to"]} | ({self.move["next"]} if self.move.get("next") is not None else set())
        stale = sorted(self.lock.held - keep)
        self.lock.held = keep
        if stale:
            self._broadcast(self.live_peers(), "Release", 1.0, nodes=stale)
        self.iso.note("released_nodes", stale)
        if hasattr(self.goals, "on_reconnect"):
            restored, reopened = self.goals.on_reconnect(self.now)
            self.iso.note("tasks_restored", restored)
            self.iso.note("tasks_reopened", reopened)

    def _invalidate_route(self, goal: int | None, dist: dict | None) -> None:
        """Deadlock ladder step 2: temporarily treat the contested next node as blocked to force a detour."""
        nxt = self.preview_path(self.node_at, dist, 1)
        if nxt and nxt[0] != goal:
            self.blocked[nxt[0]] = self.now + 10.0
            self.dl.escalations_this_goal = 0
            self.dl.note_reroute(self.now)

    # ------------------------------------------------------------------------------------------------ coordination
    def _central_follow(self) -> None:
        """BENCHMARK ONLY: execute the step assigned by the central reference planner (no claims, no pushes)."""
        plan = self.central_plan
        if plan is None or self.move is not None or plan["step"] <= self.cstep:
            return
        entry = plan["assign"].get(self.name)
        if entry is None:
            self.cstep = plan["step"]
            return
        self.goals.set_plan(entry)
        nxt = int(entry["next"])
        if nxt == self.node_at:
            self.cstep = plan["step"]
        elif self.dwell_until is None:
            self.lock.held.add(nxt)
            self._central_target = plan["step"]
            self._start_move(nxt)

    def _coordinate(self, goal: int | None, dist: dict | None) -> None:
        if self.planner == "central":
            self._central_follow()
            return
        r = self.lock.poll(self.now, self.live_peers())
        if r == "acquired":
            self.last_acquire_t = self.now
            got = self._last_claim_node
            if dist is not None and dist.get(got, 1e9) < dist.get(self.node_at, 1e9):
                self.dl.commit_progress(self.now)
            if self.move is not None and got in self.map.adj[self.move["to"]] and self.move.get("next") is None:
                self.move["next"] = got
            elif self.move is None and got in self.map.adj[self.node_at]:
                self._start_move(got)
            else:
                self.lock.release([got])
        elif r == "failed":
            self._note_conflict()
            self.pibt.mark_failed(self._last_claim_node, self.now, 0.5 if self.planner == "pibt" else 0.0)
        if self.dwell_until is not None:
            if self.pushes:
                self._answer_pushes(False)       # dwelling at a station: cannot be pushed
            return
        if self.lock.pending is not None:
            return
        if self.iso.blocks():
            # isolated / partitioned / reconnecting: no new claims, pushes or look-ahead; a granted move completes
            if self.pushes:
                self._answer_pushes(False)
            self.last_reason = "isolated:" + self.iso.state
            return
        if self.move is not None:
            self._lookahead(goal, dist)
            return
        if self.planner == "central":
            return
        if self.planner == "pibt":
            self._pibt_decide(goal, dist)
        else:
            d = self.sw.decide(self.now, self.node_at, goal, self.view, self.blocked_set())
            self.last_reason = d.reason
            if d.kind == "claim":
                self._claim(d.node)
            if self.pushes:                      # baseline never yields to pushes (it does not push either)
                self._answer_pushes(False)

    def _claim(self, v: int) -> None:
        self._last_claim_node = v
        self.lock.claim(v, self.claim_prio(), self.live_peers(), self.now)

    def _pibt_decide(self, goal: int | None, dist: dict | None) -> None:
        wp = self.waiting_push
        if wp is not None:
            if self.view.holder(wp["node"]) is None:
                self.waiting_push = None
                self._claim(wp["node"])
            elif self.now - wp["t0"] > (PUSH_ACKED_TIMEOUT if wp.get("acked") else PUSH_TIMEOUT):
                self._note_conflict()
                self.pibt.mark_failed(wp["node"], self.now)
                self.waiting_push = None
            return
        pushes = list(self.pushes.values())
        # fault injection: with claims bypassed the planner is also blind to peer occupancy (forced convergence)
        view = self.view if not self.lock.bypass else PeerView(PeerTable([]))
        d = self.pibt.decide(self.now, self.node_at, self.prev, dist, self.claim_prio(), view,
                             self.blocked_set(), pushes, self.dwell_until is not None)
        self.last_reason = d.kind + (":" + d.reason if d.reason else "")
        if d.kind == "claim":
            self._claim(d.node)
        elif d.kind == "push":
            pid = f"{self.name}:p{self.tick}"
            chain = sorted({self.name} | set().union(*[set(p["chain"]) for p in pushes])) if pushes else [self.name]
            path = self.preview_path(self.node_at, dist) if dist is not None else []
            self._send(d.peer, "Push", PUSH_TIMEOUT, node=d.node, priority=self.claim_prio(), from_node=self.node_at,
                       chain=chain, push_id=pid, path=[self.node_at] + path)
            self.waiting_push = {"peer": d.peer, "node": d.node, "push_id": pid, "t0": self.now}
        elif d.kind == "fail_push":
            self._answer_pushes(False)

    def _lookahead(self, goal: int | None, dist: dict | None) -> None:
        """While approaching the next node, claim the following straight node to pass through without stopping."""
        mv = self.move
        if mv.get("next") is not None or self.pushes or goal is None or dist is None or mv["to"] == goal:
            return
        x, y, _ = self.pose
        tx, ty = self.map.pos[mv["to"]]
        if math.hypot(tx - x, ty - y) > 1.0:
            return
        u = mv["to"]
        if self.planner == "pibt":
            cands = [v for v in self.map.adj[u] if dist.get(v, 1e9) < dist.get(u, 1e9) and v not in self.blocked_set()
                     and straight_bonus(self.map, mv["frm"], u, v) == 0]
        else:
            nxt = self.sw.next_nodes(u)
            cands = [nxt[0]] if nxt and straight_bonus(self.map, mv["frm"], u, nxt[0]) == 0 else []
            if cands and self.sw.decide(self.now, u, goal, self.view, self.blocked_set()).kind != "claim":
                cands = []
        for v in cands:
            if (self.lock.bypass or self.view.holder(v) is None) and self.pibt.failed_until.get(v, -1) <= self.now:
                self._claim(v)
                return

    # ------------------------------------------------------------------------------------------------ motion
    def _motion(self, goal: int | None) -> tuple[float, float]:
        mv = self.move
        if mv is not None:
            through = mv.get("next") is not None and straight_bonus(self.map, mv["frm"], mv["to"], mv["next"]) == 0
            (v, w), arrived = self.motion.track(self.pose, mv["start"], self.map.pos[mv["to"]], stop_at_goal=not through)
            if arrived and not through:
                mv["arrived"] = True
                self._advance_move()
            return v, w
        # stationary: keep centred on the held node
        x, y, _ = self.pose
        cx, cy = self.map.pos[self.node_at]
        if math.hypot(x - cx, y - cy) > 0.15:
            (v, w), _ = self.motion.track(self.pose, (x, y), (cx, cy), stop_at_goal=True, v_cap=0.3)
            return v, w
        return self.motion.stop()

    def _peers_body(self) -> list[tuple[float, float]]:
        x, y, th = self.pose
        c, s = math.cos(th), math.sin(th)
        out = []
        for e in self.table.known():
            px, py, pth = e.state["pose"]
            cx, cy = px + MIR.wheel_dx * math.cos(pth), py + MIR.wheel_dx * math.sin(pth)
            dx, dy = cx - x, cy - y
            out.append((c * dx + s * dy, -s * dx + c * dy))
        return out

    # ------------------------------------------------------------------------------------------------ publication
    def _publish(self, goal: int | None, dist: dict | None) -> None:
        status = ("dwelling" if self.dwell_until is not None else "moving" if self.move is not None
                  else "waiting" if goal is not None else "idle")
        mv = self.move
        self.net.publish("RobotState", self.now, STATE_TTL, pose=[round(v, 3) for v in self.pose],
                         vel=[round(self.motion.v, 3), round(self.motion.w, 3)], battery=self.battery,
                         task_id=self.goals.task_id(), goal_node=goal, status=status, node=self.node_at,
                         held=sorted(self.lock.held), prio=round(self.prio, 4), boosted=self.dl.boosted(self.now),
                         moving_to=None if mv is None else mv["to"], dwelling=self.dwell_until is not None,
                         stalled=self.dl.stalled(self.now), cstep=self.cstep,
                         carrying=self.goals.carrying(),
                         done=[c["task"] for c in getattr(self.goals, "completed", [])[-20:]])
        path = self.preview_path(self.node_at if mv is None else mv["to"], dist)
        if mv is not None:
            path = [mv["to"]] + path
        self.net.publish("RobotIntent", self.now, STATE_TTL, route_revision=self.tick, path=path,
                         target_node=None if mv is None else mv["to"], priority=round(self.prio, 4),
                         valid_until=round(self.now + STATE_TTL, 3))
        self.publish_count += 1

    @property
    def battery(self) -> float:
        return round(self._battery, 4)

    def _status(self, goal: int | None, dist: dict | None) -> dict:
        g = "-" if goal is None else str(self.map.rc(goal))
        mode = ("ISOLATED" if self.iso.blocks() else "DWELL" if self.dwell_until is not None else "MOVE" if self.move is not None
                else "PUSHED" if self.pushes else "WAIT" if goal is not None else "IDLE")
        boost = " BOOST" if self.dl.boosted(self.now) else ""
        waited = self.now - self.goal_since if goal is not None else 0.0
        hud = f"{mode:6s} {self.goals.label()} -> {g} age={waited:5.1f}s{boost} {self.last_reason[:18]}"
        x, y, _ = self.pose
        start = (x, y)
        pts = [list(start)]
        if self.move is not None:
            pts.append(list(self.map.pos[self.move["to"]]))
            base = self.move["to"]
        else:
            base = self.node_at
        pts += [list(self.map.pos[v]) for v in self.preview_path(base, dist, 6)]
        return {"hud": hud, "viz_path": pts, "viz_held": [list(self.map.pos[v]) for v in sorted(self.lock.held)]}

    def stats(self) -> dict:
        lat = self.lock.decision_latency
        return {"lock": self.lock.stats, "net": self.net.stats(), "shield": dict(self.shield.interventions),
                "deadlock": {"escalations": self.dl.escalations, "detections": self.dl.detections[-50:],
                             "recoveries": self.dl.recoveries[-50:], "arrivals": self.dl.arrivals[-50:],
                             **self.dl.episode_summary()},
                "claim_latency_mean": round(float(np.mean(lat)), 4) if lat else None,
                "claim_latency_max": round(float(np.max(lat)), 4) if lat else None,
                "compute_mean_ms": round(1000 * self.decide_time / max(1, self.tick), 3),
                "compute_max_ms": round(1000 * self.decide_max, 3),
                "compute_p99_ms": round(1000 * float(np.percentile(self.compute_samples, 99)), 3) if self.compute_samples else None,
                "wait_time": round(self.wait_time, 2), "publish_count": self.publish_count,
                "peer_table": {"rejected_stale": self.table.rejected_stale, "rejected_old": self.table.rejected_old,
                               "outages": self.table.outage_summary()},
                "goals": self.goals.stats(), "stopwait_timeouts": self.sw.timeouts,
                "blockages_detected": self.blockage.detections[-20:],
                "blocked_known": {str(k): v for k, v in self.blocked_ever.items()},
                "advisor": self.advisor.stats() if self.advisor else None,
                "isolation": self.iso.summary(),
                "optb": self.optb.stats() if self.optb else None}
