"""Decentralized task allocation (design doc §10.4, FR-04, FR-09, PR-05).

Every robot keeps a replicated task table.  Orders reach all robots from the WMS order stream; allocation is a
peer-to-peer auction with consensus by deterministic rule (CBBA-style, bundle size 1):

  * an available robot bids, for its cheapest open task, B_ij = w_d d_ij + w_t t_ij + w_c C_ij + w_b B_i + w_r R_ij
    (eq. 4) and broadcasts TaskBid(task, cost, revision) to all peers (DEALER/ROUTER);
  * everyone keeps the best (cost, name) bid heard per task and revision; a robot whose own bid is still the best
    after `bid_window` declares itself winner with TaskConsensus(task, winner, revision, cost);
  * conflicting consensus messages are resolved by the same rule: the lower (cost, name) keeps the task, the other
    abandons it and bids again.  Winners re-broadcast consensus periodically and advertise task_id in RobotState,
    so lost messages are repaired (eventual consistency).

Release / reassignment (FR-09): when a task becomes infeasible for its robot (pickup/drop unreachable, or route
cost > `replan_cost_factor` x the bid estimate) it is released with revision+1; when a peer holding a task is
declared LOST, every robot independently re-opens that task with revision+1.  Revisions make stale bids and
consensus messages harmless.

Communication isolation (v3, preflight P4 / addendum A): a robot that is itself isolated does not reopen the tasks
of peers it loses meanwhile (they are `suspended`, not biddable); on reconnection they are restored if their winner is
fresh again, otherwise reopened.  The robot that physically carries a load (own payload sensor) is the authority for
that order: it re-asserts ownership on any higher-revision reopen and ignores TaskDone for it; a robot whose payload
sensor reports no load after its pickup releases the task instead of reporting a false delivery.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fleet.world.mapgrid import WarehouseMap

BID_TTL = 2.0
CONSENSUS_REPEAT = 1.0
V_NOMINAL = 0.8


@dataclass
class Task:
    task_id: str
    pickup: int
    drop: int
    release: float
    revision: int = 0
    status: str = "open"                 # open | assigned | suspended | done
    winner: str | None = None
    win_cost: float = float("inf")
    best: tuple[float, str] | None = None   # best (cost, name) heard for the current revision
    t_consensus: float | None = None
    exclude_until: dict = field(default_factory=dict)   # name -> time (robots that released it recently)


def better(a: tuple[float, str], b: tuple[float, str] | None) -> bool:
    return b is None or (round(a[0], 3), a[1]) < (round(b[0], 3), b[1])


class TaskManager:
    def __init__(self, cfg: dict, name: str, m: WarehouseMap, brain):
        self.cfg, self.name, self.map, self.brain = cfg, name, m, brain
        t = cfg["tasks"]
        self.w = {k: float(t[k]) for k in ("w_dist", "w_time", "w_cong", "w_batt", "w_risk")}
        self.window = float(t["bid_window"])
        self.dwell = float(t["dwell"])
        self.replan_factor = float(cfg["planning"]["replan_cost_factor"])
        b = cfg["battery"]
        self.batt_low, self.batt_full = float(b["low"]), float(b["full"])
        self.tasks: dict[str, Task] = {}
        self.mine: Task | None = None
        self.phase: str | None = None         # None | "pickup" | "drop" | "charge"
        self.my_bid: tuple[str, int, float, float] | None = None   # (task_id, revision, cost, t_sent)
        self.est_cost = 0.0
        self.charger: int | None = None
        self.park: int | None = None
        self.parked = False
        self.idle_since: float | None = 0.0
        self.park_delay = 3.0
        self.events: list[dict] = []
        self.last_consensus_tx = -1e9
        self.alloc_latency: list[float] = []
        self.completed: list[dict] = []
        self.released: list[dict] = []
        self.conflicts = 0
        self.reassigned_from_lost: list[dict] = []
        self.charges: list[dict] = []
        self._goal_ver = 0
        self.suspended: list[dict] = []
        self.reasserted: list[dict] = []
        self.pickup_failed: list[dict] = []
        self.ignored_done = 0

    # ------------------------------------------------------------------------------------------- goal interface
    def current(self) -> int | None:
        if self.phase == "pickup":
            return self.mine.pickup
        if self.phase == "drop":
            return self.mine.drop
        if self.phase == "charge":
            return self.charger
        if self.phase == "park":
            return self.park
        return None

    def goal_key(self):
        return (self.mine.task_id if self.mine else None, self.mine.revision if self.mine else None, self.phase, self._goal_ver)

    def label(self) -> str:
        if self.phase == "charge":
            return "charging"
        if self.phase == "park":
            return "parking"
        return f"{self.mine.task_id}:{self.phase}" if self.mine else "idle"

    def task_id(self):
        return self.mine.task_id if self.mine else None

    def carrying(self) -> bool:
        return self.phase == "drop"

    @property
    def finished(self) -> bool:
        return False

    def on_reached(self, now: float) -> float:
        if self.phase == "charge":
            return 1.0
        if self.phase == "park":
            return 0.0
        return self.dwell

    def drain_events(self) -> list[dict]:
        ev, self.events = self.events, []
        return ev

    def on_dwell_done(self, now: float) -> list[dict]:
        if self.phase == "pickup":
            self.events.append({"task_id": self.mine.task_id, "phase": "picked"})
            self.phase = "drop"
        elif self.phase == "drop":
            t = self.mine
            self.events.append({"task_id": t.task_id, "phase": "delivered"})
            t.status = "done"
            self.completed.append({"task": t.task_id, "t": round(now, 3)})
            self.brain._broadcast(self.brain.live_peers(), "TaskDone", 0.0, task_id=t.task_id, revision=t.revision)
            self.mine, self.phase = None, None
        elif self.phase == "park":
            self.phase, self.park = None, None        # parked: idle, still bidding, pushable
            self.parked = True
        elif self.phase == "charge":
            if self.brain.battery >= self.batt_full:
                self.charges[-1].update({"end": round(now, 3), "battery_end": round(self.brain.battery, 4)})
                self.phase, self.charger = None, None
            else:
                self._goal_ver += 1          # stay docked: re-dwell on the charger
        return self.drain_events()

    # ------------------------------------------------------------------------------------------- inputs
    def on_orders(self, orders: list[dict], now: float) -> None:
        for o in orders:
            if o["task_id"] not in self.tasks:
                self.tasks[o["task_id"]] = Task(o["task_id"], int(o["pickup"]), int(o["dropoff"]), float(o["release"]))

    def handle(self, m: dict, now: float) -> None:
        t = self.tasks.get(m.get("task_id"))
        if t is None:
            return
        typ = m["type"]
        if typ == "TaskBid":
            if m["revision"] > t.revision and self.mine is t and self._holding_load():
                self._reopen(t, m["revision"], now)          # peers re-auction an order we carry: re-assert
            elif m["revision"] == t.revision and t.status == "open":
                bid = (float(m["estimated_cost"]), m["src"])
                if better(bid, t.best):
                    t.best = bid
        elif typ == "TaskConsensus":
            self._on_consensus(t, m, now)
        elif typ == "TaskDone":
            if self.mine is t and self._holding_load():
                self.ignored_done += 1          # we physically carry this order: it cannot have been delivered
                return
            if m["revision"] >= t.revision:
                t.status = "done"
                if self.mine is t:          # someone else delivered it: drop our copy
                    self.mine, self.phase = None, None
        elif typ == "TaskAnnouncement":
            if m["revision"] > t.revision and t.status != "done":
                self._reopen(t, m["revision"], now)
                if m.get("status") == "released":
                    t.exclude_until[m["src"]] = now + 20.0

    def _on_consensus(self, t: Task, m: dict, now: float) -> None:
        if m["revision"] < t.revision or t.status == "done":
            return
        if m["revision"] > t.revision:
            self._reopen(t, m["revision"], now)
        claim = (float(m["cost"]), m["winner"])
        if t.status == "assigned" and t.winner == self.name and m["winner"] != self.name:
            # two robots both believe they won: lower (cost, name) keeps it
            if better(claim, (t.win_cost, self.name)):
                self.conflicts += 1
                self._abandon(t, now)
                t.status, t.winner, t.win_cost = "assigned", m["winner"], claim[0]
            return
        if t.status in ("open", "suspended") or better(claim, (t.win_cost, t.winner or "~")):
            if t.status == "open" and t.t_consensus is None:
                t.t_consensus = now
            t.status, t.winner, t.win_cost = "assigned", m["winner"], claim[0]
            if self.my_bid and self.my_bid[0] == t.task_id:
                self.my_bid = None

    def on_peer_done(self, done: list[str], now: float) -> None:
        """Completed task ids advertised in peers' RobotState (repairs lost TaskDone messages)."""
        for tid in done:
            t = self.tasks.get(tid)
            if t is not None and t.status != "done":
                t.status = "done"
                if self.mine is t:
                    self.mine, self.phase = None, None

    def on_peer_lost(self, name: str, now: float) -> None:
        iso = getattr(self.brain, "iso", None)
        for t in self.tasks.values():
            if t.status == "assigned" and t.winner == name:
                if iso is not None and iso.blocks():
                    # we are isolated ourselves: the peer is probably fine, do not take its task over
                    t.status = "suspended"
                    self.suspended.append({"task": t.task_id, "from": name, "t": round(now, 3)})
                    continue
                self.reassigned_from_lost.append({"task": t.task_id, "from": name, "t": round(now, 3)})
                self._reopen(t, t.revision + 1, now)

    def on_reconnect(self, now: float) -> tuple[list[str], list[str]]:
        """After isolation: restore suspended tasks whose winner is fresh again, reopen the others."""
        from fleet.comms.peer_table import FRESH
        restored, reopened = [], []
        for t in self.tasks.values():
            if t.status != "suspended":
                continue
            e = self.brain.table.entries.get(t.winner)
            if e is not None and e.status == FRESH:
                t.status = "assigned"
                restored.append(t.task_id)
            else:
                self.reassigned_from_lost.append({"task": t.task_id, "from": t.winner, "t": round(now, 3)})
                self._reopen(t, t.revision + 1, now)
                reopened.append(t.task_id)
        return restored, reopened

    def _holding_load(self) -> bool:
        """Phase 'drop' and the payload sensor does not contradict it (None = sensor not provided)."""
        return self.phase == "drop" and getattr(self.brain, "load", None) is not False

    # ------------------------------------------------------------------------------------------- internals
    def _reopen(self, t: Task, revision: int, now: float) -> None:
        if self.mine is t and revision > t.revision and self._holding_load():
            # carrier authority: only the robot holding the load can deliver it
            t.revision, t.status, t.winner, t.win_cost, t.best = revision + 1, "assigned", self.name, -1.0, None
            self.reasserted.append({"task": t.task_id, "revision": t.revision, "t": round(now, 3)})
            self._tx_consensus(t, now)
            return
        if self.mine is t and revision > t.revision:
            self.mine, self.phase = None, None
        t.revision, t.status, t.winner, t.win_cost, t.best = revision, "open", None, float("inf"), None
        if self.my_bid and self.my_bid[0] == t.task_id:
            self.my_bid = None

    def _abandon(self, t: Task, now: float) -> None:
        if self.mine is t:
            self.mine, self.phase = None, None

    def release_current(self, now: float, reason: str, force: bool = False) -> None:
        t = self.mine
        if t is None or self.phase == "charge":
            return
        if self.phase == "drop" and not force:
            return          # holding the load: keep trying (load cannot be handed over in this model)
        self.released.append({"task": t.task_id, "t": round(now, 3), "reason": reason})
        self.mine, self.phase = None, None
        new_rev = t.revision + 1
        self._reopen(t, new_rev, now)
        t.exclude_until[self.name] = now + 20.0
        self.brain._broadcast(self.brain.live_peers(), "TaskAnnouncement", 0.0, task_id=t.task_id, pickup=t.pickup,
                              dropoff=t.drop, deadline=None, priority=0, status="released", revision=new_rev)

    def bid_cost(self, t: Task, now: float) -> float | None:
        b = self.brain
        here = b.node_at if b.move is None else b.move["to"]
        if t.pickup in b.blocked_set() or t.drop in b.blocked_set():
            return None                     # infeasible while blocked: leave it open
        dp = b.dist_field(t.pickup)
        dd = b.dist_field(t.drop)
        if dp is None or dd is None or here not in dp or t.drop not in dp:
            return None
        hops = dp[here] + dp[t.drop]
        d = hops * self.map.cell
        t_hat = d / V_NOMINAL
        occ = [st.get("node") for _, (_, st) in b.peer_nodes().items()]
        cong = sum(1 for n in occ if n is not None and n in dp and dp[n] <= 2)
        batt = max(0.0, 1.0 - b.battery)
        risk = sum(1 for v in b.blocked_set() if v in dp and dp[v] <= dp.get(here, 0))
        w = self.w
        return w["w_dist"] * d + w["w_time"] * t_hat + w["w_cong"] * cong + w["w_batt"] * 10 * batt + w["w_risk"] * risk

    def tick(self, now: float, brain) -> None:
        iso = getattr(brain, "iso", None)
        if iso is not None and iso.blocks():
            return                  # isolated: no bids, releases or re-planning decisions until reconnected
        if self.mine is not None and self.phase == "drop" and getattr(brain, "load", None) is False:
            # payload sensor: the pickup did not happen (e.g. the load is still on a robot we had lost contact with)
            self.pickup_failed.append({"task": self.mine.task_id, "t": round(now, 3)})
            self.release_current(now, "load absent after pickup", force=True)
            return
        # periodic consensus re-broadcast for the task we hold (repairs lost messages)
        if self.mine is not None and now - self.last_consensus_tx >= CONSENSUS_REPEAT:
            self._tx_consensus(self.mine, now)
        if self.mine is not None and self.phase in ("pickup", "drop"):
            self._check_feasible(now)
        if self.mine is not None or self.phase == "charge":
            self.idle_since = None
            return
        if brain.battery < self.batt_low:
            self._go_charge(now)
            return
        if self.idle_since is None:
            self.idle_since = now
        if (self.phase is None and not self.parked and self.my_bid is None and now - self.idle_since >= self.park_delay
                and brain.node_at not in self._rest_spots()):
            self._go_park(now)
        # auction
        if self.my_bid is not None:
            tid, rev, cost, t_sent = self.my_bid
            t = self.tasks[tid]
            if t.status != "open" or t.revision != rev:
                self.my_bid = None
            elif t.best is not None and better(t.best, (cost, self.name)) and t.best[1] != self.name:
                self.my_bid = None                 # outbid: choose again next tick
            elif now - t_sent >= self.window:
                self._win(t, cost, now)
                return
            else:
                return
        best = None
        for t in self.tasks.values():
            if t.status != "open" or t.exclude_until.get(self.name, -1) > now:
                continue
            c = self.bid_cost(t, now)
            if c is None:
                continue
            if t.best is not None and t.best[1] != self.name and better(t.best, (c, self.name)):
                continue
            if best is None or (c, t.release, t.task_id) < (best[0], best[1].release, best[1].task_id):
                best = (c, t)
        if best is not None:
            c, t = best
            if better((c, self.name), t.best):
                t.best = (c, self.name)
            self.my_bid = (t.task_id, t.revision, c, now)
            brain._broadcast(brain.live_peers(), "TaskBid", BID_TTL, task_id=t.task_id, estimated_cost=round(c, 4),
                             route_cost=round(c, 4), revision=t.revision)

    def _rest_spots(self) -> list[int]:
        return self.map.stations("park") + self.map.stations("charger")

    def _go_park(self, now: float) -> None:
        """Idle robots clear the aisles (real fleets park idle AMRs); any free rest/charger alcove will do."""
        b = self.brain
        taken = {st.get("goal_node") for _, (_, st) in b.peer_nodes().items()} | b.view.occupied()
        spots = [c for c in self._rest_spots() if c not in taken]
        if not spots:
            return
        here = b.node_at
        self.park = min(spots, key=lambda c: (b.map.bfs_dist(c).get(here, 1e9), c))
        self.phase = "park"

    def _win(self, t: Task, cost: float, now: float) -> None:
        self.parked = False
        if self.phase == "park":
            self.phase, self.park = None, None
        t.status, t.winner, t.win_cost = "assigned", self.name, cost
        t.t_consensus = now
        self.alloc_latency.append(round(now - t.release, 3))
        self.mine, self.phase, self.my_bid = t, "pickup", None
        self.idle_since = None
        self.est_cost = cost
        self._tx_consensus(t, now)

    def _tx_consensus(self, t: Task, now: float) -> None:
        self.last_consensus_tx = now
        self.brain._broadcast(self.brain.live_peers(), "TaskConsensus", 0.0, task_id=t.task_id, winner=self.name,
                              revision=t.revision, confidence=1.0, cost=round(t.win_cost, 4))

    def _check_feasible(self, now: float) -> None:
        b = self.brain
        goal = self.current()
        here = b.node_at
        blocked = b.blocked_set()
        if goal in blocked:
            self.release_current(now, "goal blocked")
            return
        d = b.map.bfs_dist(goal, frozenset(blocked))
        if here not in d:
            self.release_current(now, "unreachable")
            return
        if self.phase == "pickup":
            cost_now = (d[here] + (b.dist_field(self.mine.drop) or {}).get(self.mine.pickup, 0)) * self.map.cell
            if cost_now > self.replan_factor * max(self.est_cost, self.map.cell * 2):
                self.release_current(now, "route cost exceeded")

    def _go_charge(self, now: float) -> None:
        b = self.brain
        taken = {st.get("goal_node") for _, (_, st) in b.peer_nodes().items()} | b.view.occupied()
        chargers = [c for c in self.map.stations("charger") if c not in taken]
        if not chargers:
            return
        here = b.node_at
        dist = {c: b.map.bfs_dist(c).get(here, 1e9) for c in chargers}
        self.charger = min(chargers, key=lambda c: (dist[c], c))
        self.phase = "charge"
        self.charges.append({"start": round(now, 3), "charger": self.charger, "battery_start": round(b.battery, 4)})

    def stats(self) -> dict:
        return {"completed": self.completed, "alloc_latency": self.alloc_latency, "released": self.released,
                "conflicts": self.conflicts, "reassigned_from_lost": self.reassigned_from_lost, "charges": self.charges,
                "known_tasks": len(self.tasks), "open": sum(t.status == "open" for t in self.tasks.values()),
                "suspended": self.suspended, "reasserted": self.reasserted, "pickup_failed": self.pickup_failed,
                "ignored_done": self.ignored_done}
