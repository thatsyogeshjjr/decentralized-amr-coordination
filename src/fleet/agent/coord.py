"""Decentralized traffic coordination (design doc §10.6, FR-06).

Layer 1 - node-claim mutual exclusion (safety-relevant, used by every policy).
  A robot always holds the node it stands on, and additionally the node it is moving into.  To move into v it
  sends Claim(v, priority) to every live peer over DEALER/ROUTER and may only start moving once *all* of them
  granted.  A peer denies when it holds v, when it has an outstanding claim on v of higher (priority, name), or
  when it already granted v to someone else.  A lower-priority pending claimant that grants aborts its own claim.
  => at most one robot can acquire a node; message loss/delay only causes retries (claims time out, never
  succeed by default).  Peers declared LOST need not answer, but their last known nodes stay occupied.

Layer 2 - policy (which node to claim):
  * PIBTPolicy: asynchronous PIBT.  Candidates are neighbours that reduce distance-to-goal; a candidate held by a
    stationary lower-priority robot triggers Push(priority inheritance); the pushed robot must vacate (not into
    the pusher's node, preferring nodes off the pusher's path) or reply failure, upon which the pusher backtracks
    to its next candidate.
  * StopWaitPolicy: the stop-and-wait baseline (App. B): follow the A* route; halt whenever the next nodes overlap
    a peer's held/intended nodes and that peer has precedence; resume when the region is clear.  A timeout
    re-plan around the conflicting robot prevents permanent mutual waiting.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from fleet.agent.astar import astar
from fleet.comms.peer_table import LOST, PeerTable
from fleet.world.mapgrid import WarehouseMap

CLAIM_TTL = 1.0
CLAIM_TIMEOUT = 1.2
CLAIM_RESEND = 0.2
GRANT_TIMEOUT = 3.0
PUSH_TIMEOUT = 4.0          # no acknowledgement within this time -> backtrack
PUSH_ACKED_TIMEOUT = 15.0  # acknowledged (possibly chained) evasion


def rank(prio: float, name: str) -> tuple[float, str]:
    """Total order used by every tie: higher priority wins, then lexicographically *smaller* name."""
    return (prio, "".join(chr(255 - ord(c)) for c in name))


# ------------------------------------------------------------------------------------------- peer view helpers
class PeerView:
    def __init__(self, table: PeerTable):
        self.t = table

    def holder(self, v: int) -> tuple[str, dict] | None:
        for e in self.t.known():
            if v in e.state.get("held", []):
                return e.name, e.state
        return None

    def occupied(self) -> set[int]:
        out: set[int] = set()
        for e in self.t.known():
            out.update(e.state.get("held", []))
        return out

    def lost_nodes(self) -> set[int]:
        out: set[int] = set()
        for e in self.t.entries.values():
            if e.status == LOST and e.state is not None:
                out.update(e.state.get("held", []))
        return out

    def intent_path(self, name: str) -> list[int]:
        e = self.t.entries.get(name)
        return list(e.intent.get("path", [])) if e and e.intent else []


# ------------------------------------------------------------------------------------------- lock manager
@dataclass
class PendingClaim:
    node: int
    claim_id: str
    prio: float
    need: set
    granted: set = field(default_factory=set)
    denied: bool = False
    t0: float = 0.0
    last_send: float = 0.0


class LockManager:
    def __init__(self, name: str, send, broadcast, bypass: bool = False):
        self.name = name
        self.bypass = bypass         # FAULT INJECTION ONLY: acquire without asking peers (tests the safety shield)
        self._send = send            # send(peer, type, ttl, **body)
        self._broadcast = broadcast  # broadcast(peers, type, ttl, **body)
        self.held: set[int] = set()
        self.pending: PendingClaim | None = None
        self.grants: dict[int, tuple[str, str, float]] = {}   # node -> (holder, claim_id, t)
        self._ids = itertools.count(1)
        self.stats = {"claims": 0, "acquired": 0, "denied": 0, "timeouts": 0, "granted_to_peers": 0, "denied_peers": 0,
                      "cancelled": 0}
        self.decision_latency: list[float] = []

    # -- own claims ------------------------------------------------------------------------------------------
    def claim(self, v: int, prio: float, live_peers: list[str], now: float) -> None:
        cid = f"{self.name}:{next(self._ids)}"
        self.pending = PendingClaim(v, cid, prio, set(live_peers), t0=now, last_send=now)
        self.stats["claims"] += 1
        if self.bypass:
            live_peers = []
            self.pending.need = set()
        if live_peers:
            self._broadcast(live_peers, "Claim", CLAIM_TTL, node=v, priority=prio, claim_id=cid)

    def poll(self, now: float, live_peers: list[str]) -> str | None:
        """Advance the pending claim. Returns 'acquired' | 'failed' | None (still pending / nothing pending)."""
        p = self.pending
        if p is None:
            return None
        p.need &= set(live_peers)               # peers that died meanwhile need not answer
        if p.denied:
            return self._fail("denied")
        if p.need <= p.granted:
            self.held.add(p.node)
            self.pending = None
            self.stats["acquired"] += 1
            self.decision_latency.append(now - p.t0)
            return "acquired"
        if now - p.t0 > CLAIM_TIMEOUT:
            return self._fail("timeouts")
        if now - p.last_send >= CLAIM_RESEND:
            missing = sorted(p.need - p.granted)
            self._broadcast(missing, "Claim", CLAIM_TTL, node=p.node, priority=p.prio, claim_id=p.claim_id)
            p.last_send = now
        return None

    def _fail(self, why: str) -> str:
        p = self.pending
        self.pending = None
        self.stats["denied" if why == "denied" else "timeouts"] += 1
        if p.granted:
            self._broadcast(sorted(p.granted), "Release", CLAIM_TTL, nodes=[p.node])
        return "failed"

    def cancel(self) -> None:
        """Abandon the pending claim (communication isolation); peers that granted it get a Release."""
        p = self.pending
        if p is None:
            return
        self.pending = None
        self.stats["cancelled"] += 1
        if p.granted:
            self._broadcast(sorted(p.granted), "Release", CLAIM_TTL, nodes=[p.node])

    def release(self, nodes: list[int]) -> None:
        for v in nodes:
            self.held.discard(v)

    # -- peer claims ---------------------------------------------------------------------------------------------
    def on_claim(self, msg: dict, my_prio: float, now: float, extra_busy: set[int] = frozenset()) -> None:
        v, src, cid = msg["node"], msg["src"], msg["claim_id"]
        g = self.grants.get(v)
        if g is not None and now - g[2] > GRANT_TIMEOUT:
            del self.grants[v]
            g = None
        ok = True
        if v in self.held or v in extra_busy:
            ok = False
        elif g is not None and g[0] != src:
            ok = False
        elif self.pending is not None and self.pending.node == v:
            if rank(self.pending.prio, self.name) > rank(msg["priority"], src):
                ok = False
            else:
                self.pending.denied = True     # yield: my own claim on v can no longer succeed
        if ok:
            self.grants[v] = (src, cid, now)
            self.stats["granted_to_peers"] += 1
        else:
            self.stats["denied_peers"] += 1
        self._send(src, "ClaimReply", CLAIM_TTL, node=v, claim_id=cid, granted=ok)

    def on_reply(self, msg: dict) -> None:
        p = self.pending
        if p is None or msg["claim_id"] != p.claim_id:
            return
        if msg["granted"]:
            p.granted.add(msg["src"])
        else:
            p.denied = True

    def on_release(self, msg: dict) -> None:
        for v in msg["nodes"]:
            g = self.grants.get(v)
            if g is not None and g[0] == msg["src"]:
                del self.grants[v]

    def on_peer_state(self, src: str, held: list[int]) -> None:
        """Once a holder advertises the node itself, it protects it; my grant record is no longer needed."""
        for v in held:
            g = self.grants.get(v)
            if g is not None and g[0] == src:
                del self.grants[v]


# ------------------------------------------------------------------------------------------- policies
@dataclass
class Decision:
    kind: str                 # "claim" | "push" | "wait" | "fail_push"
    node: int | None = None
    peer: str | None = None
    reason: str = ""


def straight_bonus(m: WarehouseMap, prev: int | None, u: int, v: int) -> int:
    if prev is None:
        return 0
    (r0, c0), (r1, c1), (r2, c2) = m.rc(prev), m.rc(u), m.rc(v)
    return 0 if (r1 - r0, c1 - c0) == (r2 - r1, c2 - c1) else 1


class PIBTPolicy:
    def __init__(self, m: WarehouseMap, name: str, tiebreak):
        self.m, self.name, self.tb = m, name, tiebreak
        self.failed_until: dict[int, float] = {}

    def mark_failed(self, v: int, now: float, hold: float = 2.0) -> None:
        self.failed_until[v] = now + hold

    def decide(self, now: float, u: int, prev: int | None, dist: dict[int, int] | None, eff_prio: float,
               view: PeerView, blocked: set[int], pushes: list[dict], dwelling: bool) -> Decision:
        if dwelling:
            return Decision("fail_push" if pushes else "wait", reason="dwelling")
        pushed = bool(pushes)
        avoid_hard = {p["from_node"] for p in pushes}
        avoid_soft = set().union(*[set(p["path"]) for p in pushes]) if pushes else set()
        chain = set().union(*[set(p["chain"]) for p in pushes]) if pushes else set()
        D = (lambda v: dist.get(v, 10 ** 6)) if dist is not None else (lambda v: 0)
        nbrs = [v for v in self.m.adj[u] if v not in blocked and v not in avoid_hard]
        if pushed:
            cands = sorted(nbrs, key=lambda v: (v in avoid_soft, D(v), straight_bonus(self.m, prev, u, v), self.tb(v), v))
        else:
            if dist is None:
                return Decision("wait", reason="no goal")
            cands = sorted([v for v in nbrs if D(v) < D(u)],
                           key=lambda v: (D(v), straight_bonus(self.m, prev, u, v), self.tb(v), v))
        for v in cands:
            if self.failed_until.get(v, -1.0) > now:
                continue
            h = view.holder(v)
            if h is None:
                return Decision("claim", v)
            peer, st = h
            if st.get("moving_to") is not None or st.get("dwelling") or peer in chain:
                continue
            if view.t.entries[peer].status == LOST:
                continue
            if rank(eff_prio, self.name) > rank(float(st.get("prio", -1e9)), peer):
                return Decision("push", v, peer)
        if pushed:
            return Decision("fail_push", reason="no escape")
        return Decision("wait", reason="blocked")


class StopWaitPolicy:
    """Stop-and-wait baseline.  Uses the same claim layer for safety; never pushes."""

    def __init__(self, m: WarehouseMap, name: str, timeout: float, horizon: int = 3):
        self.m, self.name, self.timeout, self.K = m, name, timeout, horizon
        self.route: list[int] = []
        self.route_goal: int | None = None
        self.halt_since: float | None = None
        self.detour_block: set[int] = set()
        self.timeouts = 0

    def plan(self, u: int, goal: int, blocked: set[int]) -> None:
        self.route = astar(self.m, u, goal, blocked | self.detour_block) or astar(self.m, u, goal, blocked) or []
        self.route_goal = goal

    def next_nodes(self, u: int) -> list[int]:
        if u in self.route:
            i = self.route.index(u)
            return self.route[i + 1: i + 1 + self.K]
        return []

    def decide(self, now: float, u: int, goal: int | None, view: PeerView, blocked: set[int]) -> Decision:
        if goal is None or u == goal:
            self.halt_since = None
            return Decision("wait", reason="no goal")
        if self.route_goal != goal or u not in self.route or any(v in blocked for v in self.next_nodes(u)):
            self.detour_block = set()
            self.plan(u, goal, blocked)
        mine = self.next_nodes(u)
        if not mine:
            return Decision("wait", reason="no route")
        conflict_peers = []
        for e in view.t.known():
            held = list(e.state.get("held", []))
            ipath = view.intent_path(e.name)[: self.K]
            for k, v in enumerate(mine):
                if v in held or v in ipath:
                    their_hops = 0 if v in held else ipath.index(v) + 1
                    my_hops = k + 1
                    if their_hops < my_hops or (their_hops == my_hops and e.name < self.name) or e.status == LOST:
                        conflict_peers.append(e.name)
                    break
        if conflict_peers:
            if self.halt_since is None:
                self.halt_since = now
            if now - self.halt_since > self.timeout:
                # re-plan around the robots we are waiting for
                occ = set()
                for p in conflict_peers:
                    occ.update(view.t.entries[p].state.get("held", []))
                self.detour_block = occ
                self.plan(u, goal, blocked)
                self.halt_since = now
                self.timeouts += 1
                if self.route and len(self.route) > 1 and self.route[1] not in view.occupied():
                    return Decision("claim", self.route[1], reason="detour")
                # no detour: yield by backing off to a free neighbour outside the conflicting robots' regions
                region = set()
                for p in conflict_peers:
                    region.update(view.t.entries[p].state.get("held", []))
                    region.update(view.intent_path(p)[: self.K])
                occ = view.occupied()
                back = [v for v in self.m.adj[u] if v not in occ and v not in blocked and v not in region]
                if back:
                    return Decision("claim", min(back), reason="backoff")
            return Decision("wait", reason="yield:" + ",".join(conflict_peers))
        self.halt_since = None
        v = mine[0]
        if view.holder(v) is not None:
            return Decision("wait", reason="occupied")
        return Decision("claim", v)
