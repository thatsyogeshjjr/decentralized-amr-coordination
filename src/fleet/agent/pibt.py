"""Reference (synchronous) PIBT - Priority Inheritance with Backtracking, Okumura et al., IJCAI 2019.

Used as (a) the specification the decentralized protocol in `coord.py` implements asynchronously, (b) the
planner of the benchmark-only centralized reference, and (c) the step function of the graph-level training
simulator.  One call computes the next node of every agent for one synchronous timestep.
"""
from __future__ import annotations

from typing import Callable

from fleet.world.mapgrid import WarehouseMap


def pibt_step(m: WarehouseMap, pos: list[int], prio: list[float], dist: list[Callable[[int], float]],
              blocked: set[int] | frozenset[int] = frozenset(), tiebreak: Callable[[int, int], float] | None = None,
              paths: list[list[int]] | None = None) -> list[int]:
    """pos[i]: current node; prio[i]: higher moves first; dist[i](v): distance-to-goal heuristic of agent i.
    paths[i] (optional): agent i's intended next nodes.  A *pushed* agent first ranks candidates that are off its
    pusher's intended path (evasion: step aside instead of being herded ahead of the pusher into a dead end),
    then by its own distance-to-goal as in standard PIBT.  Returns next nodes with no vertex/swap conflicts."""
    n = len(pos)
    occ_now = {p: i for i, p in enumerate(pos)}
    nxt: list[int | None] = [None] * n
    occ_next: dict[int, int] = {}

    def cands(i: int, pusher: int | None) -> list[int]:
        c = [v for v in m.adj[pos[i]] if v not in blocked] + [pos[i]]
        tb = (lambda v: tiebreak(i, v)) if tiebreak else (lambda v: 0.0)
        avoid = set(paths[pusher]) if (paths is not None and pusher is not None) else set()
        return sorted(c, key=lambda v: (v in avoid, dist[i](v), tb(v), v))

    def solve(i: int, pusher: int | None) -> bool:
        for v in cands(i, pusher):
            if v in occ_next:
                continue
            if pusher is not None and v == pos[pusher]:
                continue
            nxt[i] = v
            occ_next[v] = i
            k = occ_now.get(v)
            if k is not None and k != i and nxt[k] is None:
                if not solve(k, i):
                    continue
            return True
        nxt[i] = pos[i]
        occ_next[pos[i]] = i
        return False

    for i in sorted(range(n), key=lambda a: -prio[a]):
        if nxt[i] is None:
            solve(i, None)
    return [int(v) for v in nxt]  # type: ignore[arg-type]


def check_step(pos: list[int], nxt: list[int]) -> None:
    """Raise if a joint move has vertex or swap conflicts (used by tests and the training simulator)."""
    if len(set(nxt)) != len(nxt):
        raise AssertionError(f"vertex conflict {nxt}")
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if nxt[i] == pos[j] and nxt[j] == pos[i] and pos[i] != pos[j]:
                raise AssertionError(f"swap conflict {i}<->{j}")
