"""A* route generation over the aisle graph (design doc §10.6).

State = (node, heading) so that in-place turns of the differential drive are costed: a 90-degree turn costs
`turn_cost` metres-equivalent.  Blocked nodes are excluded; optional per-node extra cost models congestion.
"""
from __future__ import annotations

import heapq
import itertools

from fleet.world.mapgrid import WarehouseMap


def _dir(m: WarehouseMap, u: int, v: int) -> int:
    (r1, c1), (r2, c2) = m.rc(u), m.rc(v)
    return {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}[(r2 - r1, c2 - c1)]


def astar(m: WarehouseMap, start: int, goal: int, blocked: set[int] | frozenset[int] = frozenset(),
          node_cost: dict[int, float] | None = None, turn_cost: float = 1.2, start_heading: int | None = None) -> list[int] | None:
    """Return node path [start, ..., goal] or None if unreachable.  `start` itself is never treated as blocked."""
    if goal in blocked and goal != start:
        return None
    if start == goal:
        return [start]
    cell = m.cell
    gr, gc = m.rc(goal)

    def h(n: int) -> float:
        r, c = m.rc(n)
        return (abs(r - gr) + abs(c - gc)) * cell

    tie = itertools.count()
    s0 = (start, -1 if start_heading is None else start_heading)
    g = {s0: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {s0: None}
    pq = [(h(start), next(tie), s0)]
    closed = set()
    while pq:
        _, _, s = heapq.heappop(pq)
        if s in closed:
            continue
        closed.add(s)
        u, hd = s
        if u == goal:
            path = []
            while s is not None:
                path.append(s[0])
                s = parent[s]
            return path[::-1]
        for v in m.adj[u]:
            if v in blocked:
                continue
            d = _dir(m, u, v)
            turn = 0.0 if hd < 0 or hd == d else (2 * turn_cost if (hd - d) % 4 == 2 else turn_cost)
            ns = (v, d)
            ng = g[s] + cell + turn + (node_cost.get(v, 0.0) if node_cost else 0.0)
            if ng < g.get(ns, float("inf")):
                g[ns] = ng
                parent[ns] = s
                heapq.heappush(pq, (ng + h(v), next(tie), ns))
    return None


def path_cost(m: WarehouseMap, path: list[int], turn_cost: float = 1.2) -> float:
    cost, hd = 0.0, -1
    for u, v in zip(path[:-1], path[1:]):
        d = _dir(m, u, v)
        if hd >= 0 and hd != d:
            cost += 2 * turn_cost if (hd - d) % 4 == 2 else turn_cost
        cost += m.cell
        hd = d
    return cost
