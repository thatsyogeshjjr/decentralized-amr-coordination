"""Compact numerical state for the learned priority advisor (design doc §10.9).

The same 8 features are computed by the graph-level trainer and by the robot agents at runtime:
  0 normalized hop distance to current goal          4 robots within 2 hops (local density)
  1 normalized time since task start (waiting)       5 robots on own next 3 route nodes
  2 phase: 1 = carrying load (to drop), 0 = to pickup 6 own node degree (1 = dead end .. 4 = junction) / 4
  3 robots within 1 hop                               7 free neighbour fraction
"""
from __future__ import annotations

from typing import Callable

import numpy as np

N_FEATURES = 8
DIST_NORM = 20.0
TIME_NORM = 60.0


def feature_vector(dist_to_goal: float, waited: float, carrying: bool, n1: int, n2: int, on_route: int,
                   degree: int, free_frac: float) -> np.ndarray:
    return np.array([min(dist_to_goal, 60) / DIST_NORM, min(waited, 300) / TIME_NORM, float(carrying),
                     n1, n2, on_route, degree / 4.0, free_frac], dtype=np.float32)


def local_counts(adj: dict[int, list[int]], node: int, others: set[int], route: list[int]) -> tuple[int, int, int, int, float]:
    one = set(adj[node])
    two = {w for u in one for w in adj[u]} | one
    two.discard(node)
    n1 = len(one & others)
    n2 = len(two & others)
    on_route = len(set(route[:3]) & others)
    free = sum(1 for v in adj[node] if v not in others)
    return n1, n2, on_route, len(adj[node]), free / max(1, len(adj[node]))


def robot_features(m, pos: list[int], goal: list, phase: list, elapsed: list[float], i: int,
                   D: Callable[[int], dict[int, int]]) -> np.ndarray:
    """Graph-simulator variant.  `elapsed` is measured in steps (~2 s each in the physical world)."""
    others = {p for j, p in enumerate(pos) if j != i}
    g = goal[i]
    dist = D(g).get(pos[i], 60) if g is not None else 0
    # next 3 nodes along a greedy descent of the distance field
    route, u = [], pos[i]
    if g is not None:
        dd = D(g)
        for _ in range(3):
            nb = [v for v in m.adj[u] if dd.get(v, 1e9) < dd.get(u, 1e9)]
            if not nb:
                break
            u = min(nb)
            route.append(u)
    n1, n2, onr, deg, free = local_counts(m.adj, pos[i], others, route)
    return feature_vector(dist, elapsed[i] * 2.0, phase[i] == "drop", n1, n2, onr, deg, free)
