"""Graph-level (discrete, no physics) fleet simulator sharing the planner code.

Used for (a) algorithm unit tests and (b) optimizing the learned priority model over thousands of episodes
(user-approved; all evaluation is done in the visual MuJoCo world).  One step = every robot moves at most one
edge (synchronous PIBT).  Deadlock handling uses the agents' `DeadlockMonitor` with time measured in steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from fleet.agent.deadlock import BOOST, DeadlockMonitor
from fleet.agent.pibt import check_step, pibt_step
from fleet.world.mapgrid import WarehouseMap


@dataclass
class GridResult:
    steps: int | None
    reached_at: list
    deadlock_escalations: int
    completed: int = 0


def within2(m: WarehouseMap, a: int, b: int) -> bool:
    return a == b or b in m.adj[a] or any(b in m.adj[u] for u in m.adj[a])


def intended_path(m: WarehouseMap, start: int, dfield: dict[int, int], k: int = 4) -> list[int]:
    """Next k nodes of a greedy descent of the distance field (what an agent advertises as its intent)."""
    out, u = [], start
    for _ in range(k):
        nb = [v for v in m.adj[u] if dfield.get(v, 1e9) < dfield.get(u, 1e9)]
        if not nb:
            break
        u = min(nb, key=lambda v: (dfield[v], v))
        out.append(u)
    return out


def _deadlock_inputs(m, pos, prio, boosted, i, stalled):
    near = [j for j in range(len(pos)) if j != i and within2(m, pos[i], pos[j])]
    return any(prio[j] > prio[i] and stalled[j] for j in near), any(boosted[j] for j in near)


def run_goals(m: WarehouseMap, starts: list[int], goals: list[int], max_steps: int = 300,
              deadlock_window: int | None = 6, boost_steps: int = 8, seed: int = 0) -> GridResult:
    """Each robot travels to one goal and stays there (one-shot MAPF instance)."""
    n = len(starts)
    pos = list(starts)
    dists = [m.bfs_dist(g) for g in goals]
    dfun = [lambda v, d=d: d.get(v, 1e9) for d in dists]
    tb = np.random.default_rng(seed).random((n, max(m.pos) + 1))
    elapsed = [0.0] * n
    mon = [DeadlockMonitor(window=deadlock_window or 1e9, eps=0.5, boost_time=boost_steps) for _ in range(n)]
    reached = [None] * n
    for t in range(max_steps):
        boosted = [mon[i].boosted(t) for i in range(n)]
        prio = [elapsed[i] + i * 1e-3 + (BOOST if boosted[i] else 0.0) for i in range(n)]
        paths = [intended_path(m, pos[i], dists[i]) for i in range(n)]
        nxt = pibt_step(m, pos, prio, dfun, tiebreak=lambda i, v: tb[i, v], paths=paths)
        check_step(pos, nxt)
        pos = nxt
        stalled = [mon[j].stalled(t + 1) for j in range(n)]
        for i in range(n):
            at_goal = pos[i] == goals[i]
            if at_goal:
                elapsed[i] = 0.0
                if reached[i] is None:
                    reached[i] = t + 1
            else:
                elapsed[i] += 1.0
            if deadlock_window is not None:
                hn, bn = _deadlock_inputs(m, pos, prio, boosted, i, stalled)
                mon[i].update(t + 1, None if at_goal else dists[i].get(pos[i], 1e9), hn, bn)
        if all(p == g for p, g in zip(pos, goals)):
            return GridResult(t + 1, reached, sum(x.escalations for x in mon))
    return GridResult(None, reached, sum(x.escalations for x in mon))


def run_tasks(m: WarehouseMap, starts: list[int], tasks: list[tuple[int, int, int]], priority_fn: Callable | None = None,
              max_steps: int = 2000, deadlock_window: int = 6, boost_steps: int = 8, dwell: int = 1, seed: int = 0) -> GridResult:
    """Lifelong pickup->drop episode.  tasks: (release_step, pickup, drop).  Nearest-idle assignment (graph level
    only).  priority_fn(features) -> score is added to the PIBT priority (learned advisor hook)."""
    from fleet.learning.features import robot_features
    n = len(starts)
    pos = list(starts)
    tb = np.random.default_rng(seed).random((n, max(m.pos) + 1))
    pending = sorted(tasks)
    queue: list[tuple[int, int, int]] = []
    goal: list[int | None] = [None] * n
    phase: list[str | None] = [None] * n
    task_of: list = [None] * n
    wait = [0] * n
    elapsed = [0.0] * n
    mon = [DeadlockMonitor(window=deadlock_window, eps=0.5, boost_time=boost_steps) for _ in range(n)]
    dist_cache: dict[int, dict[int, int]] = {}
    done = 0

    def D(g: int) -> dict[int, int]:
        if g not in dist_cache:
            dist_cache[g] = m.bfs_dist(g)
        return dist_cache[g]

    for t in range(max_steps):
        while pending and pending[0][0] <= t:
            queue.append(pending.pop(0))
        for i in range(n):
            if phase[i] is None and queue:
                k = min(range(len(queue)), key=lambda q: (D(queue[q][1]).get(pos[i], 1e9), q))
                _, p, d = queue.pop(k)
                task_of[i], goal[i], phase[i] = (p, d), p, "pick"
                elapsed[i] = 0.0
                mon[i].reset_goal(t, p)
        boosted = [mon[i].boosted(t) for i in range(n)]
        prio = []
        for i in range(n):
            # priority in seconds-equivalent (one graph step ~ 2 s of MiR motion), as in the deployed agents
            base = 2.0 * elapsed[i] if phase[i] else -1.0
            if priority_fn is not None and phase[i]:
                base += float(priority_fn(robot_features(m, pos, goal, phase, elapsed, i, D)))
            prio.append(base + i * 1e-3 + (BOOST if boosted[i] else 0.0))
        dwelling = [wait[i] > 0 for i in range(n)]
        # robots dwelling at a station are immovable: top priority and a distance field that only allows staying
        prio_eff = [p + 10 * BOOST if dwelling[i] else p for i, p in enumerate(prio)]
        dfun = []
        for i in range(n):
            if dwelling[i]:
                dfun.append(lambda v, p=pos[i]: 0.0 if v == p else 1e9)
            elif goal[i] is not None:
                dfun.append(lambda v, g=goal[i]: D(g).get(v, 1e9))
            else:
                dfun.append(lambda v: 0.0)
        paths = [intended_path(m, pos[i], D(goal[i])) if goal[i] is not None else [] for i in range(n)]
        nxt = pibt_step(m, pos, prio_eff, dfun, tiebreak=lambda i, v: tb[i, v], paths=paths)
        check_step(pos, nxt)
        pos = nxt
        stalled = [mon[j].stalled(t + 1) for j in range(n)]
        for i in range(n):
            if wait[i] > 0:
                wait[i] -= 1
                if wait[i] == 0:
                    if phase[i] == "pick":
                        phase[i], goal[i] = "drop", task_of[i][1]
                        mon[i].reset_goal(t, goal[i])
                    else:
                        phase[i], goal[i], task_of[i] = None, None, None
                        elapsed[i] = 0.0
                        done += 1
                continue
            if goal[i] is not None and pos[i] == goal[i]:
                wait[i] = dwell
                continue
            if phase[i]:
                elapsed[i] += 1.0
                hn, bn = _deadlock_inputs(m, pos, prio, boosted, i, stalled)
                mon[i].update(t + 1, D(goal[i]).get(pos[i], 1e9), hn, bn)
        if not pending and not queue and all(p is None for p in phase):
            return GridResult(t + 1, [], sum(x.escalations for x in mon), done)
    return GridResult(None, [], sum(x.escalations for x in mon), done)
