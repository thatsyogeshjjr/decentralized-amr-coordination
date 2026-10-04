"""A* and reference PIBT on the conflict cases named in the V&V plan (corridor swap, 4-way, crossing)."""
import numpy as np
import pytest

from fleet.common.config import PROJECT_ROOT
from fleet.learning.gridsim import run_goals, run_tasks

from fleet.agent.astar import astar, path_cost
from fleet.agent.pibt import check_step, pibt_step
from fleet.world.mapgrid import WarehouseMap


MAPFILE = PROJECT_ROOT / "configs/maps/warehouse_small.txt"


def N(m, r, c):
    return r * m.W + c


def run(m, starts, goals, max_steps=200):
    pos = list(starts)
    n = len(pos)
    dists = [m.bfs_dist(g) for g in goals]
    elapsed = [0.0] * n
    reached_at = [None] * n
    for t in range(max_steps):
        prio = [elapsed[i] + i * 1e-3 for i in range(n)]
        nxt = pibt_step(m, pos, prio, [lambda v, d=d: d.get(v, 1e9) for d in dists])
        check_step(pos, nxt)
        for i in range(n):
            assert nxt[i] == pos[i] or nxt[i] in m.adj[pos[i]]
        pos = nxt
        for i in range(n):
            if pos[i] == goals[i]:
                elapsed[i] = 0.0
                if reached_at[i] is None:
                    reached_at[i] = t + 1
            else:
                elapsed[i] += 1.0
        if all(p == g for p, g in zip(pos, goals)):
            return t + 1, reached_at
    return None, reached_at


def test_astar_shortest_and_turn_aware():
    m = WarehouseMap.from_text("""
#######
#.....#
#.###.#
#.....#
#######""")
    p = astar(m, N(m, 1, 1), N(m, 3, 5))
    assert p[0] == N(m, 1, 1) and p[-1] == N(m, 3, 5) and len(p) == 7
    for u, v in zip(p[:-1], p[1:]):
        assert v in m.adj[u]
    # exactly one 90-degree turn on the chosen route (L-shape, never a zig-zag)
    assert path_cost(m, p) == pytest.approx(6 * 1.5 + 1.2)
    # blocking the top-right corner forces the other L
    p2 = astar(m, N(m, 1, 1), N(m, 3, 5), blocked={N(m, 1, 5)})
    assert N(m, 1, 5) not in p2 and N(m, 3, 1) in p2
    assert astar(m, N(m, 1, 1), N(m, 3, 5), blocked={N(m, 1, 2), N(m, 2, 1)}) is None


CORRIDOR_POCKET = """
#######
###.###
#.....#
#######"""
PLUS = """
#########
####.####
####.####
####.####
#.......#
####.####
####.####
####.####
#########"""


def test_plain_pibt_fails_on_tree_but_never_conflicts():
    """Known PIBT limitation (reachability needs biconnected graphs): it can stall in dead ends - safely."""
    m = WarehouseMap.from_text(CORRIDOR_POCKET)
    a, b = N(m, 2, 1), N(m, 2, 5)
    steps, _ = run(m, [a, b], [b, a], max_steps=60)
    assert steps is None


@pytest.mark.parametrize("seed", range(10))
def test_corridor_swap_with_pocket_recovered_by_escalation(seed):
    m = WarehouseMap.from_text(CORRIDOR_POCKET)
    a, b = N(m, 2, 1), N(m, 2, 5)
    r = run_goals(m, [a, b], [b, a], seed=seed)
    assert r.steps is not None and r.steps <= 20
    assert r.deadlock_escalations >= 1


def test_corridor_swap_without_pocket_is_impossible_and_stays_safe():
    m = WarehouseMap.from_text("""
#######
#.....#
#######""")
    r = run_goals(m, [N(m, 1, 1), N(m, 1, 5)], [N(m, 1, 5), N(m, 1, 1)], max_steps=80)
    assert r.steps is None    # no solution exists; every step was conflict-checked inside run_goals


@pytest.mark.parametrize("seed", range(10))
def test_four_way_intersection(seed):
    m = WarehouseMap.from_text(PLUS)
    starts = [N(m, 1, 4), N(m, 4, 7), N(m, 7, 4), N(m, 4, 1)]
    goals = [N(m, 7, 4), N(m, 4, 1), N(m, 1, 4), N(m, 4, 7)]
    r = run_goals(m, starts, goals, seed=seed, max_steps=200)
    assert r.steps is not None and r.steps <= 40


@pytest.mark.parametrize("seed", range(5))
def test_crossing_traffic_in_warehouse(seed):
    m = WarehouseMap.load(MAPFILE)
    starts = [N(m, 1, 1), N(m, 1, 11), N(m, 7, 1), N(m, 7, 11), N(m, 4, 6)]
    goals = [N(m, 7, 11), N(m, 7, 1), N(m, 1, 11), N(m, 1, 1), N(m, 1, 6)]
    r = run_goals(m, starts, goals, seed=seed)
    assert r.steps is not None and r.steps <= 40


def test_lifelong_pickup_delivery_graph_level():
    m = WarehouseMap.load(MAPFILE)
    rng = np.random.default_rng(0)
    P, D = m.stations("pickup"), m.stations("drop")
    tasks = [(int(k * 4), int(rng.choice(P)), int(rng.choice(D))) for k in range(20)]
    r = run_tasks(m, [m.starts[i] for i in range(3)], tasks, seed=0)
    assert r.steps is not None and r.completed == 20
