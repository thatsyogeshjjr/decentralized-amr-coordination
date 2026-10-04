"""Blocked-aisle detection from the range ring (FR-08), synthetic range data (no simulator)."""
import math

from fleet.agent.blockage import BlockageDetector
from fleet.common.config import PROJECT_ROOT
from fleet.world.mapgrid import WarehouseMap
from fleet.world.robot_mjcf import MIR

M = WarehouseMap.load(PROJECT_ROOT / "configs/maps/warehouse_small.txt")
N = lambda r, c: r * M.W + c


def ring(hits: dict[int, float]) -> list[float]:
    r = [-1.0] * MIR.range_rays
    for k, d in hits.items():
        r[k] = d
    return r


def test_pallet_in_aisle_detected_after_confirmation():
    d = BlockageDetector(M, confirm=3)
    x, y = M.pos[N(4, 3)]
    pose = (x, y, 0.0)                         # facing east along the central aisle; pallet on (4, 5)
    r = ring({0: 3.0 - 0.55, 1: 2.5, 23: 2.5})
    out = [d.update(pose, r, [], set(), t * 0.05)[0] for t in range(3)]
    assert out[0] == [] and out[1] == [] and out[2] == [N(4, 5)]


def test_peer_robot_and_racks_are_not_blockages():
    d = BlockageDetector(M, confirm=2)
    x, y = M.pos[N(4, 3)]
    pose = (x, y, 0.0)
    peer = (M.pos[N(4, 5)][0] + MIR.wheel_dx, y)
    rng = ring({0: 3.0 - 0.45, 6: 0.8, 18: 0.8})   # peer ahead; rack faces north/south of the aisle
    for t in range(5):
        nb, _ = d.update(pose, rng, [peer], set(), t * 0.05)
        assert nb == []


def test_cleared_node_is_revalidated():
    d = BlockageDetector(M, confirm=3, clear_confirm=4)
    x, y = M.pos[N(4, 3)]
    pose = (x, y, 0.0)
    blocked = {N(4, 5)}
    free = ring({0: 5.5})                       # ray passes over (4,5) centre (3 m) and continues to 5.5 m
    res = [d.update(pose, free, [], blocked, t * 0.05)[1] for t in range(4)]
    assert res[-1] == [N(4, 5)] and all(r == [] for r in res[:-1])
    # still occupied -> never cleared
    d2 = BlockageDetector(M, confirm=3, clear_confirm=4)
    occ = ring({0: 2.45})
    assert all(d2.update(pose, occ, [], blocked, t * 0.05)[1] == [] for t in range(10))
