import math

import pytest

from fleet.agent.safety_shield import SafetyParams, SafetyShield
from fleet.world.robot_mjcf import MIR

NO_HIT = [-1.0] * MIR.range_rays


def ring_with(angle_deg: float, dist: float) -> list[float]:
    r = list(NO_HIT)
    r[int(round(angle_deg / 15)) % MIR.range_rays] = dist
    return r


@pytest.fixture
def shield():
    return SafetyShield(SafetyParams())


def test_free_space_passes_through(shield):
    v, w, why = shield.filter(0.8, 0.3, NO_HIT, [])
    assert (v, w, why) == (0.8, 0.3, None)


def test_obstacle_ahead_stops_before_contact(shield):
    front = MIR.wheel_dx + MIR.half_length
    v, _, why = shield.filter(1.0, 0.0, ring_with(0, front + 0.15), [])
    assert v == 0.0 and why == "obstacle_ahead"
    v, _, _ = shield.filter(1.0, 0.0, ring_with(0, front + 0.6), [])
    assert 0 < v < 1.0 and v == pytest.approx(math.sqrt(2 * 0.8 * 0.4))
    # obstacle outside the lateral swath (e.g. rack beside the aisle) does not limit forward motion
    v, _, _ = shield.filter(1.0, 0.0, ring_with(90, 0.8), [])
    assert v == 1.0


def test_reverse_uses_rear_rays(shield):
    rear = MIR.half_length - MIR.wheel_dx
    v, _, why = shield.filter(-0.3, 0.0, ring_with(180, rear + 0.1), [])
    assert v == 0.0 and why == "obstacle_behind"
    assert shield.filter(0.5, 0.0, ring_with(180, rear + 0.1), [])[0] == 0.5


def test_peer_position_vetoes_motion_even_without_range_hit(shield):
    # peer (e.g. known from messages) directly ahead at 1.0 m: must not drive toward it
    v, _, _ = shield.filter(0.7, 0.0, NO_HIT, [(1.0, 0.0)])
    assert v == 0.0
    # same peer behind us: forward motion allowed
    assert shield.filter(0.7, 0.0, NO_HIT, [(-1.0, 0.0)])[0] == 0.7
    # peer in a parallel aisle (1.5 m to the side) does not block
    assert shield.filter(0.7, 0.0, NO_HIT, [(0.5, 1.5)])[0] == 0.7


def test_rotation_blocked_inside_sweep(shield):
    v, w, why = shield.filter(0.0, 1.0, ring_with(45, MIR.sweep_radius), [])
    assert w == 0.0 and why == "turn_blocked"
    # aisle racks at 0.8 m to the side leave room to turn
    assert shield.filter(0.0, 1.0, ring_with(90, 0.8), [])[1] == 1.0


def test_shield_only_reduces(shield):
    import random
    rnd = random.Random(1)
    for _ in range(500):
        ranges = [rnd.choice([-1.0, rnd.uniform(0.2, 5.0)]) for _ in range(MIR.range_rays)]
        peers = [(rnd.uniform(-3, 3), rnd.uniform(-3, 3)) for _ in range(rnd.randint(0, 3))]
        v0, w0 = rnd.uniform(-0.5, 1.0), rnd.uniform(-1.2, 1.2)
        v, w, _ = shield.filter(v0, w0, ranges, peers)
        assert abs(v) <= abs(v0) + 1e-12 and abs(w) <= abs(w0) + 1e-12
        assert v == 0 or math.copysign(1, v) == math.copysign(1, v0)
