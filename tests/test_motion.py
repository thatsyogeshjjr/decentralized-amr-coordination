import math

import pytest

from fleet.agent.motion import MotionController, MotionParams, twist_to_wheels, wheels_to_twist, wrap


def integrate(pose, v, w, dt):
    x, y, th = pose
    return x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, wrap(th + w * dt)


def drive(start_pose, seg_start, goal, steps=2000, dt=0.05):
    c = MotionController(MotionParams(), dt)
    pose = start_pose
    for k in range(steps):
        (v, w), arrived = c.track(pose, seg_start, goal)
        if arrived:
            return pose, k
        pose = integrate(pose, v, w, dt)
    return pose, None


def test_wheel_twist_roundtrip():
    for v, w in [(0.5, 0.0), (0.0, 1.0), (-0.3, 0.4), (1.0, -1.2)]:
        v2, w2 = wheels_to_twist(*twist_to_wheels(v, w))
        assert v2 == pytest.approx(v) and w2 == pytest.approx(w)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, -math.pi / 2, 2.5])
def test_reaches_goal_from_any_heading(yaw):
    pose, k = drive((0.0, 0.0, yaw), (0.0, 0.0), (4.5, 0.0))
    assert k is not None, "controller never reported arrival"
    assert math.hypot(pose[0] - 4.5, pose[1]) < 0.1


def test_corrects_lateral_offset_and_overshoot():
    # start offset 0.12 m off the aisle centre line and slightly past the goal: must not spin forever
    pose, k = drive((3.08, 0.12, 0.3), (0.0, 0.0), (3.0, 0.0))
    assert k is not None and k < 400
    assert math.hypot(pose[0] - 3.0, pose[1]) < 0.16


def test_respects_speed_and_accel_limits():
    p = MotionParams()
    dt = 0.05
    c = MotionController(p, dt)
    pose = (0.0, 0.0, 0.0)
    prev_v = 0.0
    for _ in range(300):
        (v, w), arrived = c.track(pose, (0.0, 0.0), (10.0, 0.0))
        assert v <= p.v_max + 1e-9 and abs(w) <= p.w_max + 1e-9
        assert abs(v - prev_v) <= p.a_max * dt + 1e-9
        prev_v = v
        pose = integrate(pose, v, w, dt)
        if arrived:
            break


def test_short_move_behind_is_driven_in_reverse():
    c = MotionController(MotionParams(), 0.05)
    pose = (0.0, 0.0, 0.0)                       # facing +x, target 1.5 m behind
    min_v = 0.0
    for k in range(400):
        (v, w), arrived = c.track(pose, (0.0, 0.0), (-1.5, 0.0))
        min_v = min(min_v, v)
        if arrived:
            break
        pose = integrate(pose, v, w, 0.05)
    assert arrived and math.hypot(pose[0] + 1.5, pose[1]) < 0.1
    assert abs(wrap(pose[2])) < 0.3                # never turned around
    assert -0.5 - 1e-9 <= min_v < -0.3             # reversed, within the reverse speed limit
