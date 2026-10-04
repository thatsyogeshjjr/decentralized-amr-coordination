"""Differential-drive motion controller: track a straight segment to a target point (rotate-then-drive).

Output is a body twist (v, w) which `twist_to_wheels` converts to MiR100 wheel speeds.  Acceleration limits
model the drive's ramping; the safety shield runs *after* this controller and can only reduce motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from fleet.world.robot_mjcf import MIR


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class MotionParams:
    v_max: float = 1.0
    w_max: float = 1.2
    a_max: float = 0.6
    alpha_max: float = 2.0
    pos_tol: float = 0.08
    heading_tol: float = 0.30
    k_w: float = 2.5
    k_ct: float = 1.5
    v_rev_max: float = 0.5

    @classmethod
    def from_cfg(cls, c: dict) -> "MotionParams":
        return cls(**{k: c[k] for k in ("v_max", "w_max", "a_max", "alpha_max", "pos_tol", "heading_tol") if k in c})


class MotionController:
    def __init__(self, p: MotionParams, dt: float):
        self.p = p
        self.dt = dt
        self.v = 0.0
        self.w = 0.0
        self._seg = None
        self._rev = False

    def reset(self) -> None:
        self.v = self.w = 0.0
        self._seg = None
        self._rev = False

    @property
    def reversing(self) -> bool:
        return self._rev

    def _ramp(self, v_des: float, w_des: float) -> tuple[float, float]:
        dv = max(-self.p.a_max * self.dt, min(self.p.a_max * self.dt, v_des - self.v))
        dw = max(-self.p.alpha_max * self.dt, min(self.p.alpha_max * self.dt, w_des - self.w))
        self.v += dv
        self.w += dw
        return self.v, self.w

    def stop(self) -> tuple[float, float]:
        return self._ramp(0.0, 0.0)

    def track(self, pose: tuple[float, float, float], start: tuple[float, float], goal: tuple[float, float],
              stop_at_goal: bool = True, v_cap: float | None = None) -> tuple[tuple[float, float], bool]:
        """Drive along segment start->goal.  Returns ((v, w), arrived).

        A segment whose direction is behind the robot (> ~125 deg) when it begins is driven in reverse (the MiR100
        has a rear scanner and drives backwards), saving a 180-degree turn; reverse speed is limited."""
        seg = (tuple(start), tuple(goal))
        if seg != self._seg:
            self._seg = seg
            L0 = math.hypot(goal[0] - start[0], goal[1] - start[1])
            back = L0 > 0.2 and abs(wrap(math.atan2(goal[1] - start[1], goal[0] - start[0]) - pose[2])) > 2.2
            self._rev = back and stop_at_goal
        if self._rev:
            x, y, th = pose
            self.v = -self.v
            (v, w), arrived = self._track_fwd((x, y, wrap(th + math.pi)), start, goal, stop_at_goal,
                                              min(v_cap or self.p.v_max, self.p.v_rev_max))
            self.v = -self.v
            return (-v, w), arrived
        return self._track_fwd(pose, start, goal, stop_at_goal, v_cap)

    def _track_fwd(self, pose, start, goal, stop_at_goal, v_cap):
        x, y, th = pose
        gx, gy = goal
        dist = math.hypot(gx - x, gy - y)
        sx, sy = start
        L = math.hypot(gx - sx, gy - sy)
        if L < 0.2:
            # degenerate segment (re-centring on the current node): approach along the bearing
            if dist < self.p.pos_tol:
                return (self.stop() if stop_at_goal else (self.v, self.w)), True
            seg_th = math.atan2(gy - y, gx - x)
            sx, sy = x - 1e-3 * math.cos(seg_th), y - 1e-3 * math.sin(seg_th)
            L = dist
        else:
            seg_th = math.atan2(gy - sy, gx - sx)
        ux, uy = math.cos(seg_th), math.sin(seg_th)
        progress = (x - sx) * ux + (y - sy) * uy
        remaining = L - progress
        ct = -uy * (x - sx) + ux * (y - sy)            # cross-track error (left positive)
        if abs(remaining) < self.p.pos_tol and abs(ct) < 0.15:
            return (self.stop() if stop_at_goal else (self.v, self.w)), True
        eh = wrap(seg_th - th)
        vmax = self.p.v_max if v_cap is None else min(self.p.v_max, v_cap)
        if remaining < 0.0:
            # overshoot: back up along the segment line
            v_des = -min(0.25, math.sqrt(2 * self.p.a_max * 0.9 * -remaining))
            w_des = self.p.k_w * eh + self.p.k_ct * ct
            return self._ramp(v_des, max(-self.p.w_max, min(self.p.w_max, w_des))), False
        # pure pursuit: chase a point `ld` ahead of the projection on the segment (capped at the goal)
        ld = max(0.35, min(0.8, 0.6 * abs(self.v) + 0.35))
        a = min(L, progress + ld)
        tx, ty = sx + a * ux, sy + a * uy
        e = wrap(math.atan2(ty - y, tx - x) - th)
        align_tol = self.p.heading_tol if abs(self.v) > 0.05 else 0.12   # hysteresis: start straight, tolerate drift
        if abs(e) > align_tol and (remaining > 0.3 or abs(ct) > 0.12):
            # before turning in place, roll along the current heading onto the segment start (node centre): the
            # turning disc is only guaranteed clear of the racks when centred in the aisle cell
            lon = (sx - x) * math.cos(th) + (sy - y) * math.sin(th)
            if abs(lon) > 0.05 and progress < 0.3:
                return self._ramp(math.copysign(min(0.3, 1.2 * abs(lon) + 0.05), lon), 0.0), False
            if abs(self.v) > 0.05:
                return self._ramp(0.0, self.w), False
            # turn in place towards the pursuit point (MiR is a differential drive: no lateral motion)
            w_des = max(-self.p.w_max, min(self.p.w_max, self.p.k_w * e))
            return self._ramp(0.0, w_des), False
        if stop_at_goal:
            v_des = min(vmax, math.sqrt(2 * self.p.a_max * 0.9 * remaining), 0.3 + remaining)
        else:
            v_des = vmax
        v_des *= max(0.0, math.cos(e))
        w_des = max(-self.p.w_max, min(self.p.w_max, self.p.k_w * e))
        return self._ramp(v_des, w_des), False


def twist_to_wheels(v: float, w: float) -> tuple[float, float]:
    half = MIR.track / 2
    return (v - w * half) / MIR.wheel_radius, (v + w * half) / MIR.wheel_radius


def wheels_to_twist(wl: float, wr: float) -> tuple[float, float]:
    r = MIR.wheel_radius
    return r * (wl + wr) / 2, r * (wr - wl) / MIR.track
