"""Deterministic safety shield (design doc §9-10, FR-10).

Last stage before actuation: it can only *reduce* the commanded twist.  Inputs are this robot's own range ring
(independent of communication) and the last known chassis positions of peers (including stale / lost peers, which
are kept at their last reported position).  Rules:
  * forward / reverse: limit |v| so the robot can brake (at `brake_decel`) before closing to `stop_margin` from
    the nearest obstacle inside its lateral swath;
  * peers: no translation toward a peer whose chassis centre is closer than `peer_radius`;
  * in-place rotation: blocked if anything is inside the swept disc (+`turn_clear`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from fleet.world.robot_mjcf import MIR, MiR100


@dataclass
class SafetyParams:
    stop_margin: float = 0.20
    brake_decel: float = 0.8
    peer_radius: float = 1.15
    turn_clear: float = 0.03
    lateral_margin: float = 0.10
    peer_half_length: float = 0.49

    @classmethod
    def from_cfg(cls, c: dict) -> "SafetyParams":
        return cls(**{k: c[k] for k in ("stop_margin", "brake_decel", "peer_radius", "turn_clear") if k in c})


class SafetyShield:
    def __init__(self, p: SafetyParams, robot: MiR100 = MIR):
        self.p = p
        self.r = robot
        self.angles = robot.ray_angles()
        self.front = robot.wheel_dx + robot.half_length
        self.rear = robot.half_length - robot.wheel_dx
        self.interventions: dict[str, int] = {}

    def _note(self, why: str) -> None:
        self.interventions[why] = self.interventions.get(why, 0) + 1

    def free_distance(self, ranges: list[float], forward: bool) -> float:
        """Free travel distance along +x (forward) or -x (reverse) inside the robot's lateral swath."""
        half_w = self.r.half_width + self.p.lateral_margin
        best = float("inf")
        for a, rng in zip(self.angles, ranges):
            if rng is None or rng < 0:
                continue
            x, y = rng * math.cos(a), rng * math.sin(a)
            if abs(y) > half_w:
                continue
            if forward and x > 0:
                best = min(best, x - self.front)
            elif not forward and x < 0:
                best = min(best, -x - self.rear)
        return best

    def filter(self, v: float, w: float, ranges: list[float], peers_body: list[tuple[float, float]]) -> tuple[float, float, str | None]:
        """peers_body: peer chassis centres in this robot's base_link frame.  Returns (v, w, reason)."""
        why = None
        if abs(v) > 1e-6:
            fwd = v > 0
            free = self.free_distance(ranges, fwd)
            for px, py in peers_body:
                ahead = px if fwd else -px
                if ahead > 0 and abs(py) < self.r.half_width + self.p.peer_half_length + self.p.lateral_margin:
                    free = min(free, ahead - (self.front if fwd else self.rear) - self.p.peer_half_length)
                if math.hypot(px, py) < self.p.peer_radius and ahead > 0 and abs(math.atan2(py, ahead)) < math.radians(60):
                    free = min(free, 0.0)
            room = free - self.p.stop_margin
            vmax = math.sqrt(2 * self.p.brake_decel * room) if room > 0 else 0.0
            if abs(v) > vmax:
                v = math.copysign(vmax, v)
                why = "obstacle_ahead" if fwd else "obstacle_behind"
        if abs(w) > 1e-6 and abs(v) < 0.05:
            sweep = self.r.sweep_radius + self.p.turn_clear
            hit = any(rng is not None and 0 <= rng < sweep for rng in ranges)
            near_peer = any(math.hypot(px, py) < sweep + self.p.peer_half_length * 0.6 for px, py in peers_body)
            if hit or near_peer:
                w = 0.0
                why = "turn_blocked"
        if why:
            self._note(why)
        return v, w, why
