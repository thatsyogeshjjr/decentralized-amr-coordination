"""Blocked-aisle detection from the robot's own range ring (design doc §10.7, FR-08).

A range hit is *unexplained* when it lies in the interior of an aisle cell (well away from racks/walls, which sit
in their own cells) and no known robot (fresh, stale or lost) is within reach of it.  An aisle node with
unexplained hits on `confirm` consecutive control ticks is declared blocked: it is invalidated for `ttl` seconds
and a BlockageEvent is broadcast so peers invalidate it too.  A blocked node that is later observed clear (rays
passing over its centre and continuing well beyond it) for `clear_confirm` ticks is re-validated and an
unblock event is broadcast.
"""
from __future__ import annotations

import math

from fleet.world.mapgrid import WarehouseMap
from fleet.world.robot_mjcf import MIR


class BlockageDetector:
    def __init__(self, m: WarehouseMap, confirm: int = 3, clear_confirm: int = 20, interior: float = 0.6,
                 peer_reach: float = 0.8, max_range: float = 4.5):
        self.m = m
        self.confirm, self.clear_confirm = confirm, clear_confirm
        self.interior, self.peer_reach, self.max_range = interior, peer_reach, max_range
        self.angles = MIR.ray_angles()
        self.hits: dict[int, int] = {}
        self.clear: dict[int, int] = {}
        self.detections: list[dict] = []

    def update(self, pose, ranges, peer_centres: list[tuple[float, float]], known_blocked: set[int],
               now: float) -> tuple[list[int], list[int]]:
        """Returns (newly_blocked_nodes, newly_cleared_nodes)."""
        x, y, th = pose
        seen: set[int] = set()
        cleared_obs: set[int] = set()
        cell = self.m.cell
        for a, r in zip(self.angles, ranges):
            ca, sa = math.cos(th + a), math.sin(th + a)
            if 0 < r <= self.max_range:
                hx, hy = x + r * ca, y + r * sa
                n = self.m.nearest_node(hx, hy)
                nx, ny = self.m.pos[n]
                if max(abs(hx - nx), abs(hy - ny)) < self.interior and \
                        all(math.hypot(hx - px, hy - py) > self.peer_reach for px, py in peer_centres):
                    seen.add(n)
            # clear evidence: the ray passed over a blocked node's centre and continued > 0.8 m beyond it
            reach = r if r > 0 else MIR.range_cutoff
            for b in known_blocked:
                bx, by = self.m.pos[b]
                along = (bx - x) * ca + (by - y) * sa
                if 0.5 < along < reach - 0.8 and abs(-(bx - x) * sa + (by - y) * ca) < 0.2 and along < self.max_range:
                    cleared_obs.add(b)
        new_blocked, new_clear = [], []
        for n in list(self.hits):
            if n not in seen:
                del self.hits[n]
        for n in seen:
            self.hits[n] = self.hits.get(n, 0) + 1
            if self.hits[n] == self.confirm and n not in known_blocked:
                new_blocked.append(n)
                self.detections.append({"t": round(now, 3), "node": n})
        for n in list(self.clear):
            if n not in cleared_obs or n in seen:
                del self.clear[n]
        for n in cleared_obs - seen:
            self.clear[n] = self.clear.get(n, 0) + 1
            if self.clear[n] == self.clear_confirm:
                new_clear.append(n)
        return new_blocked, new_clear
