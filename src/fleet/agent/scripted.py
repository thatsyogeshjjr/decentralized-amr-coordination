"""Scripted-route brain: follows a fixed node sequence from the config.  Used only to verify the
differential-drive model and motion controller in isolation (milestone M1), never for coordination."""
from __future__ import annotations

from fleet.agent.motion import MotionController, MotionParams, twist_to_wheels
from fleet.world.mapgrid import WarehouseMap


class ScriptedBrain:
    def __init__(self, cfg: dict, idx: int, name: str, wmap: WarehouseMap):
        self.map = wmap
        route = cfg["waypoints"].get(name, [])
        self.route = [r * wmap.W + c for r, c in route]
        self.k = 0
        self.ctrl = MotionController(MotionParams.from_cfg(cfg["motion"]), 1.0 / cfg["run"]["control_hz"])
        self.start_xy = None

    def step(self, sense: dict) -> dict:
        pose = tuple(sense["pose"])
        if self.start_xy is None:
            self.start_xy = pose[:2]
        if self.k >= len(self.route):
            v, w = self.ctrl.stop()
            return {"wheels": twist_to_wheels(v, w), "done": abs(sense["vel"][0]) < 0.02, "status": {"hud": "route complete"}}
        goal = self.map.pos[self.route[self.k]]
        (v, w), arrived = self.ctrl.track(pose, self.start_xy, goal, stop_at_goal=True)
        if arrived:
            self.start_xy = goal
            self.k += 1
        pts = [pose[:2]] + [self.map.pos[n] for n in self.route[self.k:]]
        return {"wheels": twist_to_wheels(v, w), "done": False,
                "status": {"hud": f"waypoint {self.k}/{len(self.route)}", "viz_path": [list(p) for p in pts]}}
