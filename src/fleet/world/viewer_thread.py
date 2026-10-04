"""Viewer thread: renders world snapshots in the MuJoCo passive viewer so rendering never stalls lockstep."""
from __future__ import annotations

import threading

import mujoco
import numpy as np

from fleet.world.mjcf_builder import ROBOT_COLORS


class ViewerThread:
    """Renders snapshots of the world in the MuJoCo passive viewer without stalling the lockstep loop."""

    def __init__(self, model: mujoco.MjModel, viewer, d_vis: mujoco.MjData):
        self.m, self.viewer, self.d = model, viewer, d_vis
        self._lock = threading.Lock()
        self._snap: dict | None = None
        self._event = threading.Event()
        self._stop = False
        self.frames = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def post(self, snap: dict) -> None:
        with self._lock:
            self._snap = snap
        self._event.set()

    def stop(self) -> None:
        self._stop = True
        self._event.set()
        self.thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop:
            self._event.wait(0.5)
            self._event.clear()
            with self._lock:
                snap, self._snap = self._snap, None
            if snap is None or not self.viewer.is_running():
                continue
            self.d.qpos[:] = snap["qpos"]
            self.d.mocap_pos[:] = snap["mocap"]
            self.d.time = snap["time"]
            mujoco.mj_kinematics(self.m, self.d)
            self.viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, snap["text"], None)])
            with self.viewer.lock():
                scn = self.viewer.user_scn
                scn.ngeom = 0
                for i, path, held in snap["viz"]:
                    col = np.array([*ROBOT_COLORS[i % len(ROBOT_COLORS)], 0.6], dtype=np.float32)
                    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                        if scn.ngeom >= scn.maxgeom - 1:
                            break
                        g = scn.geoms[scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), col)
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.05,
                                             np.array([x1, y1, 0.03]), np.array([x2, y2, 0.03]))
                        scn.ngeom += 1
                    for (x, y) in held:
                        if scn.ngeom >= scn.maxgeom - 1:
                            break
                        g = scn.geoms[scn.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.6, 0.6, 0.004]),
                                            np.array([x, y, 0.006]), np.eye(3).flatten(), col * np.array([1, 1, 1, 0.45], dtype=np.float32))
                        scn.ngeom += 1
            self.viewer.sync()
            self.frames += 1
