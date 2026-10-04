"""Generate the MuJoCo warehouse world (MJCF) from a WarehouseMap and a fleet description."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fleet.world.mapgrid import WarehouseMap
from fleet.world.robot_mjcf import mir_actuators, mir_assets, mir_body

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "mir100"

ROBOT_COLORS = [
    (0.90, 0.30, 0.10), (0.10, 0.45, 0.90), (0.15, 0.70, 0.30), (0.75, 0.20, 0.75),
    (0.95, 0.75, 0.05), (0.05, 0.75, 0.75), (0.55, 0.35, 0.15), (0.40, 0.40, 0.95),
]
STATION_RGBA = {"pickup": "0.20 0.75 0.30 0.85", "drop": "0.20 0.45 0.95 0.85", "charger": "1.0 0.80 0.10 0.85",
                "park": "0.55 0.55 0.60 0.85"}
BLOCK_PARK = (-50.0, -50.0)


def robot_name(i: int) -> str:
    return f"amr{i}"


def _rack_xml(m: WarehouseMap, r: int, c: int, rng: np.random.Generator) -> str:
    x, y = m.rack_world(r, c)
    h = m.cell / 2
    border = r in (0, m.H - 1) or c in (0, m.W - 1)
    col = f'<geom type="box" size="{h - 0.05} {h - 0.05} 1.0" pos="{x} {y} 1.0" contype="4" conaffinity="2" group="3" rgba="0.5 0.5 0.5 0.3"/>'
    if border:
        return col + (f'<geom type="box" size="{h} {h} 0.75" pos="{x} {y} 0.75" material="wall" '
                      f'contype="0" conaffinity="0" group="1"/>')
    parts = [col]
    s = h - 0.08
    for dx, dy in ((-s, -s), (-s, s), (s, -s), (s, s)):
        parts.append(f'<geom type="box" size="0.04 0.04 1.1" pos="{x + dx} {y + dy} 1.1" material="rack_frame" '
                     f'contype="0" conaffinity="0" group="1"/>')
    for z in (0.12, 0.85, 1.55):
        parts.append(f'<geom type="box" size="{s + 0.04} {s + 0.04} 0.025" pos="{x} {y} {z}" material="rack_beam" '
                     f'contype="0" conaffinity="0" group="1"/>')
        for _ in range(int(rng.integers(1, 4))):
            bw, bd, bh = rng.uniform(0.18, 0.32, size=3)
            px, py = rng.uniform(-s + bw, s - bw), rng.uniform(-s + bd, s - bd)
            tone = rng.uniform(0.55, 0.8)
            parts.append(f'<geom type="box" size="{bw:.3f} {bd:.3f} {bh:.3f}" pos="{x + px:.3f} {y + py:.3f} {z + 0.025 + bh:.3f}" '
                         f'rgba="{tone:.2f} {tone * 0.75:.2f} {tone * 0.45:.2f} 1" contype="0" conaffinity="0" group="1"/>')
    return "".join(parts)


def build_world_xml(m: WarehouseMap, robots: list[tuple[int, float]], n_blocks: int = 4, seed: int = 0) -> str:
    """robots: list of (start_node, yaw) per robot index. Returns a full MJCF document."""
    rng = np.random.default_rng(seed)
    ex, ey = m.extent()
    cx, cy = ex / 2, ey / 2
    meshdir = ASSET_DIR.as_posix()

    statics = [_rack_xml(m, r, c, rng) for r, c in m.racks]
    floor_marks = []
    for n, k in m.kind.items():
        x, y = m.pos[n]
        if k in STATION_RGBA:
            floor_marks.append(f'<geom type="box" size="{m.cell / 2 - 0.08} {m.cell / 2 - 0.08} 0.002" pos="{x} {y} 0.002" '
                               f'rgba="{STATION_RGBA[k]}" contype="0" conaffinity="0" group="1"/>')
        else:
            floor_marks.append(f'<geom type="cylinder" size="0.06 0.0015" pos="{x} {y} 0.0015" rgba="0.95 0.85 0.2 0.9" '
                               f'contype="0" conaffinity="0" group="1"/>')
    for u, v in m.edges():
        (x1, y1), (x2, y2) = m.pos[u], m.pos[v]
        L = math.hypot(x2 - x1, y2 - y1)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        floor_marks.append(f'<geom type="box" size="{L / 2} 0.012 0.001" pos="{(x1 + x2) / 2} {(y1 + y2) / 2} 0.001" '
                           f'euler="0 0 {ang}" rgba="0.95 0.85 0.2 0.35" contype="0" conaffinity="0" group="1"/>')

    blocks = []
    for k in range(n_blocks):
        blocks.append(f"""
    <body name="block{k}" mocap="true" pos="{BLOCK_PARK[0] - 3 * k} {BLOCK_PARK[1]} 0">
      <geom type="box" size="0.55 0.45 0.07" pos="0 0 0.07" rgba="0.55 0.38 0.18 1" contype="4" conaffinity="2"/>
      <geom type="box" size="0.45 0.40 0.30" pos="0 0 0.44" rgba="0.80 0.20 0.15 1" contype="4" conaffinity="2"/>
      <geom type="box" size="0.46 0.41 0.04" pos="0 0 0.70" rgba="1 1 1 1" contype="0" conaffinity="0"/>
    </body>""")

    bodies, acts = [], []
    for i, (node, yaw) in enumerate(robots):
        x, y = m.pos[node]
        name = robot_name(i)
        bodies.append(mir_body(name, x, y, yaw, ROBOT_COLORS[i % len(ROBOT_COLORS)]))
        acts.append(mir_actuators(name))

    return f"""<mujoco model="warehouse">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="1600" offheight="1000" azimuth="90" elevation="-55"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
    <map znear="0.05" zfar="200"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.75 0.80 0.88" rgb2="0.35 0.40 0.48" width="256" height="256"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.62 0.63 0.64" rgb2="0.58 0.59 0.60" width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="{m.W} {m.H}" reflectance="0.05"/>
    <material name="wall" rgba="0.70 0.70 0.68 1"/>
    <material name="rack_frame" rgba="0.15 0.30 0.65 1"/>
    <material name="rack_beam" rgba="0.95 0.50 0.10 1"/>
    {mir_assets(meshdir)}
  </asset>
  <worldbody>
    <light name="sun" pos="{cx} {cy} 25" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="{cx - ex / 3} {cy} 12" dir="0.3 0 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="{ex / 2 + m.cell} {ey / 2 + m.cell} 0.1" pos="{cx} {cy} 0" material="floor"
          contype="1" conaffinity="1" friction="1.0 0.02 0.001"/>
    {''.join(floor_marks)}
    {''.join(statics)}
    {''.join(blocks)}
    {''.join(bodies)}
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""
