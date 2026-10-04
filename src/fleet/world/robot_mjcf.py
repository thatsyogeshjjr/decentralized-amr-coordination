"""MiR100 (Mobile Industrial Robots) MJCF model.

Geometry and inertial parameters are transcribed from DFKI's `mir_description/urdf/include/mir_v1.urdf.xacro`
(BSD-3, copy in assets/mir100/).  Visual meshes are the original STL files.  Collision is simplified to a
chassis box, two drive-wheel cylinders and four frictionless caster spheres (standard MuJoCo practice for
passive casters).  The robot frame is the URDF `base_link`: origin on the floor below the drive-wheel axle.

Collision bit layout used across the world:
  bit 1 (1): ground contact   - floor <-> wheels/casters
  bit 2 (2): robot chassis    - chassis <-> chassis, chassis <-> static obstacles
  bit 3 (4): static obstacles - racks, walls, dynamic blockage pallets
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MiR100:
    base_mass: float = 77.0
    wheel_radius: float = 0.0625
    wheel_width: float = 0.032
    wheel_dx: float = 0.037646        # mesh/chassis offset ahead of the drive axle
    wheel_dy: float = 0.222604        # half wheel separation
    caster_radius: float = 0.0625
    front_caster_dx: float = 0.3037
    back_caster_dx: float = 0.3078
    caster_dy: float = 0.203
    laser_dx: float = 0.392
    laser_dy: float = 0.2358
    laser_dz: float = 0.1914
    half_length: float = 0.449        # from mesh bounds (0.898 m)
    half_width: float = 0.2925        # from mesh bounds (0.585 m)
    height: float = 0.354
    max_speed: float = 1.5            # MiR100 datasheet top speed, m/s
    range_rays: int = 24              # 360 deg ring, 15 deg pitch (two corner S300s give ~360 deg coverage)
    range_height: float = 0.19
    range_cutoff: float = 6.0
    susp_stiffness: float = 10000.0   # N/m drive-unit spring
    susp_preload: float = 0.030       # m spring pre-compression at nominal ride height (~300 N per wheel)
    susp_damping: float = 600.0       # N s/m
    suspension: bool = True
    caster_gap: float = 0.0
    wheel_friction: float = 1.2
    kv: float = 10.0                  # wheel velocity-servo gain (N m per rad/s); tuned for no stick-slip
    torque_limit: float = 30.0

    @property
    def track(self) -> float:
        return 2 * self.wheel_dy

    @property
    def sweep_radius(self) -> float:
        """Radius of the disc swept when turning in place about base_link."""
        fx = self.wheel_dx + self.half_length
        return math.hypot(fx, self.half_width)

    def ray_angles(self) -> list[float]:
        return [2 * math.pi * k / self.range_rays for k in range(self.range_rays)]

    def body_extent(self, angle: float) -> float:
        """Distance from base_link origin to the chassis boundary along body-frame direction `angle`."""
        c, s = math.cos(angle), math.sin(angle)
        cx = self.wheel_dx
        # ray (t*c, t*s) exits box |x - cx| <= hl, |y| <= hw
        ts = []
        if abs(c) > 1e-9:
            for bx in (cx + self.half_length, cx - self.half_length):
                t = bx / c
                if t > 0 and abs(t * s) <= self.half_width + 1e-9:
                    ts.append(t)
        if abs(s) > 1e-9:
            for by in (self.half_width, -self.half_width):
                t = by / s
                if t > 0 and abs(t * c - cx) <= self.half_length + 1e-9:
                    ts.append(t)
        return min(ts)


MIR = MiR100()


def mir_assets(meshdir: str) -> str:
    return f"""
    <mesh name="mir_base" file="{meshdir}/mir_100_base.stl"/>
    <mesh name="mir_wheel" file="{meshdir}/wheel.stl"/>
    <mesh name="mir_caster" file="{meshdir}/caster_wheel_base.stl"/>
    <mesh name="mir_sick" file="{meshdir}/sick_lms-100.stl"/>
    <material name="mir_white" rgba="0.92 0.92 0.90 1" specular="0.4" shininess="0.5"/>
    <material name="mir_black" rgba="0.08 0.08 0.08 1"/>
    <material name="mir_silver" rgba="0.65 0.65 0.68 1" specular="0.6"/>
    <material name="sick_yellow" rgba="0.95 0.78 0.10 1"/>
    """


def mir_body(name: str, x: float, y: float, yaw: float, rgba: tuple[float, float, float], p: MiR100 | None = None) -> str:
    """One MiR100 body with free joint, wheels, casters, range-ring sites and a status/cargo marker."""
    p = p or MIR
    r, g, b = rgba
    m = p.base_mass
    ixx = m / 12 * (0.30 ** 2 + 0.20 ** 2)
    iyy = m / 12 * (0.50 ** 2 + 0.20 ** 2)
    izz = m / 12 * (0.50 ** 2 + 0.30 ** 2)
    fx = p.front_caster_dx + p.wheel_dx
    bx = -p.back_caster_dx + p.wheel_dx
    caster_hub_z = p.caster_radius + 0.094
    sites = []
    for k, a in enumerate(p.ray_angles()):
        sites.append(
            f'<site name="{name}_ray{k}" pos="0 0 {p.range_height}" zaxis="{math.cos(a):.6f} {math.sin(a):.6f} 0" '
            f'size="0.005" rgba="1 0 0 0" group="4"/>'
        )
    casters = []
    # Casters are frictionless swivels (priority=1 makes their condim/friction win over the floor's in the contact
    # pair).  The platform stands on its four casters; the drive wheels are on a sprung suspension (see wheels).
    caster_gap = p.caster_gap
    for tag, cx, cy in (("fl", fx, p.caster_dy), ("fr", fx, -p.caster_dy), ("bl", bx, p.caster_dy), ("br", bx, -p.caster_dy)):
        casters.append(
            f'<geom name="{name}_caster_{tag}" type="sphere" size="{p.caster_radius - caster_gap}" pos="{cx} {cy} {p.caster_radius}" '
            f'condim="1" priority="1" friction="0 0 0" contype="1" conaffinity="0" group="3" mass="0.5" rgba="0 0 0 0"/>'
            f'<geom type="mesh" mesh="mir_caster" pos="{cx} {cy} {caster_hub_z}" material="mir_silver" '
            f'contype="0" conaffinity="0" group="1" mass="0"/>'
        )
    wheels = []
    for tag, sy in (("wl", 1.0), ("wr", -1.0)):
        susp = (f'<joint name="{name}_{tag}_susp" type="slide" axis="0 0 1" range="-0.05 0.02" '
                f'stiffness="{p.susp_stiffness}" springref="{-p.susp_preload}" damping="{p.susp_damping}"/>') if p.suspension else ""
        wheels.append(f"""
      <body name="{name}_{tag}" pos="0 {sy * p.wheel_dy} {p.wheel_radius}">
        <!-- sprung drive unit (as on the real MiR100): ~{p.susp_stiffness * p.susp_preload:.0f} N preload per wheel keeps
             traction independent of pitch and of how the static load splits between casters and wheels -->
        {susp}
        <joint name="{name}_{tag}" type="hinge" axis="0 1 0" damping="0.05"/>
        <!-- sphere collision = single tyre contact patch (a cylinder yields two edge contacts that scrub against
             each other during in-place turns and cause stick-slip) -->
        <geom type="sphere" size="{p.wheel_radius}" mass="1.0" friction="{p.wheel_friction} 0.02 0.001" contype="1" conaffinity="0"
              group="3" rgba="0.5 0.5 0.5 0.3"/>
        <geom type="mesh" mesh="mir_wheel" material="mir_black" contype="0" conaffinity="0" group="1" mass="0"/>
      </body>""")
    yaw_deg = math.degrees(yaw)
    return f"""
    <body name="{name}" pos="{x} {y} 0.001" euler="0 0 {yaw_deg}">
      <freejoint name="{name}_free"/>
      <inertial pos="{-0.05 + p.wheel_dx} 0 0.15" mass="{m}" diaginertia="{ixx:.4f} {iyy:.4f} {izz:.4f}"/>
      <geom name="{name}_shell" type="mesh" mesh="mir_base" pos="{p.wheel_dx} 0 0" material="mir_white"
            contype="0" conaffinity="0" group="1" mass="0"/>
      <geom name="{name}_chassis" type="box" size="{p.half_length} {p.half_width} 0.15"
            pos="{p.wheel_dx} 0 0.20" contype="2" conaffinity="6" group="3" mass="0" rgba="0.5 0.5 0.5 0.3"/>
      <geom name="{name}_stripe" type="box" size="{p.half_length * 0.7} {p.half_width * 0.75} 0.004"
            pos="{p.wheel_dx} 0 {p.height + 0.004}" rgba="{r} {g} {b} 1" contype="0" conaffinity="0" group="1" mass="0"/>
      <geom name="{name}_cargo" type="box" size="0.30 0.22 0.15" pos="{p.wheel_dx} 0 {p.height + 0.16}"
            rgba="0.72 0.55 0.33 0" contype="0" conaffinity="0" group="1" mass="0"/>
      <geom type="mesh" mesh="mir_sick" pos="{p.laser_dx + p.wheel_dx} {p.laser_dy} {p.laser_dz}" euler="0 0 45"
            material="sick_yellow" contype="0" conaffinity="0" group="1" mass="0"/>
      <geom type="mesh" mesh="mir_sick" pos="{-p.laser_dx + p.wheel_dx} {-p.laser_dy} {p.laser_dz}" euler="0 0 -135"
            material="sick_yellow" contype="0" conaffinity="0" group="1" mass="0"/>
      {''.join(casters)}
      {''.join(sites)}
      {''.join(wheels)}
    </body>"""


def mir_actuators(name: str, p: MiR100 | None = None) -> str:
    """Velocity-controlled drive wheels.  The range ring is cast once per control tick with mj_multiRay from
    the `{name}_ray{k}` site directions (see World.ranges) rather than as per-substep rangefinder sensors."""
    p = p or MIR
    wmax = p.max_speed / p.wheel_radius * 1.1
    return "".join(
        f'<velocity name="{name}_{t}" joint="{name}_{t}" kv="{p.kv}" ctrlrange="{-wmax:.2f} {wmax:.2f}" '
        f'forcerange="{-p.torque_limit} {p.torque_limit}" ctrllimited="true" forcelimited="true"/>'
        for t in ("wl", "wr")
    )
