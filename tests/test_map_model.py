import math

import mujoco
import pytest

from fleet.common.config import PROJECT_ROOT
from fleet.world.mapgrid import WarehouseMap
from fleet.world.mjcf_builder import build_world_xml
from fleet.world.robot_mjcf import MIR

MAP = PROJECT_ROOT / "configs/maps/warehouse_small.txt"


def test_map_parse_and_graph():
    m = WarehouseMap.load(MAP)
    assert (m.H, m.W) == (9, 13)
    assert sorted(m.starts) == [0, 1, 2]
    assert len(m.stations("pickup")) == 3 and len(m.stations("drop")) == 3 and len(m.stations("charger")) == 2
    assert len(m.stations("park")) == 2
    # undirected, 4-connected, only passable cells
    for u, nb in m.adj.items():
        for v in nb:
            assert u in m.adj[v]
            (x1, y1), (x2, y2) = m.pos[u], m.pos[v]
            assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(m.cell)
    # whole aisle network is connected
    assert len(m.bfs_dist(m.starts[0])) == len(m.pos)


def test_bfs_respects_blocked_nodes():
    m = WarehouseMap.from_text("#####\n#0..#\n#####")
    a, mid, b = sorted(m.pos)
    assert m.bfs_dist(b)[a] == 2
    assert a not in m.bfs_dist(b, blocked={mid})


def test_bad_symbol_rejected():
    with pytest.raises(ValueError):
        WarehouseMap.from_text("#x#")


def test_mir100_parameters_match_urdf():
    assert MIR.wheel_radius == 0.0625 and MIR.wheel_dy == 0.222604 and MIR.base_mass == 77.0
    assert MIR.body_extent(0.0) == pytest.approx(MIR.half_length + MIR.wheel_dx)
    assert MIR.body_extent(math.pi) == pytest.approx(MIR.half_length - MIR.wheel_dx)
    assert MIR.body_extent(math.pi / 2) == pytest.approx(MIR.half_width)


def test_range_ring_sees_racks_walls_and_robots():
    """Regression: mj_ray ignores alpha-0 geoms, so collision geoms must be ray-visible (static query, no stepping)."""
    from fleet.common.config import load_config
    from fleet.world.sim_server import World
    w = World(load_config("configs/scenarios/m1_scripted.toml"))
    r0 = w.ranges(0)                      # amr0 at (4,2) facing east; amr1 at (4,6) 6 m ahead
    rear = MIR.half_length - MIR.wheel_dx
    assert r0[0] == pytest.approx(6.0 - rear, abs=0.05)          # amr1's chassis box
    assert r0[6] == pytest.approx(0.8, abs=0.05)                  # rack face north of the aisle: 1.5 - 0.7
    assert r0[12] == pytest.approx(3.0 - 0.7, abs=0.05)           # pickup wall bay: wall box face west
    assert all(x > 0 for x in r0)


def test_world_model_compiles_with_mir_meshes():
    """Compile only (no stepping): checks the generated MJCF, mesh import and mesh placement."""
    m = WarehouseMap.load(MAP)
    xml = build_world_xml(m, [(m.starts[i], 0.0) for i in range(3)])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    assert model.nu == 6 and model.nmesh == 4
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "amr0_shell")
    mid = model.geom_dataid[gid]
    a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[a:a + n] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    x0, y0 = m.pos[m.starts[0]]
    # MiR100 shell: 0.898 x 0.585 x 0.354 m, centred wheel_dx ahead of the drive axle
    assert (v[:, 0].max() - v[:, 0].min()) == pytest.approx(0.898, abs=0.01)
    assert (v[:, 1].max() - v[:, 1].min()) == pytest.approx(0.585, abs=0.01)
    assert (v[:, 0].max() + v[:, 0].min()) / 2 == pytest.approx(x0 + MIR.wheel_dx, abs=0.01)
    assert v[:, 2].max() == pytest.approx(0.355, abs=0.01)
