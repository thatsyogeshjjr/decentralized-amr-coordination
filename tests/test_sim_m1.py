"""M1 gate: three MiR100s drive scripted routes in the visual MuJoCo warehouse, lockstep over ZeroMQ."""
import numpy as np
import pytest

from simrun import run_scenario

pytestmark = pytest.mark.sim


def test_three_robots_drive_routes(tmp_path):
    r = run_scenario("configs/scenarios/m1_scripted.toml", tmp_path)
    assert not r["aborted"]
    assert r["n_robots"] == 3 and all(r["alive"])
    assert all(r["done_flags"]), "every robot must complete its route"
    assert r["sim_time"] < 120
    s = r["safety"]
    assert s["robot_contacts"] == 0 and s["static_contacts"] == 0 and s["eq8_events"] == 0
    # path length close to the route length (no wandering): routes are 18, 21, 30 m
    for got, want in zip(s["path_len"], [18.0, 21.0, 30.0]):
        assert want - 0.5 < got < want * 1.15
    # stays near the aisle centre lines
    t = np.load(tmp_path / "m1_scripted" / "seed_1" / "traj.npy")
    for i in range(3):
        x, y = t[:, 1 + 3 * i], t[:, 2 + 3 * i]
        dev = np.minimum(np.abs(x - np.round(x / 1.5) * 1.5), np.abs(y - np.round(y / 1.5) * 1.5))
        assert dev.max() < 0.25
    for name, code in r["agent_exit_codes"].items():
        assert code == 0, name
