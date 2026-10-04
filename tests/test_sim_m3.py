"""M3 gate (visual MuJoCo, accelerated): A* + decentralized PIBT conflict resolution, stop-and-wait baseline,
deterministic safety shield under forced convergence (FR-06, FR-10, PR-01, PR-04), and run reproducibility."""
import pytest

from simrun import run_scenario

pytestmark = pytest.mark.sim


def _agents(r):
    return list(r["agents"].values())


def test_crossing_conflicts_resolved_by_pibt(tmp_path):
    r = run_scenario("configs/scenarios/m3_crossing.toml", tmp_path)
    assert all(r["done_flags"]), "all goal sequences must complete"
    s = r["safety"]
    assert s["robot_contacts"] == 0 and s["eq8_events"] == 0 and s["static_contacts"] == 0
    for a in _agents(r):
        assert a["claim_latency_mean"] < 0.2           # PR-04 local conflict decision < 200 ms (sim time)
        assert a["compute_p99_ms"] < 10                 # SC-08: per-tick planning compute (p99) well inside the 50 ms tick
        assert a["net"]["barrier_timeouts"] == 0
    # conflicts actually happened and were resolved through the protocol
    assert sum(a["lock"]["denied"] for a in _agents(r)) > 0


def test_stop_and_wait_baseline_is_safe_and_slower(tmp_path):
    pibt = run_scenario("configs/scenarios/m3_crossing.toml", tmp_path, **{"run.name": "pibt"})
    sw = run_scenario("configs/scenarios/m3_crossing.toml", tmp_path, **{"run.name": "sw", "fleet.planner": "stop_wait",
                                                                          "run.duration": 400.0})
    assert all(sw["done_flags"])
    assert sw["safety"]["collisions"] == 0
    assert pibt["sim_time"] < sw["sim_time"]


def test_forced_convergence_stopped_by_safety_shield(tmp_path):
    """FR-10: with the claim protocol disabled (fault injection) robots drive at each other; the shield must
    prevent every contact and keep chassis centres beyond d_safe."""
    r = run_scenario("configs/scenarios/fr10_forced_convergence.toml", tmp_path)
    s = r["safety"]
    assert s["robot_contacts"] == 0 and s["eq8_events"] == 0
    assert s["min_pair_dist"] > 0.9
    # every robot was actively stopped by its shield (driving forward or, for short moves, in reverse)
    assert all(a["shield"].get("obstacle_ahead", 0) + a["shield"].get("obstacle_behind", 0) > 50 for a in _agents(r))


def test_runs_are_reproducible(tmp_path):
    a = run_scenario("configs/scenarios/m3_crossing.toml", tmp_path, **{"run.name": "a", "run.duration": 40.0,
                                                                         "run.stop_when_done": False})
    b = run_scenario("configs/scenarios/m3_crossing.toml", tmp_path, **{"run.name": "b", "run.duration": 40.0,
                                                                         "run.stop_when_done": False})
    assert a["final_poses"] == b["final_poses"]
    assert a["safety"]["path_len"] == b["safety"]["path_len"]
