"""M4 gate (visual MuJoCo, accelerated): dynamic pickup/delivery orders allocated peer-to-peer; reassignment after
a robot crash (FR-01, FR-04, FR-09, PR-05; acceptance items 1 and 'robot loss')."""
import pytest

from simrun import run_scenario

pytestmark = pytest.mark.sim


def test_dynamic_orders_delivered_by_decentralized_auction(tmp_path):
    r = run_scenario("configs/scenarios/m4_tasks.toml", tmp_path, **{"tasks.count": 8, "run.duration": 600.0})
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"] == 8
    assert t["rejected_reports"] == []                 # every pickup/delivery claim verified against ground truth
    assert r["safety"]["collisions"] == 0 and r["safety"]["static_contacts"] == 0
    done = [c["task"] for a in r["agents"].values() for c in a["goals"]["completed"]]
    assert sorted(done) == sorted(o["id"] for o in t["orders"])     # each order completed exactly once
    workers = [a for a in r["agents"].values() if a["goals"]["completed"]]
    assert len(workers) == 3                            # work actually distributed over all three robots
    # PR-05: orders released while robots were free are allocated in < 1 s
    first = [a["goals"]["alloc_latency"][0] for a in r["agents"].values()]
    assert max(first) < 1.0


def test_crashed_robot_task_is_reassigned_and_delivered(tmp_path):
    r = run_scenario("configs/scenarios/m4_robot_loss.toml", tmp_path)
    assert r["alive"] == [True, False, True]
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"] == 4
    assert r["safety"]["collisions"] == 0
    survivors = [r["agents"]["amr0"], r["agents"]["amr2"]]
    for a in survivors:
        re = a["goals"]["reassigned_from_lost"]
        assert re and re[0]["from"] == "amr1" and 12.0 < re[0]["t"] <= 12.0 + 3.0 + 0.2   # detected via heartbeat loss
    lost_task = survivors[0]["goals"]["reassigned_from_lost"][0]["task"]
    delivered_by = [n for n, a in r["agents"].items() if lost_task in [c["task"] for c in a["goals"]["completed"]]]
    assert delivered_by and delivered_by[0] != "amr1"


def test_centralized_reference_benchmark_runs_safely(tmp_path):
    """Benchmark-only App. D configuration 2: a central process owns the plan (synchronous PIBT steps)."""
    r = run_scenario("configs/scenarios/exp_overlap.toml", tmp_path, **{"fleet.planner": "central", "tasks.count": 6,
                                                                       "run.name": "central"})
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"] == 6 and t["rejected_reports"] == []
    assert r["safety"]["collisions"] == 0
    assert r["agent_exit_codes"]["central"] is not None


def test_low_battery_robot_charges_and_returns_to_work(tmp_path):
    r = run_scenario("configs/scenarios/m4_battery.toml", tmp_path)
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"]
    ch = r["agents"]["amr0"]["goals"]["charges"]
    assert ch, "amr0 must go charging when below the 25 % threshold"
    c = ch[0]
    assert c["battery_start"] < 0.25 and c["battery_end"] >= 0.9 and c["end"] > c["start"]
    assert r["safety"]["collisions"] == 0


def test_demo_mode_runs_in_real_time(tmp_path):
    """Pitch/demo mode: same simulation, paced at 1x wall clock (never used for data)."""
    r = run_scenario("configs/scenarios/m1_scripted.toml", tmp_path, **{"run.mode": "demo", "run.duration": 8.0,
                                                                         "run.stop_when_done": False, "run.name": "demo"})
    assert 0.8 < r["real_time_factor"] <= 1.05
