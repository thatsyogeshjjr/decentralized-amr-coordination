"""Option B advisor inside the robot agents, visual MuJoCo (accelerated): bounded priority influence, deterministic
fallback accounting, local inference profile (design doc v3 §10.9, PR-09).  Performance claims come only from the
advisor_ablation campaign, not from these tests."""
import pytest

from fleet.learning.congestion import ADJ_MAX
from simrun import ROOT, run_scenario

pytestmark = pytest.mark.sim


def check_common(r, mode):
    assert r["safety"]["collisions"] == 0 and r["safety"]["static_contacts"] == 0
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"] and t["rejected_reports"] == []
    for a in r["agents"].values():
        o = a["optb"]
        assert o["mode"] == mode and o["calls"] > 0
        assert set(o["fallbacks"]) <= {"isolated", "stale_input", "exception", "invalid_output"}
        assert 0.0 <= o["delta_mean"] <= ADJ_MAX
    return [a["optb"] for a in r["agents"].values()]


def test_heuristic_advisor_runs_safely_with_bounded_adjustment(tmp_path):
    r = run_scenario("configs/scenarios/m4_tasks.toml", tmp_path, **{"tasks.count": 6, "run.duration": 600.0,
                                                                     "fleet.priority_advisor": "heuristic"})
    st = check_common(r, "heuristic")
    assert any(o["delta_nonzero_frac"] > 0 for o in st)          # the heuristic actually influenced priorities


def test_learned_advisor_runs_safely_and_reports_profile(tmp_path):
    model = ROOT / "models" / "optb_advisor.pt"
    assert model.exists(), "train the Option B model first (python -m fleet.learning.train_optb train)"
    r = run_scenario("configs/scenarios/m4_tasks.toml", tmp_path, **{"tasks.count": 6, "run.duration": 600.0,
                                                                     "fleet.priority_advisor": "learned"})
    st = check_common(r, "learned")
    for o in st:
        assert o["n_params"] > 0 and o["file_bytes"] > 0 and o["model_version"].startswith("optb-v1")
        assert o["infer_p99_ms"] is not None and o["infer_p99_ms"] < 50.0     # development host, wall clock
