"""M5 gate (visual MuJoCo, accelerated): blocked-aisle handling and deadlock detection/recovery
(FR-07, FR-08, FR-09, PR-06; acceptance items 3 and 4)."""
import pytest

from simrun import run_scenario

pytestmark = pytest.mark.sim


def test_blocked_aisle_detected_shared_and_rerouted(tmp_path):
    r = run_scenario("configs/scenarios/m5_blockage.toml", tmp_path)
    placed = [e for e in r["events"] if e["event"] == "block_placed"]
    assert placed, "fault injector placed the pallet"
    node = placed[0]["node"]
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"] == 3
    s = r["safety"]
    assert s["collisions"] == 0 and s["static_contacts"] == 0        # nobody touched the pallet
    sensed = [a for a in r["agents"].values() if any(d["node"] == node for d in a["blockages_detected"])]
    assert sensed, "at least one robot detected the pallet with its own range ring"
    assert min(d["t"] for a in sensed for d in a["blockages_detected"]) - placed[0]["t"] < 2.0
    # every live robot learned about the blockage (own sensing or a peer's BlockageEvent)
    assert all(str(node) in a["blocked_known"] for a in r["agents"].values())


def test_blocked_pickup_releases_task_for_reassignment(tmp_path):
    r = run_scenario("configs/scenarios/m5_block_release.toml", tmp_path)
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"] == 2
    released = [x for a in r["agents"].values() for x in a["goals"]["released"]]
    assert released and released[0]["reason"] == "goal blocked" and released[0]["task"] == "T000"
    t0 = next(o for o in t["orders"] if o["id"] == "T000")
    assert t0["picked_t"] > 41.0          # only picked after the station was clear again
    assert r["safety"]["collisions"] == 0 and r["safety"]["static_contacts"] == 0


def test_known_deadlock_detected_and_recovered(tmp_path):
    on = run_scenario("configs/scenarios/m5_deadlock.toml", tmp_path, **{"run.name": "on"})
    off = run_scenario("configs/scenarios/m5_deadlock.toml", tmp_path, **{"run.name": "off",
                                                                          "planning.deadlock_window": 1e9})
    assert not any(off["done_flags"]), "without deadlock handling the swap must stay deadlocked (case is real)"
    assert all(on["done_flags"])
    dl = [a["deadlock"] for a in on["agents"].values()]
    assert sum(d["escalations"] for d in dl) >= 1
    rec = [x for d in dl for x in d["recoveries"]]
    assert rec and max(rec) < 5.0                    # PR-06: resolved < 5 s after detection
    assert on["safety"]["collisions"] == 0 and off["safety"]["collisions"] == 0
