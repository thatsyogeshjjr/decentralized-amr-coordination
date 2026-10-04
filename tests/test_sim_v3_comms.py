"""v3 FR-14 gates (visual MuJoCo, accelerated): temporary peer disconnection, persistent partition and carrier
reconciliation (preflight P4 + addendum A).  Safety alone is not accepted as evidence: each test also checks the
isolation episode, recovery and task progress."""
import pytest

from simrun import run_scenario

pytestmark = pytest.mark.sim


def delivered_once(r):
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"]
    done = [c["task"] for a in r["agents"].values() for c in a["goals"]["completed"]]
    assert sorted(done) == sorted(o["id"] for o in t["orders"])          # each order completed exactly once
    assert not [x for x in t["rejected_reports"] if x["phase"] == "delivered"]   # never a false delivery


def test_short_disconnect_isolates_reconnects_and_completes(tmp_path):
    r = run_scenario("configs/scenarios/v3_disconnect.toml", tmp_path)
    assert r["safety"]["collisions"] == 0 and r["safety"]["static_contacts"] == 0
    delivered_once(r)
    assert r["tasks"]["rejected_reports"] == []
    a1 = r["agents"]["amr1"]
    ep, = a1["isolation"]["episodes"]
    assert 40.0 < ep["t_isolated"] <= 41.2                  # detected via heartbeat (1 s stale) after link loss
    assert ep["t_partitioned"] is None and 50.0 <= ep["t_reconnect"] <= 50.2
    assert ep["t_resume"] == pytest.approx(ep["t_reconnect"] + 1.0, abs=0.06)
    assert a1["isolation"]["state"] == "connected"
    assert a1["net"]["impair"]["drops_by_reason"]["disconnect"] > 0
    for peer in ("amr0", "amr2"):
        a = r["agents"][peer]
        assert a["isolation"]["episodes"] == []             # connected peers are not isolated
        o = [x for x in a["peer_table"]["outages"]["closed"] if x["peer"] == "amr1"]
        assert o and o[0]["recovered"] >= 50.0 and o[0]["worst"] == "lost"
    assert {s["from"] for s in a1["goals"]["suspended"]} == {"amr0", "amr2"}   # isolated robot took nothing over


def test_long_disconnect_reports_partition_then_recovers(tmp_path):
    r = run_scenario("configs/scenarios/v3_disconnect.toml", tmp_path,
                     **{"comms.disconnect": [{"robot": 1, "t": 40.0, "duration": 45.0}], "run.name": "v3_disc45"})
    assert r["safety"]["collisions"] == 0
    delivered_once(r)
    ep, = r["agents"]["amr1"]["isolation"]["episodes"]
    assert ep["t_partitioned"] == pytest.approx(ep["t_isolated"] + 30.0, abs=0.06)
    assert 85.0 <= ep["t_reconnect"] <= 85.2 and ep["outage_s"] > 44.0


def test_carrier_disconnect_payload_sensor_and_reassertion(tmp_path):
    r = run_scenario("configs/scenarios/v3_carrier_disconnect.toml", tmp_path)
    assert r["safety"]["collisions"] == 0
    delivered_once(r)
    ag = r["agents"]
    assert ag["amr0"]["goals"]["completed"][0]["task"] == "T000"         # the carrier delivered it
    assert ag["amr0"]["goals"]["reasserted"]                              # ... after re-asserting ownership
    failed = [p for n in ("amr1", "amr2") for p in ag[n]["goals"]["pickup_failed"]]
    rejected = r["tasks"]["rejected_reports"]
    # every rejected report is a pickup of the carried order, and the robot that made it released the order
    assert rejected and all(x["phase"] == "picked" and x["task_id"] == "T000" for x in rejected)
    assert len(failed) == len(rejected)
