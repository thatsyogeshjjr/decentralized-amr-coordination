"""v3 FR-14 / PR-10 impairment semantics (preflight P7): burst loss, disconnection, FIFO vs reordering, effective
delay quantization, trace and aggregates, peer outage episodes."""
import csv

import numpy as np
import pytest

from fleet.comms.fault_injector import FaultInjector
from fleet.comms.peer_table import FRESH, LOST, PeerTable

TICK = 0.05


def msg(src, seq, ts, typ="RobotState"):
    return {"src": src, "seq": seq, "ts": ts, "type": typ}


def run_ticks(fi, sends, n_ticks):
    """sends: {tick: [(chan, msg)]}.  Emulates the barrier: a message sent at tick k is offered at tick k+1."""
    out = []
    for k in range(n_ticks):
        now = round(k * TICK, 4)
        for chan, m in sends.get(k - 1, []):
            fi.offer(chan, m, now)
        out += [(now, m) for m in fi.deliver(now)]
    return out


@pytest.mark.parametrize("lat_ms,eff_ms", [(0, 50), (10, 50), (50, 50), (100, 100)])
def test_effective_delay_is_tick_quantized(lat_ms, eff_ms):
    fi = FaultInjector(latency=lat_ms / 1000)
    got = run_ticks(fi, {3: [("pub", msg("amr1", 1, 3 * TICK))]}, 12)
    (t, m), = got
    assert round(t - m["ts"], 4) == eff_ms / 1000
    s = fi.summary()
    assert s["configured"]["latency_s"] == lat_ms / 1000 and s["eff_delay_median_s"] == eff_ms / 1000


def test_fifo_by_default_and_on_dm_even_with_reorder_flag():
    for reorder, chan in [(False, "pub"), (True, "dm")]:
        fi = FaultInjector(jitter=0.3, seed=4, reorder_pub=reorder)
        sends = {k: [(chan, msg("amr1", k + 1, k * TICK))] for k in range(40)}
        got = run_ticks(fi, sends, 80)
        seqs = [m["seq"] for _, m in got]
        assert seqs == sorted(seqs) and len(seqs) == 40
        assert fi.reorder_count == 0


def test_jitter_reorders_pub_when_enabled_and_peer_table_rejects_old():
    fi = FaultInjector(jitter=0.3, seed=4, reorder_pub=True)
    sends = {k: [("pub", {**msg("amr1", k + 1, k * TICK), "ttl": 5.0})] for k in range(40)}
    got = run_ticks(fi, sends, 80)
    seqs = [m["seq"] for _, m in got]
    assert seqs != sorted(seqs) and fi.reorder_count > 0
    pt = PeerTable(["amr1"])
    for now, m in got:
        pt.update(m, now)
    assert pt.rejected_old == fi.reorder_count          # every out-of-order arrival is rejected as old
    assert pt.entries["amr1"].state_seq == 40


def test_burst_loss_rate_and_run_length():
    p_enter, p_exit = 0.1 * 0.2 / 0.9, 0.2              # stationary P(bad) = 0.1, mean burst length 5
    fi = FaultInjector(burst={"p_enter": p_enter, "p_exit": p_exit}, seed=11)
    n = 60000
    drops = []
    for k in range(n):
        before = fi.dropped
        fi.offer("pub", msg("amr1", k, 0.0))
        drops.append(fi.dropped > before)
    d = np.array(drops)
    assert abs(d.mean() - 0.1) < 0.015
    runs, cur = [], 0
    for x in d:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    assert 4.3 < np.mean(runs) < 5.7
    assert fi.summary()["drops_by_reason"] == {"burst": int(d.sum())}


def test_burst_is_deterministic_and_does_not_perturb_main_rng():
    a = FaultInjector(loss=0.2, seed=3)
    b = FaultInjector(loss=0.2, seed=3, burst={"p_enter": 0.0, "p_exit": 1.0})   # burst chain never enters B
    for k in range(500):
        a.offer("pub", msg("amr1", k, 0.0))
        b.offer("pub", msg("amr1", k, 0.0))
    assert [m["seq"] for m in a.deliver(1)] == [m["seq"] for m in b.deliver(1)]


def test_disconnect_window_drops_both_directions_only_in_window():
    sched = [{"robot": 1, "t": 1.0, "duration": 0.5}]
    rx0 = FaultInjector(disconnect=sched, name="amr0")    # receiver amr0: drops traffic *from* amr1
    rx1 = FaultInjector(disconnect=sched, name="amr1")    # receiver amr1: drops everything it would receive
    for k in range(40):
        ts = k * TICK
        rx0.offer("pub", msg("amr1", k, ts))
        rx0.offer("pub", msg("amr2", k, ts))
        rx1.offer("dm", msg("amr0", k, ts, "Claim"))
    in_window = sum(1 for k in range(40) if 1.0 <= k * TICK < 1.5)
    assert rx0.drops_by_reason["disconnect"] == in_window
    assert rx1.drops_by_reason["disconnect"] == in_window
    assert all(m["src"] != "amr1" or not 1.0 <= m["ts"] < 1.5 for m in rx0.deliver(10))


def test_trace_matches_aggregates(tmp_path):
    p = tmp_path / "trace.csv"
    fi = FaultInjector(latency=0.05, loss=0.3, seed=2, name="amr0", trace_path=str(p))
    sends = {k: [("pub", msg("amr1", k + 1, k * TICK)), ("dm", msg("amr2", k + 1, k * TICK, "Claim"))] for k in range(50)}
    run_ticks(fi, sends, 60)
    fi.close()
    rows = list(csv.DictReader(open(p)))
    out = [r["outcome"] for r in rows]
    s = fi.summary()
    assert out.count("drop_random") == s["drops_by_reason"]["random"] == fi.dropped
    assert out.count("delivered") == s["delivered"]
    assert out.count("queued") == s["offered"] - s["dropped"]
    d = [r for r in rows if r["outcome"] == "delivered"]
    assert all(r["dst"] == "amr0" and float(r["eff_delay"]) == pytest.approx(0.05) for r in d)


def test_peer_outage_episode_recorded():
    pt = PeerTable(["amr1"], stale_after=1.0, lost_after=3.0)
    st = lambda seq, ts: {"type": "RobotState", "src": "amr1", "seq": seq, "ts": ts, "ttl": 2.0}
    pt.update(st(1, 0.0), 0.0)
    pt.refresh(0.5)
    pt.refresh(1.6)                     # stale
    pt.refresh(3.5)                     # lost
    assert pt.entries["amr1"].status == LOST and pt.outage_summary()["open"][0]["worst"] == LOST
    pt.update(st(2, 5.0), 5.0)
    pt.refresh(5.0)
    s = pt.outage_summary()
    assert pt.entries["amr1"].status == FRESH and s["n_closed"] == 1 and not s["open"]
    assert s["closed"][0]["duration"] == pytest.approx(5.0) and s["closed"][0]["worst"] == LOST
