"""Statistics used in the V&V report (medians, bootstrap CIs, paired improvement vs the stop-and-wait baseline)."""
import numpy as np

from fleet.experiments.analysis import bootstrap_ci, paired_improvement, summarize


def run(seed, makespan, collisions=0, delivered=10, sim_time=None):
    return {"seed": seed, "makespan": makespan, "sim_time": sim_time or (makespan or 500.0), "all_delivered": makespan is not None,
            "collisions": collisions, "static_contacts": 0, "min_pair_dist": 1.2, "throughput": delivered / (makespan or 500),
            "bnet_per_robot_Bps": 1000.0, "msgs_per_s": 50.0, "claim_latency_mean": 0.1, "alloc_latency_median": 0.3,
            "compute_p99_ms": 0.3, "advisor_p99_ms": None, "deadlock_escalations": 1, "deadlock_recovery_max": 2.0,
            "rx_drop_frac": 0.0}


def test_bootstrap_ci_contains_true_median_and_is_deterministic():
    x = np.random.default_rng(0).normal(100, 10, 50)
    lo, hi = bootstrap_ci(x)
    assert lo < np.median(x) < hi and hi - lo < 15
    assert bootstrap_ci(x) == (lo, hi)
    assert bootstrap_ci([1.0]) is None


def test_paired_improvement_matches_hand_computation():
    base = [run(s, 200.0) for s in range(1, 11)]
    method = [run(s, 150.0) for s in range(1, 11)]
    imp = paired_improvement(base, method)
    assert imp["n_pairs"] == 10 and imp["median_pct"] == 25.0 and imp["wins"] == 10
    # a baseline seed that never finished is excluded rather than counted as a win
    base[0]["makespan"] = None
    assert paired_improvement(base, method)["n_pairs"] == 9
    # an unfinished method run is charged its end time (conservative)
    method[1]["makespan"], method[1]["sim_time"] = None, 400.0
    assert min(paired_improvement(base, method)["per_seed"]) == -100.0


def test_summary_counts_failures_and_collisions():
    runs = [run(1, 100.0), run(2, 120.0, collisions=1), run(3, None)]
    s = summarize(runs)
    assert s["n_runs"] == 3 and s["n_complete"] == 2 and s["collisions_total"] == 1
    assert s["makespan_median"] == 110.0
