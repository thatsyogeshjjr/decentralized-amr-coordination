"""Campaign analysis (design doc §16.3): per-run metrics, medians, dispersion and bootstrap confidence intervals.

Metrics per run
  makespan  T_make = time of the last verified delivery (eq. 9), only defined when every order was delivered
  throughput Q = completed / T (eq. 10), T = sim time at end of run
  collisions N_c (eq. 8 onsets + MuJoCo chassis contacts), min chassis-centre distance
  B_net = sum over robots of transmitted bytes / T (eq. 11), also per robot
  local decision latency (claim round-trip), allocation latency, planning compute p99, deadlock escalations/recovery

    python -m fleet.experiments.analysis results/campaigns/<name> [--baseline stop_wait]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

B_RESAMPLES = 5000


def bootstrap_ci(x, stat=np.median, alpha: float = 0.05, seed: int = 0) -> tuple[float, float] | None:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(B_RESAMPLES)])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def episode_records(agents: dict) -> tuple[list[dict], list[dict], bool]:
    """Deadlock and congestion episodes of one run (v3 §10.8).  Episodes still open in an agent's final stats are
    classified failed_run_end (deadlock) / open_run_end (congestion).  `truncated` flags agents whose episode list
    was capped (counts then come from n_episodes)."""
    eps, cg, trunc = [], [], False
    for a in agents.values():
        d = a.get("deadlock", {})
        if "n_episodes" not in d:
            continue
        raw = d["episodes"] + ([{**d["open"], "outcome": "failed_run_end"}] if d.get("open") else [])
        trunc |= d["n_episodes"] > len(raw) or d["n_congestion"] > len(d["congestion"]) + (1 if d.get("congestion_open") else 0)
        for e in raw:
            rec = round(e["t_recovered"] - e["t_detected"], 3) if e.get("t_recovered") is not None else None
            eps.append({"outcome": e["outcome"], "recovery": rec, "rerouted": e.get("t_rerouting") is not None,
                        "escalations": e.get("n_escalations")})
        for c in d["congestion"] + ([{**d["congestion_open"], "outcome": "open_run_end"}] if d.get("congestion_open") else []):
            cg.append({"outcome": c["outcome"], "duration": c.get("duration")})
    return eps, cg, trunc


def episode_stats(eps: list[dict], cg: list[dict]) -> dict:
    """Recovery-tail statistics.  Recovery time = detection -> recovered (legacy resolution metric); the % within 5 s
    uses *all* episodes as the denominator (timed-out / failed ones count as not recovered)."""
    rec = [e["recovery"] for e in eps if e["recovery"] is not None]
    out = {"n_deadlock": len(eps), "n_recovered": len(rec),
           "outcomes": {k: sum(e["outcome"] == k for e in eps) for k in
                        ("recovered", "timed_out", "failed_goal_changed", "failed_run_end")},
           "rerouted": sum(e["rerouted"] for e in eps),
           "recovery_median_s": float(np.median(rec)) if rec else None,
           "recovery_p90_s": float(np.percentile(rec, 90)) if rec else None,
           "recovery_max_s": float(np.max(rec)) if rec else None,
           "within_5s_frac": (sum(x <= 5.0 for x in rec) / len(eps)) if eps else None,
           "n_congestion": len(cg),
           "congestion_median_s": float(np.median([c["duration"] for c in cg if c["duration"] is not None]))
           if any(c["duration"] is not None for c in cg) else None,
           "congestion_became_deadlock": sum(c["outcome"] == "became_deadlock" for c in cg)}
    return out


def run_metrics(r: dict) -> dict:
    T = float(r["sim_time"])
    t = r["tasks"]
    agents = r.get("agents", {})
    tx = sum(a["net"]["tx_bytes"] for a in agents.values())
    msgs = sum(sum(a["net"]["tx_msgs"].values()) for a in agents.values())
    lat = [a["claim_latency_mean"] for a in agents.values() if a.get("claim_latency_mean") is not None]
    alloc = [x for a in agents.values() for x in a.get("goals", {}).get("alloc_latency", [])]
    rec = [x for a in agents.values() for x in a.get("deadlock", {}).get("recoveries", [])]
    comp = [a["compute_p99_ms"] for a in agents.values() if a.get("compute_p99_ms") is not None]
    adv = [a["advisor"]["infer_p99_ms"] for a in agents.values() if a.get("advisor")]
    n = r["n_robots"]
    eps, cg, trunc = episode_records(agents)
    imp = [a["net"]["impair"] for a in agents.values() if a.get("net", {}).get("impair")]
    drops: dict = {}
    for x in imp:
        for k, v in x["drops_by_reason"].items():
            drops[k] = drops.get(k, 0) + v
    iso = [e for a in agents.values() for e in (a.get("isolation") or {}).get("episodes", [])]
    iso_open = [a["isolation"]["open"] for a in agents.values() if (a.get("isolation") or {}).get("open")]
    outs = [o["duration"] for a in agents.values()
            for o in (a.get("peer_table", {}).get("outages") or {}).get("closed", [])]
    return {
        "seed": r["seed"], "planner": r["planner"], "learned": r.get("learned_priority", False), "n_robots": n,
        "completed": t["n_delivered"], "n_tasks": t["n_tasks"], "all_delivered": t["n_delivered"] == t["n_tasks"],
        "makespan": t["makespan"], "sim_time": T, "throughput": t["n_delivered"] / T if T > 0 else 0.0,
        "collisions": r["safety"]["collisions"], "eq8_events": r["safety"]["eq8_events"],
        "robot_contacts": r["safety"]["robot_contacts"], "static_contacts": r["safety"]["static_contacts"],
        "min_pair_dist": r["safety"]["min_pair_dist"],
        "bnet_Bps": tx / T if T > 0 else 0.0, "bnet_per_robot_Bps": tx / T / n if T > 0 else 0.0,
        "msgs_per_s": msgs / T if T > 0 else 0.0,
        "claim_latency_mean": float(np.mean(lat)) if lat else None,
        "alloc_latency_median": float(np.median(alloc)) if alloc else None,
        "deadlock_escalations": sum(a.get("deadlock", {}).get("escalations", 0) for a in agents.values()),
        "deadlock_recovery_max": max(rec) if rec else None,
        "compute_p99_ms": max(comp) if comp else None, "advisor_p99_ms": max(adv) if adv else None,
        "rx_drop_frac": (sum(a["net"]["rx_dropped"] for a in agents.values()) /
                         max(1, sum(a["net"]["rx_offered"] for a in agents.values()))),
        "aborted": r.get("aborted", False), "rtf": r.get("real_time_factor"),
        # v3 instrumentation (absent in v2 results -> empty / None)
        "dl_episodes": eps, "congestion": cg, "episodes_truncated": trunc, "v3": bool(imp),
        "drops_by_reason": drops,
        "eff_delay_median_s": float(np.median([x["eff_delay_median_s"] for x in imp if x["eff_delay_median_s"] is not None])) if imp else None,
        "eff_delay_max_s": max((x["eff_delay_max_s"] for x in imp if x["eff_delay_max_s"] is not None), default=None),
        "reorder_count": sum(x["reorder_count"] for x in imp),
        "rejected_stale": sum(a.get("peer_table", {}).get("rejected_stale", 0) for a in agents.values()),
        "isolation_episodes": iso, "isolation_open_at_end": len(iso_open),
        "peer_outage_s": outs,
        "rejected_reports": len(t.get("rejected_reports", [])),
        "optb": [a["optb"] for a in agents.values() if a.get("optb")],
        "rejected_deliveries": sum(x.get("phase") == "delivered" for x in t.get("rejected_reports", [])),
    }


def load_campaign(d: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted(d.glob("*/seed_*/result.json")):
        label = p.parent.parent.name
        out.setdefault(label, []).append(run_metrics(json.loads(p.read_text())))
    for v in out.values():
        v.sort(key=lambda m: m["seed"])
    return out


def summarize(runs: list[dict]) -> dict:
    ms = [m["makespan"] for m in runs if m["makespan"] is not None]
    def med(key):
        v = [m[key] for m in runs if m[key] is not None]
        return float(np.median(v)) if v else None
    q = np.percentile(ms, [25, 75]) if ms else [None, None]
    return {
        "n_runs": len(runs), "n_complete": sum(m["all_delivered"] for m in runs),
        "makespan_median": float(np.median(ms)) if ms else None, "makespan_iqr": [float(q[0]), float(q[1])] if ms else None,
        "makespan_ci95": bootstrap_ci(ms), "makespan_mean": float(np.mean(ms)) if ms else None,
        "throughput_median": med("throughput"), "collisions_total": sum(m["collisions"] for m in runs),
        "static_contacts_total": sum(m["static_contacts"] for m in runs),
        "min_pair_dist": min((m["min_pair_dist"] for m in runs if m["min_pair_dist"] is not None), default=None),
        "bnet_per_robot_Bps_median": med("bnet_per_robot_Bps"), "msgs_per_s_median": med("msgs_per_s"),
        "claim_latency_median": med("claim_latency_mean"), "alloc_latency_median": med("alloc_latency_median"),
        "compute_p99_ms_max": max((m["compute_p99_ms"] for m in runs if m["compute_p99_ms"] is not None), default=None),
        "advisor_p99_ms_max": max((m["advisor_p99_ms"] for m in runs if m["advisor_p99_ms"] is not None), default=None),
        "deadlock_escalations_total": sum(m["deadlock_escalations"] for m in runs),
        "deadlock_recovery_max": max((m["deadlock_recovery_max"] for m in runs if m["deadlock_recovery_max"] is not None), default=None),
        "rx_drop_frac_median": med("rx_drop_frac"),
        "incomplete_runs": [m["seed"] for m in runs if not m["all_delivered"]],
        "rejected_deliveries_total": sum(m.get("rejected_deliveries", 0) for m in runs),
        **(v3_summary(runs) if any(m.get("v3") for m in runs) else {}),
    }


def v3_summary(runs: list[dict]) -> dict:
    """Episode tails, impairment and isolation aggregates over a condition (v3 runs only)."""
    eps = [e for m in runs for e in m["dl_episodes"]]
    cg = [c for m in runs for c in m["congestion"]]
    drops: dict = {}
    for m in runs:
        for k, v in m["drops_by_reason"].items():
            drops[k] = drops.get(k, 0) + v
    iso = [e for m in runs for e in m["isolation_episodes"]]
    outs = [x for m in runs for x in m["peer_outage_s"]]
    eff = [m["eff_delay_median_s"] for m in runs if m["eff_delay_median_s"] is not None]
    return {
        "episodes": episode_stats(eps, cg), "episodes_truncated_runs": sum(m["episodes_truncated"] for m in runs),
        "drops_by_reason_total": drops, "eff_delay_median_s": float(np.median(eff)) if eff else None,
        "eff_delay_max_s": max((m["eff_delay_max_s"] for m in runs if m["eff_delay_max_s"] is not None), default=None),
        "reorder_total": sum(m["reorder_count"] for m in runs),
        "rejected_stale_total": sum(m["rejected_stale"] for m in runs),
        "isolation_episodes": len(iso), "isolation_partitioned": sum(e["t_partitioned"] is not None for e in iso),
        "isolation_open_at_end": sum(m["isolation_open_at_end"] for m in runs),
        "isolation_outage_median_s": float(np.median([e["outage_s"] for e in iso])) if iso else None,
        "peer_outages": len(outs), "peer_outage_median_s": float(np.median(outs)) if outs else None,
        "peer_outage_max_s": max(outs) if outs else None,
        "rejected_reports_total": sum(m["rejected_reports"] for m in runs),
        **optb_summary(runs),
    }


def optb_summary(runs: list[dict]) -> dict:
    """Option B advisor statistics over a condition: fallback frequency, adjustment activity, inference latency."""
    st = [o for m in runs for o in m.get("optb", [])]
    if not st:
        return {}
    calls = sum(o["calls"] for o in st)
    fb: dict = {}
    for o in st:
        for k, v in o["fallbacks"].items():
            fb[k] = fb.get(k, 0) + v
    p99 = [o["infer_p99_ms"] for o in st if o.get("infer_p99_ms") is not None]
    nz = [o["delta_nonzero_frac"] for o in st if o.get("delta_nonzero_frac") is not None]
    return {"optb_calls": calls, "optb_fallbacks": fb, "optb_fallback_frac": sum(fb.values()) / max(1, calls),
            "optb_nonzero_frac": float(np.mean(nz)) if nz else None, "optb_infer_p99_ms": max(p99) if p99 else None}


def paired_improvement(base: list[dict], other: list[dict]) -> dict | None:
    """(T_base - T_method) / T_base * 100 per seed, paired on seed; unfinished runs use their end time as a
    lower bound of their makespan (conservative for the method, generous for the baseline is avoided by
    excluding seeds where the *baseline* did not finish)."""
    b = {m["seed"]: m for m in base}
    vals, excluded = [], []
    for m in other:
        bm = b.get(m["seed"])
        if bm is None or bm["makespan"] is None:
            excluded.append(m["seed"])          # pre-registered rule: reference run missing or unfinished
            continue
        tm = m["makespan"] if m["makespan"] is not None else m["sim_time"]
        vals.append(100.0 * (bm["makespan"] - tm) / bm["makespan"])
    if not vals:
        return None
    return {"n_pairs": len(vals), "excluded_seeds": excluded,
            "method_unfinished_seeds": [m["seed"] for m in other if m["makespan"] is None], "median_pct": float(np.median(vals)), "mean_pct": float(np.mean(vals)),
            "ci95_median": bootstrap_ci(vals), "ci95_mean": bootstrap_ci(vals, np.mean),
            "per_seed": [round(v, 2) for v in vals], "wins": int(sum(v > 0 for v in vals))}


def plot(d: Path, data: dict[str, list[dict]], order: list[str]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    labels = [l for l in order if l in data]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4))
    vals = [[m["makespan"] for m in data[l] if m["makespan"] is not None] for l in labels]
    ax.boxplot(vals, tick_labels=labels)
    for i, v in enumerate(vals, start=1):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), v, s=12, alpha=0.6)
    ax.set_ylabel("makespan (s)")
    ax.set_title(f"{d.name}: makespan per condition (each dot = one seed)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f = d / "makespan.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)
    return files


def analyze(d: str | Path, baseline: str | None = "stop_wait", order: list[str] | None = None) -> dict:
    d = Path(d)
    data = load_campaign(d)
    order = order or sorted(data)
    summ = {l: summarize(data[l]) for l in order if l in data}
    comp = {}
    grouped = any("__" in l for l in data)
    if grouped:
        # labels "<group>__<method>": each method is compared with "<group>__<baseline>" on the same seeds
        for l in data:
            g, _, meth = l.partition("__")
            b = f"{g}__{baseline}"
            if meth != baseline and b in data:
                comp[l] = paired_improvement(data[b], data[l])
    elif baseline and baseline in data:
        for l in data:
            if l != baseline:
                comp[l] = paired_improvement(data[baseline], data[l])
    figs = plot(d, data, order) if data else []
    orch = None
    if (d / "runs.json").exists():
        rs = json.loads((d / "runs.json").read_text())
        orch = {"attempted": len(rs), "failed": [{k: r.get(k) for k in ("label", "seed", "status")} for r in rs
                                                 if r["status"] not in ("ok", "cached")],
                "retried": [{k: r.get(k) for k in ("label", "seed", "status")} for r in rs if r.get("retried")],
                "manifests": len(list(d.glob("*/seed_*/manifest.json")))}
    out = {"campaign": d.name, "conditions": summ, "improvement_vs_" + str(baseline): comp, "figures": figs,
           "orchestrator": orch,
           "runs": {l: data[l] for l in data}}
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_dir")
    ap.add_argument("--baseline", default="stop_wait")
    a = ap.parse_args(argv)
    out = analyze(a.campaign_dir, a.baseline)
    print(json.dumps({"conditions": out["conditions"], "improvement": out["improvement_vs_" + a.baseline]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
