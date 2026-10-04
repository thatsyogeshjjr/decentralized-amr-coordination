"""Generate docs/vv_report.md from real artifacts only: campaign summaries (results/campaigns/*/summary.json), the
pytest JUnit reports (results/pytest.xml = v2 evidence, results/pytest_v3.xml = current suite), the regression record
(results/regression/*/regression.json) and the Option B dataset / model metadata.  Nothing in the report is typed by
hand except docs/vv_findings.md, which is included verbatim.

Sections 1-7 are the v2 evidence (design doc v2 campaigns, unchanged numbers).  Section 8 is the v3 evidence
(design doc v3: FR-13/14, PR-09/10, deadlock episode tails, Option B ablation).

    python -m fleet.experiments.report
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from fleet.common.config import PROJECT_ROOT

CAMP = PROJECT_ROOT / "results" / "campaigns"
DOC = PROJECT_ROOT / "docs" / "vv_report.md"
V2_CAMPAIGNS = ["topology", "comms", "blockage", "density", "acceptance", "scale"]     # the 430 historical runs
V3_CAMPAIGNS = ["comms_v3", "optb_data", "advisor_ablation"]

REQ_TESTS = {
    "FR-01 >=3 AMRs": ["test_three_robots_drive_routes", "test_dynamic_orders_delivered_by_decentralized_auction"],
    "FR-02 state/intent publication": ["test_robot_state_carries_fr02_fields", "test_pub_sub_and_dealer_router_delivery"],
    "FR-03 no central controller": ["test_fleet_continues_after_dashboard_is_killed", "test_crashed_robot_task_is_reassigned_and_delivered"],
    "FR-04 P2P auction/consensus": ["test_simultaneous_tasks_get_unique_winners_quickly", "test_consensus_converges_under_message_loss",
                                    "test_dynamic_orders_delivered_by_decentralized_auction"],
    "FR-05 executable route (A*)": ["test_astar_shortest_and_turn_aware"],
    "FR-06 conflict detection/resolution": ["test_crossing_conflicts_resolved_by_pibt", "test_four_way_intersection",
                                            "test_corridor_swap_with_pocket_recovered_by_escalation"],
    "FR-07 deadlock detection/recovery": ["test_known_deadlock_detected_and_recovered"],
    "FR-08 blockage -> local re-plan": ["test_blocked_aisle_detected_shared_and_rerouted", "test_pallet_in_aisle_detected_after_confirmation"],
    "FR-09 infeasible task released": ["test_blocked_pickup_releases_task_for_reassignment", "test_crashed_robot_task_is_reassigned_and_delivered",
                                       "test_lost_peer_task_is_reassigned"],
    "FR-10 deterministic safety layer": ["test_forced_convergence_stopped_by_safety_shield", "test_shield_only_reduces"],
    "FR-11 dashboard": ["test_dashboard_serves_state_from_robot_telemetry"],
    "FR-12 operation without dashboard": ["test_fleet_continues_after_dashboard_is_killed"],
    "PR-03 state rate 5-10 Hz": ["test_state_rate_across_processes"],
    "PR-04 local decision < 200 ms": ["test_crossing_conflicts_resolved_by_pibt"],
    "PR-05 allocation < 1 s": ["test_simultaneous_tasks_get_unique_winners_quickly", "test_dynamic_orders_delivered_by_decentralized_auction"],
    "PR-06 deadlock recovery < 5 s": ["test_known_deadlock_detected_and_recovered"],
    "SC-08 compute budget": ["test_crossing_conflicts_resolved_by_pibt", "test_inference_latency_within_edge_budget"],
    "Reproducibility": ["test_runs_are_reproducible", "test_cem_is_deterministic_and_never_worse_than_its_start"],
}
REQ_TESTS_V3 = {
    "FR-13 non-controlling orchestrator (gate 7)": ["test_orchestrator_modules_have_no_control_dependencies",
                                                     "test_simlink_rejects_foreign_types_and_strips_decision_fields",
                                                     "test_runtime_boundary_and_injection_has_no_effect"],
    "FR-14 impairment: burst / disconnect / jitter / trace": ["test_burst_loss_rate_and_run_length",
                                                              "test_burst_is_deterministic_and_does_not_perturb_main_rng",
                                                              "test_disconnect_window_drops_both_directions_only_in_window",
                                                              "test_trace_matches_aggregates", "test_peer_outage_episode_recorded"],
    "FR-14 isolation / reconnection (P4)": ["test_full_cycle_isolated_reconnecting_connected",
                                            "test_timeout_reports_partition_and_relapse_during_reconnect",
                                            "test_short_disconnect_isolates_reconnects_and_completes",
                                            "test_long_disconnect_reports_partition_then_recovers",
                                            "test_carrier_disconnect_payload_sensor_and_reassertion"],
    "PR-10 delay semantics (tick quantization, FIFO, reordering)": ["test_effective_delay_is_tick_quantized",
                                                                    "test_fifo_by_default_and_on_dm_even_with_reorder_flag",
                                                                    "test_jitter_reorders_pub_when_enabled_and_peer_table_rejects_old"],
    "v3 §10.8 deadlock / congestion episodes": ["test_deadlock_episode_states_and_recovery", "test_repeated_episodes_get_distinct_ids",
                                                "test_timeout_boundary", "test_goal_change_fails_open_episode_and_run_end_reports_open",
                                                "test_congestion_vs_deadlock_classification"],
    "Option B schema / label / fallback (gate 8)": ["test_schema_is_frozen_16_features_in_fixed_order",
                                                    "test_features_from_local_view_and_conservative_missing_values",
                                                    "test_blocking_label_definition", "test_labels_use_future_but_features_do_not",
                                                    "test_adjustment_bounds_and_ordering_effect",
                                                    "test_advisor_fallbacks_are_deterministic_and_counted",
                                                    "test_split_matrix_and_disjoint_seeds", "test_model_roundtrip_and_schema_guard"],
    "Option B in visual MuJoCo": ["test_heuristic_advisor_runs_safely_with_bounded_adjustment",
                                  "test_learned_advisor_runs_safely_and_reports_profile"],
}


def junit(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        name = tc.get("name").split("[")[0]
        bad = tc.find("failure") is not None or tc.find("error") is not None
        skip = tc.find("skipped") is not None
        prev = out.get(name, "pass")
        out[name] = "FAIL" if bad or prev == "FAIL" else ("skipped" if skip else prev)
    return out


def f(x, nd=1):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def load(name: str) -> dict | None:
    p = CAMP / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def cond_table(s: dict, labels: list[str]) -> list[str]:
    rows = ["| condition | runs | all orders delivered | makespan median [IQR] (s) | 95% CI of median | collisions | min gap (m) | throughput (orders/s) |",
            "|---|---|---|---|---|---|---|---|"]
    for l in labels:
        c = s["conditions"].get(l)
        if not c:
            continue
        iqr = c["makespan_iqr"]
        ci = c["makespan_ci95"]
        rows.append(f"| {l} | {c['n_runs']} | {c['n_complete']}/{c['n_runs']} | {f(c['makespan_median'])} "
                    f"[{f(iqr[0]) if iqr else '–'}, {f(iqr[1]) if iqr else '–'}] | "
                    f"{'[' + f(ci[0]) + ', ' + f(ci[1]) + ']' if ci else '–'} | {c['collisions_total']} | "
                    f"{f(c['min_pair_dist'], 2)} | {f(c['throughput_median'], 4)} |")
    return rows


def imp_table(s: dict, key: str) -> list[str]:
    comp = s.get(key, {})
    rows = ["| method | paired seeds | median improvement vs baseline (%) | 95% CI (median) | mean (%) | seeds better |",
            "|---|---|---|---|---|---|"]
    for l, v in comp.items():
        if v is None:
            continue
        ci = v["ci95_median"]
        rows.append(f"| {l} | {v['n_pairs']} | {f(v['median_pct'])} | {'[' + f(ci[0]) + ', ' + f(ci[1]) + ']' if ci else '–'} | "
                    f"{f(v['mean_pct'])} | {v['wins']}/{v['n_pairs']} |")
    return rows


def pct_row(label: str, v: dict | None) -> str:
    if not v:
        return f"| {label} | – | – | – | – |"
    ci = v["ci95_median"]
    worse = sum(x < 0 for x in v["per_seed"])
    excl = f" (excluded seeds {v['excluded_seeds']})" if v.get("excluded_seeds") else ""
    unf = v.get("method_unfinished_seeds") or []
    note = f" (bound: {len(unf)} unfinished runs counted at their end time)" if unf else ""
    return (f"| {label} | {v['n_pairs']}{excl} | {f(v['median_pct'], 1)}{note} | "
            f"{'[' + f(ci[0], 1) + ', ' + f(ci[1], 1) + ']' if ci else '–'} | {v['wins']} / {worse} |")


def v3_sections(tests_v3: dict) -> list[str]:
    from fleet.experiments.analysis import load_campaign, paired_improvement
    L = ["## 8. v3 evidence (design doc v3, simulation-only scope)", "",
         "Everything in this section was produced with the v3 code; sections 1-7 are the v2 evidence and are unchanged.",
         "Timing semantics: claim / allocation / recovery / message-age values are **simulation time** on a 20 Hz lockstep "
         "clock (50 ms ticks); planner compute and advisor inference are **wall-clock** on the development host.", ""]
    # 9.1 tests
    L += ["### 8.1 Requirement verification, v3 additions (automated tests)", "",
          "| requirement | verifying tests | result |", "|---|---|---|"]
    for req, names in REQ_TESTS_V3.items():
        res = [tests_v3.get(n, "not run") for n in names]
        verdict = "PASS" if res and all(r == "pass" for r in res) else ("FAIL" if "FAIL" in res else "incomplete")
        L.append(f"| {req} | {', '.join('`' + n + '`' for n in names)} | {verdict} |")
    jx = PROJECT_ROOT / "results" / "pytest_v3.xml"
    if jx.exists():
        n_cases = sum(1 for _ in ET.parse(jx).getroot().iter("testcase"))
        L += ["", f"Current suite (`results/pytest_v3.xml`): {sum(v == 'pass' for v in tests_v3.values())}/{len(tests_v3)} "
              f"test functions passed ({n_cases} test cases).  The v2 evidence (`results/pytest.xml`) is kept unchanged.", ""]
    # 9.2 regression
    regs = sorted((p for p in (PROJECT_ROOT / "results" / "regression").glob("*/regression.json")
                   if "history_identical" in json.loads(p.read_text())), key=lambda p: p.stat().st_mtime)
    if regs:
        r = json.loads(regs[-1].read_text())
        L += ["### 8.2 Behavioural regression against v2 (preflight P10, addendum B)", "",
              f"Latest record: `{regs[-1].parent.name}`.  Loss-free runs must reproduce the stored v2 results exactly; lossy runs must "
              "be self-reproducible (v2 lossy runs were not bit-reproducible, see section 9).", "",
              "| run | check | result | makespan v2 (s) | makespan now (s) |", "|---|---|---|---|---|"]
        for x in r["history_identical"]:
            L.append(f"| {x['campaign']}/{x['label']} seed {x['seed']} | identical to v2 | {'yes' if x['identical'] else 'NO'} | "
                     f"{f(x['makespan_old'], 2)} | {f(x['makespan_new'], 2)} |")
        for x in r["self_reproducible"]:
            L.append(f"| {x['campaign']}/{x['label']} seed {x['seed']} | self-reproducible | {'yes' if x['self_identical'] else 'NO'} | "
                     f"{f(x['makespan_v2_realization'], 2)} | {f(x['makespan_new'], 2)} |")
        L += ["", f"Overall: **{'PASS' if r['pass'] else 'FAIL'}**.", ""]
    # 9.3 comms_v3
    s = load("comms_v3")
    L += ["### 8.3 Communication impairment (FR-14, PR-10, acceptance gate 6)", ""]
    if s:
        conds = list(s["conditions"])
        L += cond_table(s, conds) + [""]
        L += ["| condition | effective delay median / max (s) | drops by reason | reordered | stale-rejected | peer outages: n, median, max (s) | isolation episodes (partitioned) | rejected deliveries |",
              "|---|---|---|---|---|---|---|---|"]
        for l in conds:
            c = s["conditions"][l]
            L.append(f"| {l} | {f(c.get('eff_delay_median_s'), 3)} / {f(c.get('eff_delay_max_s'), 3)} | "
                     f"{json.dumps(c.get('drops_by_reason_total', {}))} | {c.get('reorder_total')} | {c.get('rejected_stale_total')} | "
                     f"{c.get('peer_outages')}, {f(c.get('peer_outage_median_s'), 2)}, {f(c.get('peer_outage_max_s'), 2)} | "
                     f"{c.get('isolation_episodes')} ({c.get('isolation_partitioned')}) | {c.get('rejected_deliveries_total')} |")
        L += ["", "Change relative to `nominal`, paired by seed (negative = slower):", "",
              "| condition | paired seeds | median change (%) | 95% CI | seeds better / worse |", "|---|---|---|---|---|"]
        data = load_campaign(CAMP / "comms_v3")
        for l in conds:
            if l != "nominal":
                L.append(pct_row(l, paired_improvement(data["nominal"], data[l])))
        L += ["", "Configured delay is not the delivery delay: a message sent at tick k is delivered at the first tick at or after "
              "max(k+1, ts + latency + jitter), so configured 0, 10 and 50 ms all deliver after one 50 ms tick.  These are "
              "simulation impairment results, not a model of real wireless timing.", ""]
    else:
        L += ["_campaign not run_", ""]
    # 9.4 deadlock / congestion tails
    L += ["### 8.4 Deadlock and congestion episodes (v3 §10.8)", "",
          "Recovery time = detection until the robot acquires a node that reduces its distance to goal; the within-5 s "
          "fraction uses all deadlock episodes as denominator (timed-out / failed count as not recovered).", "",
          "| campaign / condition | deadlock episodes | recovered | timed out | failed (goal change / run end) | rerouted | recovery median / p90 / max (s) | within 5 s | congestion episodes (median s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for camp in ("comms_v3", "advisor_ablation"):
        sm = load(camp)
        if not sm:
            continue
        for l, c in sm["conditions"].items():
            e = c.get("episodes")
            if not e:
                continue
            o = e["outcomes"]
            L.append(f"| {camp}/{l} | {e['n_deadlock']} | {o['recovered']} | {o['timed_out']} | {o['failed_goal_changed']} / {o['failed_run_end']} | "
                     f"{e['rerouted']} | {f(e['recovery_median_s'], 2)} / {f(e['recovery_p90_s'], 2)} / {f(e['recovery_max_s'], 2)} | "
                     f"{f(e['within_5s_frac'], 2)} | {e['n_congestion']} ({f(e['congestion_median_s'], 1)}) |")
    L.append("")
    # 9.5 Option B
    L += option_b_section(load_campaign, paired_improvement)
    # 9.6 orchestrator
    L += ["### 8.6 Experiment orchestration and provenance (FR-13, acceptance gate 7)", "",
          "| campaign | runs attempted | failed | retried | run manifests |", "|---|---|---|---|---|"]
    for camp in V3_CAMPAIGNS:
        sm = load(camp)
        o = (sm or {}).get("orchestrator")
        if o:
            L.append(f"| {camp} | {o['attempted']} | {len(o['failed'])} {o['failed'] if o['failed'] else ''} | "
                     f"{len(o['retried'])} | {o['manifests']} |")
    L += ["", "Each manifest records scenario, seed, config hash, code version (sha1 of `src/`), model version, the fault "
          "schedule, start/end time, completion, agent exit codes and every orchestrator attempt.  The control boundary "
          "is shown by the FR-13 tests in 8.1: orchestrator modules have no dependency on agent, comms or benchmark code, "
          "agents accept only their own sensor frame (whitelist enforced in `SimLink`), and route / priority / winner "
          "fields injected into every sense frame were stripped and left the run bit-identical.", ""]
    v3_runs = sum(sum(c["n_runs"] for c in load(cp)["conditions"].values()) for cp in V3_CAMPAIGNS if load(cp))
    v3_col = sum(sum(c["collisions_total"] for c in load(cp)["conditions"].values()) for cp in V3_CAMPAIGNS if load(cp))
    L += [f"**PR-01, v3 campaigns:** {v3_col} collisions in {v3_runs} runs (tested configurations only; not a "
          "collision-freedom guarantee).", ""]
    return L


def option_b_section(load_campaign, paired_improvement) -> list[str]:
    L = ["### 8.5 Option B congestion-aware priority advisor (v3 §10.9, App. D, acceptance gate 8)", ""]
    ds_p = PROJECT_ROOT / "data" / "optb" / "dataset.json"
    m_p = PROJECT_ROOT / "models" / "optb_advisor.json"
    t_p = PROJECT_ROOT / "models" / "optb_advisor_test.json"
    if ds_p.exists():
        ds = json.loads(ds_p.read_text())
        L += ["**Dataset** (visual MuJoCo, policy A with decision-time feature logging; spec `docs/internal_updates/option_b_spec.md`):", "",
              "| condition | runs ok / expected | rows | labelled | dropped (incomplete 10 s window) | samples (robot has a goal) |",
              "|---|---|---|---|---|---|"]
        for k, c in ds["per_condition"].items():
            L.append(f"| {k} | {c['ok']} / {c['expected']} | {c['rows']} | {c['labelled']} | {c['dropped_incomplete_window']} | {c['samples']} |")
        a = ds["adj_max_check"]
        L += ["", f"Files: " + "; ".join(f"`{v['path']}` n={v['n']} sha1 {v['sha1']} (label mean {f(v['label_mean'], 3)}, "
                                           f"non-zero {f(v['label_nonzero_frac'], 2)})" for v in ds["files"].values()) + ".",
              f"Failed / missing generation runs: {ds['failed_runs'] or 'none'} / {ds['missing_runs'] or 'none'} (not substituted).",
              f"ADJ_MAX check (train/val): {a['blocking_pairs']} blocking pairs, fraction with goal-age gap < 20 s = "
              f"{f(a['frac_goal_gap_lt_20s'], 2)} -> ADJ_MAX = 20 s kept.", ""]
    if m_p.exists():
        mm = json.loads(m_p.read_text())
        L += [f"**Model selection** (validation split only; selected `{mm['selected']}`, version `{mm['model_version']}`, "
              f"{mm['n_params']} parameters, {mm['serialized_state_bytes']} B state dict, inference p50 / p99 "
              f"{mm['inference']['p50_ms']} / {mm['inference']['p99_ms']} ms on the development host, 1 thread):", "",
              "| candidate | val MSE | val MAE | Pearson | Spearman | parameters |", "|---|---|---|---|---|---|"]
        for k, c in mm["candidates"].items():
            L.append(f"| {k} | {f(c['mse'], 4)} | {f(c['mae'], 4)} | {f(c['pearson'], 3)} | {f(c['spearman'], 3)} | {c.get('n_params', '–')} |")
        L.append("")
    if t_p.exists():
        te = json.loads(t_p.read_text())
        L += [f"**Held-out test prediction quality** (one-time evaluation of model sha1 {te['model_sha1']}):", "",
              "| family | n | model MSE | model Spearman | heuristic Spearman |", "|---|---|---|---|---|"]
        for fam, v in te["by_family"].items():
            L.append(f"| {fam} | {v['model']['n']} | {f(v['model']['mse'], 4)} | {f(v['model']['spearman'], 3)} | "
                     f"{f(v['heuristic_B']['spearman'], 3)} |")
        L.append("")
    s = load("advisor_ablation")
    if not s:
        return L + ["_ablation campaign not run_", ""]
    data = load_campaign(CAMP / "advisor_ablation")
    L += ["**Fleet-level ablation** (held-out seeds 1-10, paired by seed; positive = method faster than reference):", "",
          "| family | comparison | paired seeds | median improvement (%) | 95% CI | seeds better / worse |", "|---|---|---|---|---|---|"]
    claims = []
    for fam in ("small", "open", "narrow", "fourway", "medium8"):
        g = lambda m: data.get(f"{fam}__{m}")
        for lab, ref, meth in [("C learned vs A", "pibt", "pibt_optb"), ("B heuristic vs A", "pibt", "pibt_heur"),
                               ("C learned vs B heuristic", "pibt_heur", "pibt_optb"), ("A vs stop-and-wait", "stop_wait", "pibt"),
                               ("central vs stop-and-wait", "stop_wait", "central")]:
            if g(ref) and g(meth):
                v = paired_improvement(g(ref), g(meth))
                L.append(pct_row(f"{fam} | {lab}", v))
                if lab == "C learned vs A" and v:
                    ci = v["ci95_median"]
                    worse = sum(x < 0 for x in v["per_seed"])
                    claims.append((fam, bool(ci and ci[0] > 0 and v["wins"] > worse)))
    L += ["", "| condition | runs | all delivered | makespan median (s) | collisions | advisor fallback fraction | adjustment non-zero fraction | inference p99 (ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for l, runs in data.items():
        c = s["conditions"][l]
        L.append(f"| {l} | {c['n_runs']} | {c['n_complete']}/{c['n_runs']} | {f(c['makespan_median'])} | {c['collisions_total']} | "
                 f"{f(c.get('optb_fallback_frac'), 3)} | {f(c.get('optb_nonzero_frac'), 2)} | {f(c.get('optb_infer_p99_ms'), 3)} |")
    ok = [fam for fam, good in claims if good]
    L += ["", f"**Learning-claim rule** (pre-registered: held-out paired median CI excludes 0 and wins > losses): "
          f"satisfied for {ok if ok else 'no family'}; "
          f"{'a benefit is claimed only for those families' if ok else 'no learning benefit is claimed'}.", ""]
    return L


def main() -> int:
    tests = junit(PROJECT_ROOT / "results" / "pytest.xml")
    L = [f"# Verification & Validation Report", "",
         f"Generated {date.today().isoformat()} by `python -m fleet.experiments.report` from `results/campaigns/*/summary.json`"
         f" and `results/pytest.xml`.  All runs are visual MuJoCo runs (accelerated mode), MiR100 robots, N = 3 unless stated.",
         ""]
    L += ["## 1. Requirement verification (automated tests)", "", "| requirement | verifying tests | result |", "|---|---|---|"]
    for req, names in REQ_TESTS.items():
        res = [tests.get(n, "not run") for n in names]
        verdict = "PASS" if res and all(r == "pass" for r in res) else ("FAIL" if "FAIL" in res else "incomplete")
        L.append(f"| {req} | {', '.join('`' + n + '`' for n in names)} | {verdict} |")
    n_pass = sum(v == "pass" for v in tests.values())
    jx = PROJECT_ROOT / "results" / "pytest.xml"
    n_cases = sum(1 for _ in ET.parse(jx).getroot().iter("testcase")) if jx.exists() else 0
    L += ["", f"Test suite: {n_pass}/{len(tests)} test functions passed ({n_cases} test cases including parametrisations).", ""]

    acc = load("acceptance")
    L += ["## 2. Acceptance campaign: overlapping paths (SC-02 / PR-02, App. D)", ""]
    if acc:
        L += cond_table(acc, ["stop_wait", "central", "pibt", "pibt_learned"]) + [""]
        L += ["Makespan reduction relative to stop-and-wait, paired by seed (same order stream):", ""]
        L += imp_table(acc, "improvement_vs_stop_wait") + [""]
        pibt = acc["improvement_vs_stop_wait"].get("pibt")
        if pibt:
            ci = pibt["ci95_median"]
            ok = pibt["median_pct"] >= 20.0
            L += [f"**PR-02 verdict:** distributed A* + PIBT median makespan reduction = {f(pibt['median_pct'])} % "
                  f"(95 % CI {f(ci[0]) if ci else '–'} – {f(ci[1]) if ci else '–'} %) -> "
                  f"{'meets' if ok else 'does NOT meet'} the >= 20 % target"
                  + ("" if ok else "; see section 9 for the limiting design variables") + ".", ""]
        tot = sum(c["collisions_total"] for c in acc["conditions"].values())
        L += [f"**PR-01 (this campaign):** {tot} collisions in {sum(c['n_runs'] for c in acc['conditions'].values())} runs.", ""]
        L += ["![makespan](../results/campaigns/acceptance/makespan.png)", ""]
    else:
        L += ["_campaign not run_", ""]

    for name, title, base_key in [("topology", "3. Topology sweep", "improvement_vs_stop_wait"),
                                  ("density", "4. Traffic density sweep", "improvement_vs_stop_wait"),
                                  ("blockage", "5. Dynamic blockage rate sweep", "improvement_vs_stop_wait")]:
        s = load(name)
        L += [f"## {title}", ""]
        if s:
            L += cond_table(s, list(s["conditions"])) + [""] + imp_table(s, base_key) + [""]
        else:
            L += ["_campaign not run_", ""]

    s = load("scale")
    L += ["## 5b. Fleet size (PR-07): N = 3, 5, 8 on the 21 x 12 warehouse", ""]
    if s:
        L += cond_table(s, list(s["conditions"])) + [""] + imp_table(s, "improvement_vs_stop_wait") + [""]
        L += ["Seeds where stop-and-wait did not deliver every order within the run limit are excluded from the paired "
              "comparison (this favours the baseline).", ""]
    else:
        L += ["_campaign not run_", ""]

    L += ["## 5c. Ablation: learned priority advisor vs distributed baseline (App. D 3 vs 4)", "",
          "| setting | paired seeds | median change of learned vs PIBT (%) | 95% CI | seeds better / worse |", "|---|---|---|---|---|"]
    from fleet.experiments.analysis import load_campaign, paired_improvement
    for camp, base_l, learn_l in [("acceptance", "pibt", "pibt_learned"), ("scale", "n8__pibt", "n8__pibt_learned")]:
        d = CAMP / camp
        if not d.exists():
            continue
        data = load_campaign(d)
        if base_l in data and learn_l in data:
            v = paired_improvement(data[base_l], data[learn_l])
            if v:
                ci = v["ci95_median"]
                worse = sum(x < 0 for x in v["per_seed"])
                L.append(f"| {camp}: {learn_l} vs {base_l} | {v['n_pairs']} | {f(v['median_pct'], 2)} | "
                         f"{'[' + f(ci[0], 2) + ', ' + f(ci[1], 2) + ']' if ci else '–'} | {v['wins']} / {worse} |")
    model_meta = PROJECT_ROOT / "models" / "priority_mlp.json"
    if model_meta.exists():
        mm = json.loads(model_meta.read_text())
        L += ["", f"Graph-level training (held-out validation, {len(mm['val_seeds'])} episodes): baseline "
              f"{f(mm['val_baseline'], 2)} steps, learned {f(mm['val_learned'], 2)} steps "
              f"({f(mm['val_improvement_pct'], 2)} %); selection: {mm.get('selection', {}).get('chosen')}.", ""]

    s = load("comms")
    L += ["## 6. Communication degradation (SC-04, acceptance item 6)", ""]
    if s:
        L += cond_table(s, list(s["conditions"])) + [""]
        L += ["Change relative to ideal communication (lat0_loss0), paired by seed (negative = slower):", ""]
        L += imp_table(s, "improvement_vs_lat0_loss0") + [""]
        L += ["| condition | measured receive drop fraction | B_net per robot (B/s) | messages/s (fleet) |", "|---|---|---|---|"]
        for l, c in s["conditions"].items():
            L.append(f"| {l} | {f(c['rx_drop_frac_median'], 3)} | {f(c['bnet_per_robot_Bps_median'], 0)} | {f(c['msgs_per_s_median'], 1)} |")
        L.append("")
    else:
        L += ["_campaign not run_", ""]

    L += ["## 7. Safety, latency, compute and communication burden (v2 campaigns)", "",
          "| campaign | runs | collisions | min chassis-centre gap (m) | claim latency median (s) | allocation latency median (s) | planner compute p99 max (ms) | advisor inference p99 max (ms) | B_net per robot median (B/s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    total_runs = total_col = 0
    for p in sorted(CAMP.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if s["campaign"] not in V2_CAMPAIGNS:
            continue
        cs = list(s["conditions"].values())
        runs = sum(c["n_runs"] for c in cs)
        col = sum(c["collisions_total"] for c in cs)
        total_runs += runs
        total_col += col
        mins = [c["min_pair_dist"] for c in cs if c["min_pair_dist"] is not None]
        med = lambda k: sorted(c[k] for c in cs if c[k] is not None)[len([c for c in cs if c[k] is not None]) // 2] if any(c[k] is not None for c in cs) else None
        mx = lambda k: max((c[k] for c in cs if c[k] is not None), default=None)
        L.append(f"| {s['campaign']} | {runs} | {col} | {f(min(mins), 2) if mins else '–'} | {f(med('claim_latency_median'), 3)} | "
                 f"{f(med('alloc_latency_median'), 2)} | {f(mx('compute_p99_ms_max'), 2)} | {f(mx('advisor_p99_ms_max'), 3)} | "
                 f"{f(med('bnet_per_robot_Bps_median'), 0)} |")
    L += ["", f"**PR-01 overall:** {total_col} collisions (eq. 8 onsets + MuJoCo chassis contacts) in {total_runs} campaign runs.", ""]
    notes = PROJECT_ROOT / "docs" / "vv_findings.md"
    L += v3_sections(junit(PROJECT_ROOT / "results" / "pytest_v3.xml"))
    L += ["## 9. Findings and limitations", ""]
    L += [notes.read_text(encoding="utf-8")] if notes.exists() else ["_see docs/vv_findings.md (written after the campaigns)_"]
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
