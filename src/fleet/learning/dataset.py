"""Option B dataset from visual-MuJoCo feature logs (design doc v3 §10.9; preflight P1, P2, P9).

    python -m fleet.learning.dataset splits      # write data/optb/splits.json (BEFORE any data is generated)
    python -m fleet.learning.dataset build       # after the optb_data campaign: labels + npz files + dataset.json

Every robot agent of a run with `fleet.log_features = true` writes optb_<robot>.jsonl: one row every 10 control
ticks (2 Hz) with its decision-time feature vector and the label ingredients (held nodes, waiting flag, intent path).
Rows are joined across robots exactly on the lockstep tick.  Train/val and test go to separate files; the trainer
never opens the test file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from fleet.common.config import PROJECT_ROOT
from fleet.learning.congestion import FEATURES, N_FEATURES, SCHEMA_VERSION, blocking_labels, schema_hash

DATA = PROJECT_ROOT / "data" / "optb"
CAMPAIGN = PROJECT_ROOT / "results" / "campaigns" / "optb_data"
SPLITS = DATA / "splits.json"
SEEDS = {"train": list(range(101, 131)), "val": list(range(131, 141)), "test": list(range(1, 11))}
FAMILIES = {
    "small": {"world.map": "configs/maps/warehouse_small.txt", "fleet.n_robots": 3, "tasks.count": 12, "tasks.rate": 0.25},
    "open": {"world.map": "configs/maps/topo_open.txt", "fleet.n_robots": 3, "tasks.count": 12, "tasks.rate": 0.25},
    "narrow": {"world.map": "configs/maps/topo_narrow.txt", "fleet.n_robots": 3, "tasks.count": 12, "tasks.rate": 0.25},
    "fourway": {"world.map": "configs/maps/topo_fourway.txt", "fleet.n_robots": 3, "tasks.count": 12, "tasks.rate": 0.25},
    "medium8": {"world.map": "configs/maps/warehouse_medium.txt", "fleet.n_robots": 8, "tasks.count": 20, "tasks.rate": 0.3},
}
TEST_ONLY = ["medium8"]                       # held-out layout


def conditions() -> list[tuple[str, str, list[int]]]:
    """(family, split, seeds) - the exact run matrix of the optb_data campaign (P9): 120 + 40 + 40 + 10 = 210 runs."""
    out = []
    for fam in FAMILIES:
        for split in (("test",) if fam in TEST_ONLY else ("train", "val", "test")):
            out.append((fam, split, SEEDS[split]))
    return out


def write_splits() -> dict:
    from fleet.experiments.provenance import code_version
    if SPLITS.exists():
        raise FileExistsError(f"{SPLITS} is frozen; delete it explicitly to redefine the splits")
    for a in SEEDS:
        for b in SEEDS:
            assert a == b or not set(SEEDS[a]) & set(SEEDS[b])
    conds = conditions()
    spec = {"schema_version": SCHEMA_VERSION, "schema_hash": schema_hash(), "features": FEATURES,
            "seeds": SEEDS, "families": FAMILIES, "test_only_families": TEST_ONLY,
            "scenario": "configs/scenarios/exp_overlap.toml",
            "generation": {"fleet.planner": "pibt", "fleet.priority_advisor": "none", "fleet.log_features": True,
                           "run.mode": "accelerated"},
            "expected_runs": {f"{f}__{s}": len(sd) for f, s, sd in conds}, "expected_total": sum(len(sd) for *_, sd in conds),
            "code_version": code_version(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rule": "test split frozen before model selection; trainer reads dataset_trainval.npz only"}
    DATA.mkdir(parents=True, exist_ok=True)
    SPLITS.write_text(json.dumps(spec, indent=2))
    return spec


def read_run(run_dir: Path) -> dict[str, dict[int, dict]]:
    rows: dict[str, dict[int, dict]] = {}
    for p in sorted(run_dir.glob("optb_*.jsonl")):
        robot = p.stem[len("optb_"):]
        rows[robot] = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[robot][int(r["tick"])] = r
    return rows


def run_samples(run_dir: Path) -> tuple[list[np.ndarray], list[float], list[tuple], dict, list[float]]:
    """Feature vectors and labels of one run (robots with a goal only), plus drop counts and the blocking-pair goal
    age gaps used to sanity-check ADJ_MAX (P3)."""
    rows = read_run(run_dir)
    labels = blocking_labels(rows)
    X, y, idx = [], [], []
    n_all = sum(len(v) for v in rows.values())
    kept_goal = 0
    for robot, tick, lab in labels:
        r = rows[robot][tick]
        if not r["has_goal"]:
            continue
        kept_goal += 1
        X.append(np.asarray(r["x"], dtype=np.float32))
        y.append(lab)
        idx.append((robot, tick))
    gaps = []
    for robot, rr in rows.items():
        for tick, r in rr.items():
            if not r["has_goal"]:
                continue
            for other, oo in rows.items():
                o = oo.get(tick)
                if other != robot and o and o["has_goal"] and o["waiting"] and set(r["held"]) & set(o["path"][:4]):
                    gaps.append(abs(r["goal_since"] - o["goal_since"]))
    counts = {"rows": n_all, "labelled": len(labels), "dropped_incomplete_window": n_all - len(labels),
              "kept_with_goal": kept_goal}
    return X, y, idx, counts, gaps


def build(camp: Path = CAMPAIGN) -> dict:
    spec = json.loads(SPLITS.read_text())
    if spec["schema_hash"] != schema_hash():
        raise ValueError("feature schema changed after the splits were frozen")
    runs_json = camp / "runs.json"
    attempts = json.loads(runs_json.read_text()) if runs_json.exists() else []
    out = {"trainval": {"X": [], "y": [], "split": [], "family": [], "seed": [], "robot": [], "tick": []},
           "test": {"X": [], "y": [], "split": [], "family": [], "seed": [], "robot": [], "tick": []}}
    report = {"per_condition": {}, "failed_runs": [], "missing_runs": [], "gaps_trainval": []}
    for fam, split, seeds in conditions():
        cond = f"{fam}__{split}"
        c = {"expected": len(seeds), "ok": 0, "rows": 0, "labelled": 0, "dropped_incomplete_window": 0, "samples": 0}
        for s in seeds:
            d = camp / cond / f"seed_{s}"
            res = d / "result.json"
            if not res.exists():
                st = next((a["status"] for a in attempts if a["label"] == cond and a["seed"] == s), "not_run")
                (report["failed_runs"] if st != "not_run" else report["missing_runs"]).append(
                    {"condition": cond, "seed": s, "status": st})
                continue                      # not substituted (P9)
            X, y, idx, counts, gaps = run_samples(d)
            c["ok"] += 1
            for k in ("rows", "labelled", "dropped_incomplete_window"):
                c[k] += counts[k]
            c["samples"] += len(y)
            part = out["test" if split == "test" else "trainval"]
            part["X"] += X
            part["y"] += y
            part["split"] += [split] * len(y)
            part["family"] += [fam] * len(y)
            part["seed"] += [s] * len(y)
            part["robot"] += [r for r, _ in idx]
            part["tick"] += [t for _, t in idx]
            if split != "test":
                report["gaps_trainval"] += gaps
        report["per_condition"][cond] = c
    files = {}
    for name, part in out.items():
        p = DATA / f"dataset_{name}.npz"
        np.savez_compressed(p, X=np.asarray(part["X"], dtype=np.float32).reshape(-1, N_FEATURES),
                            y=np.asarray(part["y"], dtype=np.float32), split=np.asarray(part["split"]),
                            family=np.asarray(part["family"]), seed=np.asarray(part["seed"], dtype=np.int32),
                            robot=np.asarray(part["robot"]), tick=np.asarray(part["tick"], dtype=np.int64))
        files[name] = {"path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"), "n": len(part["y"]),
                       "sha1": hashlib.sha1(p.read_bytes()).hexdigest()[:12],
                       "label_mean": float(np.mean(part["y"])) if part["y"] else None,
                       "label_nonzero_frac": float(np.mean(np.asarray(part["y"]) > 0)) if part["y"] else None}
    g = np.asarray(report.pop("gaps_trainval"))
    meta = {"dataset_version": "optb-data-v1", "schema_hash": schema_hash(), "files": files,
            "splits_sha1": hashlib.sha1(SPLITS.read_bytes()).hexdigest()[:12], **report,
            "adj_max_check": {"blocking_pairs": int(g.size),
                              "frac_goal_gap_lt_20s": float(np.mean(g < 20.0)) if g.size else None,
                              "gap_quantiles_s": [float(q) for q in np.quantile(g, [0.1, 0.25, 0.5, 0.75, 0.9])] if g.size else None},
            "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (DATA / "dataset.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["splits", "build"])
    a = ap.parse_args(argv)
    print(json.dumps(write_splits() if a.cmd == "splits" else build(), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
