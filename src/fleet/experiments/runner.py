"""Batch experiment runner (WBS 9.0).  Every run is a full visual MuJoCo run (accelerated mode) with its own
processes and ports; runs are resumable (existing result.json files are kept).

Campaign file (TOML):
    [campaign]
    name = "acceptance"
    scenario = "configs/scenarios/exp_overlap.toml"
    seeds = [1, 2, ...]          # >= 10 per condition (PR-08)
    parallel = 3
    [[condition]]
    label = "pibt"
    set = { "fleet.planner" = "pibt" }

    python -m fleet.experiments.runner configs/experiments/acceptance.toml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fleet.common.config import PROJECT_ROOT, resolve

RESULTS = PROJECT_ROOT / "results" / "campaigns"


def expand(camp: dict) -> list[dict]:
    c = camp["campaign"]
    runs = []
    for cond in camp["condition"]:
        for seed in cond.get("seeds", c["seeds"]):
            runs.append({"label": cond["label"], "seed": int(seed), "set": dict(cond.get("set", {})),
                         "scenario": cond.get("scenario", c["scenario"])})
    return runs


def run_one(camp_name: str, r: dict, port_base: int, timeout: float, root: Path = RESULTS) -> dict:
    out_dir = root / camp_name
    res_path = out_dir / r["label"] / f"seed_{r['seed']}" / "result.json"
    if res_path.exists():
        return {"status": "cached", **r}
    sets = {"run.name": r["label"], "run.seed": r["seed"], "run.port_base": port_base,
            "run.results_dir": str(out_dir).replace("\\", "/"), "run.mode": "accelerated", **r["set"]}
    args = [sys.executable, "-m", "fleet.launch", "--scenario", str(resolve(r["scenario"]))]
    for k, v in sets.items():
        args += ["--set", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
    t0 = time.time()
    try:
        p = subprocess.run(args, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        ok = p.returncode == 0 and res_path.exists()
        status = "ok" if ok else f"failed rc={p.returncode}"
        if not ok:
            (out_dir / "failures").mkdir(parents=True, exist_ok=True)
            (out_dir / "failures" / f"{r['label']}_seed{r['seed']}_port{port_base}.log").write_text(
                p.stdout[-5000:] + p.stderr[-5000:])
    except subprocess.TimeoutExpired:
        status = "timeout"
    _note_attempt(res_path.parent, status, port_base)
    return {"status": status, "wall": round(time.time() - t0, 1), **r}


def _note_attempt(run_dir: Path, status: str, port_base: int) -> None:
    """Orchestrator record of every attempt (also for runs that never produced a result)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "manifest.json"
    m = json.loads(p.read_text()) if p.exists() else {"status": "not_started"}
    m.setdefault("attempts", []).append({"status": status, "port_base": port_base,
                                         "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
    m["orchestrator_status"] = status
    p.write_text(json.dumps(m, indent=2))


def run_campaign(path: str | Path, parallel: int | None = None, timeout: float = 1800, root: Path = RESULTS) -> list[dict]:
    with open(resolve(path), "rb") as f:
        camp = tomllib.load(f)
    name = camp["campaign"]["name"]
    runs = expand(camp)
    par = parallel or int(camp["campaign"].get("parallel", 2))
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "campaign.json").write_text(json.dumps(camp, indent=2))
    lock = threading.Lock()
    done: list[dict] = []
    port0 = 11000 + (zlib.crc32(name.encode()) % 40) * 400

    def job(i_r):
        i, r = i_r
        res = run_one(name, r, port0 + (i % 64) * 60 + 10, timeout, root)
        if res["status"].startswith("failed"):
            # infrastructure failures (e.g. a port transiently reserved by Windows/another process) are retried once on
            # a different port block; the first attempt's log stays in failures/ for the record
            res = run_one(name, r, 30000 + (zlib.crc32(f"{name}{i}".encode()) % 300) * 80, timeout, root)
            res["retried"] = True
        with lock:
            done.append(res)
            print(f"[{len(done)}/{len(runs)}] {res['label']} seed={res['seed']} {res['status']} {res.get('wall', '')}s",
                  flush=True)
        return res

    with ThreadPoolExecutor(max_workers=par) as ex:
        list(ex.map(job, enumerate(runs)))
    (root / name / "runs.json").write_text(json.dumps(done, indent=2))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args(argv)
    res = run_campaign(a.campaign, a.parallel, a.timeout)
    bad = [r for r in res if r["status"] not in ("ok", "cached")]
    print(json.dumps({"runs": len(res), "failed": len(bad)}))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
