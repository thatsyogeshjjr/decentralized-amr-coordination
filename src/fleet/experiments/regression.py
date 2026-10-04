"""Behavioural regression against stored v2 campaign results (preflight P10 + addendum B).

Re-runs a fixed set of (campaign, condition, seed) runs with the current code into results/regression/<tag>/ (the
historical files are only read) and compares makespan, every order's pickup / delivery time and the collision count.

  * loss-free runs must reproduce the stored v2 results exactly (lockstep delivery barrier)
  * lossy runs: v2 lossy runs were not bit-reproducible (join-phase beacons consumed loss-RNG draws in a number that
    depended on wall-clock join timing; fixed in v3), so they are checked for *self*-reproducibility: two runs with
    the current code must be identical.  Their difference from the stored v2 realization is reported, not gated.

    python -m fleet.experiments.regression <tag>
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib

from fleet.common.config import PROJECT_ROOT, resolve
from fleet.experiments.runner import RESULTS, run_one

SET = [("acceptance", "pibt", s) for s in (1, 2, 3)] + [("acceptance", "stop_wait", s) for s in (1, 2, 3)]
LOSSY = [("comms", "lat0_loss20", s) for s in (1, 2)]
ROOT = PROJECT_ROOT / "results" / "regression"


def _cond(camp: str, label: str) -> dict:
    with open(resolve(f"configs/experiments/{camp}.toml"), "rb") as f:
        c = tomllib.load(f)
    cond = next(x for x in c["condition"] if x["label"] == label)
    return {"label": f"{camp}__{label}", "set": dict(cond.get("set", {})),
            "scenario": cond.get("scenario", c["campaign"]["scenario"])}


def fingerprint(r: dict) -> dict:
    t = r["tasks"]
    return {"makespan": t["makespan"], "n_delivered": t["n_delivered"],
            "orders": [(o["id"], o["picked_t"], o["delivered_t"]) for o in t["orders"]],
            "collisions": r["safety"]["collisions"]}


def _one(tag: str, camp: str, label: str, seed: int, port: int) -> dict | None:
    r = {**_cond(camp, label), "seed": seed}
    run_one(tag, r, port, 1800, root=ROOT)
    p = ROOT / tag / r["label"] / f"seed_{seed}" / "result.json"
    return fingerprint(json.loads(p.read_text())) if p.exists() else None


def _old(camp: str, label: str, seed: int) -> dict | None:
    p = RESULTS / camp / label / f"seed_{seed}" / "result.json"
    return fingerprint(json.loads(p.read_text())) if p.exists() else None


def run(tag: str) -> dict:
    out = {"tag": tag, "history_identical": [], "self_reproducible": []}
    port = 16000
    for camp, label, seed in SET:
        port += 60
        a, b = _old(camp, label, seed), _one(tag, camp, label, seed, port)
        row = {"campaign": camp, "label": label, "seed": seed, "identical": a is not None and a == b,
               "makespan_old": a and a["makespan"], "makespan_new": b and b["makespan"],
               "collisions_new": b and b["collisions"]}
        out["history_identical"].append(row)
        print(json.dumps(row), flush=True)
    for camp, label, seed in LOSSY:
        port += 60
        b1 = _one(tag, camp, label, seed, port)
        b2 = _one(tag + "_rep", camp, label, seed, port + 30)
        a = _old(camp, label, seed)
        row = {"campaign": camp, "label": label, "seed": seed, "self_identical": b1 is not None and b1 == b2,
               "makespan_new": b1 and b1["makespan"], "makespan_v2_realization": a and a["makespan"],
               "equals_v2_realization": a == b1, "collisions_new": b1 and b1["collisions"]}
        out["self_reproducible"].append(row)
        print(json.dumps(row), flush=True)
    out["pass"] = all(x["identical"] for x in out["history_identical"]) and \
        all(x["self_identical"] for x in out["self_reproducible"])
    (ROOT / tag).mkdir(parents=True, exist_ok=True)
    (ROOT / tag / "regression.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    a = ap.parse_args(argv)
    res = run(a.tag)
    print(json.dumps({"pass": res["pass"]}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
