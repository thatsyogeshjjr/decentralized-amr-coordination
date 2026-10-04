"""Train / select / profile the Option B advisor on MuJoCo-generated data (design doc v3 §10.9; preflight P1-P3, P9).

    python -m fleet.learning.train_optb train      # model selection on train/val only -> models/optb_advisor.pt
    python -m fleet.learning.train_optb evaluate   # ONE-TIME held-out test evaluation of the frozen model

`train` reads data/optb/dataset_trainval.npz only and refuses to run once a test evaluation exists (the test split is
frozen before model selection and must not influence it).  Candidates: the handcrafted heuristic (fixed weights),
a constant predictor, a linear model and small MLPs; the network is selected by validation MSE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from fleet.common.config import PROJECT_ROOT
from fleet.learning.congestion import FEATURES, heuristic_score, schema_hash
from fleet.learning.dataset import DATA
from fleet.learning.optb_model import OptBNet, load, n_params, predict, save, serialized_bytes

MODEL = PROJECT_ROOT / "models" / "optb_advisor.pt"
TEST_EVAL = PROJECT_ROOT / "models" / "optb_advisor_test.json"
CANDIDATES = [[], [16], [32], [32, 16], [64, 32]]
HP = {"lr": 1e-3, "weight_decay": 1e-4, "batch": 256, "max_epochs": 300, "patience": 20, "torch_seed": 0}


def avg_rank(a: np.ndarray) -> np.ndarray:
    """Ranks with ties sharing their average rank (the label is zero for most samples)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(sa)) + 1, a.size]
    r = np.empty(a.size)
    r[order] = np.repeat((bounds[:-1] + bounds[1:] - 1) / 2.0, np.diff(bounds))
    return r


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    pred, y = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    varies = np.ptp(pred) > 0 and np.ptp(y) > 0
    return {"n": int(y.size), "mse": float(np.mean((pred - y) ** 2)), "mae": float(np.mean(np.abs(pred - y))),
            "pearson": float(np.corrcoef(pred, y)[0, 1]) if varies else None,
            "spearman": float(np.corrcoef(avg_rank(pred), avg_rank(y))[0, 1]) if varies else None}


def _fit(hidden: list[int], Xtr, ytr, Xva, yva, mu, sd) -> tuple[OptBNet, dict]:
    torch.manual_seed(HP["torch_seed"])
    m = OptBNet(hidden)
    m.mu.copy_(torch.from_numpy(mu))
    m.sd.copy_(torch.from_numpy(sd))
    opt = torch.optim.Adam(m.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    g = torch.Generator().manual_seed(HP["torch_seed"])
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(HP["max_epochs"]):
        m.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), HP["batch"]):
            b = perm[i:i + HP["batch"]]
            opt.zero_grad()
            loss = torch.mean((m(xt[b]) - yt[b]) ** 2)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(torch.mean((m(xv) - yv) ** 2))
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    m.load_state_dict(best_state)
    return m.eval(), {"epochs": ep + 1, "val_mse": best}


def _latency(m: OptBNet, X: np.ndarray, n: int = 3000) -> dict:
    torch.set_num_threads(1)
    xs = [torch.from_numpy(X[i % len(X)]).unsqueeze(0) for i in range(n)]
    ts = []
    with torch.no_grad():
        for x in xs:
            t0 = time.perf_counter()
            m(x)
            ts.append(time.perf_counter() - t0)
    a = np.array(ts[200:]) * 1000
    return {"p50_ms": round(float(np.percentile(a, 50)), 4), "p99_ms": round(float(np.percentile(a, 99)), 4),
            "host": "development workstation (not an edge device)", "threads": 1}


def train() -> dict:
    if TEST_EVAL.exists():
        raise RuntimeError(f"{TEST_EVAL.name} exists: the test set was already used; retraining is not allowed")
    path = DATA / "dataset_trainval.npz"
    meta_ds = json.loads((DATA / "dataset.json").read_text())
    d = np.load(path)
    assert set(np.unique(d["split"])) <= {"train", "val"}, "test rows must never be in the trainval file"
    tr, va = d["split"] == "train", d["split"] == "val"
    Xtr, ytr, Xva, yva = d["X"][tr], d["y"][tr], d["X"][va], d["y"][va]
    assert not set(d["seed"][tr]) & set(d["seed"][va])
    mu = Xtr.mean(0).astype(np.float32)
    sd = np.maximum(Xtr.std(0), 1e-3).astype(np.float32)
    table = {"constant_mean": _metrics(np.full_like(yva, ytr.mean()), yva),
             "heuristic_B": _metrics(np.array([heuristic_score(x) for x in Xva], dtype=np.float32), yva)}
    fits = {}
    for h in CANDIDATES:
        t0 = time.time()
        m, info = _fit(h, Xtr, ytr, Xva, yva, mu, sd)
        name = "linear" if not h else "mlp_" + "x".join(map(str, h))
        table[name] = {**_metrics(predict(m, Xva), yva), **info, "n_params": n_params(m),
                       "bytes": serialized_bytes(m), "train_s": round(time.time() - t0, 1)}
        fits[name] = (m, h)
        print(name, json.dumps(table[name]), flush=True)
    best = min(fits, key=lambda k: table[k]["mse"])
    m, h = fits[best]
    meta = {"model_version": f"optb-v1-{hashlib.sha1(json.dumps(table[best], sort_keys=True).encode()).hexdigest()[:8]}",
            "selected": best, "selection_metric": "validation MSE", "features": FEATURES, "schema_hash": schema_hash(),
            "label": "blocking impact y in [0,1] (preflight P2)", "hyperparameters": HP, "candidates": table,
            "train_metrics": _metrics(predict(m, Xtr), ytr), "val_metrics": table[best],
            "normalization": {"mu": mu.tolist(), "sd": sd.tolist(), "fitted_on": "train split only"},
            "n_train": int(tr.sum()), "n_val": int(va.sum()), "dataset": meta_ds["files"]["trainval"],
            "dataset_version": meta_ds["dataset_version"], "splits_sha1": meta_ds["splits_sha1"],
            "train_seeds": sorted(set(int(s) for s in d["seed"][tr])), "val_seeds": sorted(set(int(s) for s in d["seed"][va])),
            "serialized_state_bytes": serialized_bytes(m), "inference": _latency(m, Xva),
            "trained": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save(m, h, MODEL, meta)
    return json.loads(MODEL.with_suffix(".json").read_text())


def evaluate() -> dict:
    """One-time held-out evaluation of the frozen model (prediction quality; fleet benefit is the MuJoCo ablation)."""
    if TEST_EVAL.exists():
        return json.loads(TEST_EVAL.read_text())
    m, meta = load(MODEL)
    d = np.load(DATA / "dataset_test.npz")
    out = {"model_version": meta["model_version"], "model_sha1": hashlib.sha1(MODEL.read_bytes()).hexdigest()[:12],
           "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"), "by_family": {}}
    for fam in ["all"] + sorted(set(d["family"])):
        sel = np.ones(len(d["y"]), bool) if fam == "all" else d["family"] == fam
        X, y = d["X"][sel], d["y"][sel]
        out["by_family"][fam] = {"model": _metrics(predict(m, X), y),
                                 "heuristic_B": _metrics(np.array([heuristic_score(x) for x in X], dtype=np.float32), y)}
    TEST_EVAL.write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate"])
    a = ap.parse_args(argv)
    print(json.dumps(train() if a.cmd == "train" else evaluate(), indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
