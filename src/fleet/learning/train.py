"""Optimize the priority MLP for fleet makespan with the cross-entropy method (CEM) in the graph-level simulator.

The training objective is the fleet-level one named in the design doc (§10.9): mean makespan over a batch of
seeded lifelong pickup/delivery episodes (different fleet sizes and arrival rates), with a penalty per unfinished
task.  Training and validation seeds are disjoint; the final checkpoint stores seeds, hyper-parameters and the
baseline (no-advisor) score on the same validation set, so improvements are reported, not assumed.

    python -m fleet.learning.train --out models/priority_mlp.pt [--iters 40 --pop 32]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from fleet.common.config import resolve
from fleet.learning.gridsim import run_tasks
from fleet.learning.model import NumpyMLP, n_params, save, to_torch
from fleet.world.mapgrid import WarehouseMap

MAPS = ["configs/maps/warehouse_small.txt"]


def make_episode(m: WarehouseMap, seed: int, n_robots: int, n_tasks: int, rate: float):
    rng = np.random.default_rng(seed)
    P, D = m.stations("pickup"), m.stations("drop")
    t, tasks = 0.0, []
    for _ in range(n_tasks):
        t += rng.exponential(1.0 / rate)
        tasks.append((int(t), int(rng.choice(P)), int(rng.choice(D))))
    starts = [m.starts[i] for i in range(n_robots)] if n_robots <= len(m.starts) else None
    if starts is None:
        free = [n for n in m.nodes if m.kind[n] == "aisle"]
        idx = rng.choice(len(free), size=n_robots, replace=False)
        starts = [free[i] for i in idx]
    return starts, tasks


def episode_set(seeds: list[int], profile: str = "congested") -> list[tuple]:
    """'congested' (default): fleets of 3/5/6 robots with steady (0.5/step) or burst (3/step) arrivals - priorities
    only matter when robots compete for aisles; 'light': 3/5 robots, 0.3-0.6 orders/step."""
    out = []
    for s in seeds:
        m = WarehouseMap.load(resolve(MAPS[s % len(MAPS)]))
        if profile == "light":
            n, rate = [3, 5][s % 2], [0.3, 0.6][(s // 2) % 2]
        else:
            n, rate = [3, 5, 6][s % 3], [0.5, 3.0][(s // 3) % 2]
        starts, tasks = make_episode(m, s, n, 24, rate)
        out.append((m, starts, tasks, s))
    return out


def evaluate(policy, episodes, max_steps: int = 1500) -> float:
    total = 0.0
    for m, starts, tasks, s in episodes:
        r = run_tasks(m, starts, tasks, priority_fn=policy, max_steps=max_steps, seed=s, dwell=1)
        total += (r.steps if r.steps is not None else max_steps) + 50.0 * (len(tasks) - r.completed)
    return total / len(episodes)


def collect_features(episodes, cap: int = 20000) -> np.ndarray:
    """Dataset generation (WBS 7.0): feature vectors seen by robots with a goal during baseline rollouts."""
    xs: list[np.ndarray] = []

    def rec(x):
        if len(xs) < cap:
            xs.append(np.array(x, dtype=np.float32))
        return 0.0

    for m, starts, tasks, s in episodes:
        run_tasks(m, starts, tasks, priority_fn=rec, seed=s, dwell=1, max_steps=1500)
    return np.stack(xs)


def fit_prior(X: np.ndarray, target_fn, seed: int, epochs: int = 2000) -> np.ndarray:
    """Supervised warm start: fit the eq.-7 MLP (PyTorch, Adam, MSE) to a prior priority function; returns the
    flattened parameter vector in NumpyMLP order."""
    import torch
    from fleet.learning.model import PriorityMLP
    torch.manual_seed(seed)
    m = PriorityMLP()
    y = torch.tensor([target_fn(x) for x in X], dtype=torch.float32).unsqueeze(1)
    Xt = torch.from_numpy(X)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m(Xt), y)
        loss.backward()
        opt.step()
    return np.concatenate([m.l1.weight.detach().numpy().ravel(), m.l1.bias.detach().numpy(), m.l2.weight.detach().numpy().ravel(),
                           m.l2.bias.detach().numpy(), m.l3.weight.detach().numpy().ravel(), m.l3.bias.detach().numpy()]).astype(np.float64)


def cem(train_eps, iters: int, pop: int, elite: int, seed: int, sigma0: float = 0.5, log=print, batch: int | None = None,
        mu0: np.ndarray | None = None):
    """Cross-entropy method.  With `batch`, every iteration draws a fresh batch of episodes from the training pool
    (identical for all candidates of that iteration) - this regularises against over-fitting a fixed episode set.
    Returns ((best_score, best_theta), history); the deployable policy is history[-1]['mu'] (the final mean)."""
    rng = np.random.default_rng(seed)
    d = n_params()
    mu = np.zeros(d) if mu0 is None else np.array(mu0, dtype=np.float64)
    sigma = np.full(d, sigma0)
    best = (float("inf"), mu.copy())
    history = []
    for it in range(iters):
        eps = train_eps if batch is None else [train_eps[i] for i in rng.choice(len(train_eps), size=batch, replace=False)]
        cands = mu + sigma * rng.standard_normal((pop, d))
        cands[0] = mu                                          # always re-evaluate the current mean
        scores = np.array([evaluate(NumpyMLP(c), eps) for c in cands])
        order = np.argsort(scores)
        el = cands[order[:elite]]
        mu = el.mean(axis=0)
        sigma = 0.8 * el.std(axis=0) + 0.2 * sigma + 0.01
        if scores[order[0]] < best[0]:
            best = (float(scores[order[0]]), cands[order[0]].copy())
        history.append({"iter": it, "best": round(float(scores[order[0]]), 3), "mean": round(float(scores.mean()), 3),
                        "elite_mean": round(float(scores[order[:elite]].mean()), 3),
                        "batch_baseline": round(float(evaluate(None, eps)), 3) if batch is not None else None})
        log(json.dumps(history[-1]))
    history[-1]["mu"] = mu.tolist()
    return best, history


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/priority_mlp.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=int, default=12, help="number of training episodes")
    ap.add_argument("--val", type=int, default=24, help="number of held-out validation episodes")
    ap.add_argument("--profile", default="congested", choices=["congested", "light"])
    ap.add_argument("--batch", type=int, default=None, help="episodes drawn from the training pool per iteration")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="warm start: fit the MLP to -prior * normalized distance-to-goal before CEM (0 = off)")
    ap.add_argument("--select", type=int, default=24, help="selection episodes (disjoint from train and validation)")
    a = ap.parse_args(argv)
    train_seeds = list(range(1000, 1000 + a.train))
    val_seeds = list(range(5000, 5000 + a.val))
    train_eps, val_eps = episode_set(train_seeds, a.profile), episode_set(val_seeds, a.profile)
    t0 = time.time()
    baseline_train = evaluate(None, train_eps)
    print(json.dumps({"baseline_train": baseline_train}), flush=True)
    mu0 = None
    prior_info = None
    if a.prior > 0:
        X = collect_features(train_eps)
        mu0 = fit_prior(X, lambda x: -a.prior * float(x[0]), a.seed)
        prior_info = {"target": f"-{a.prior} * x0 (normalized distance to goal)", "samples": int(len(X))}
        print(json.dumps({"prior_fit_samples": int(len(X))}), flush=True)
    (best_score, best_theta), history = cem(train_eps, a.iters, a.pop, a.elite, a.seed,
                                            log=lambda s: print(s, flush=True), batch=a.batch,
                                            sigma0=0.1 if mu0 is not None else 0.5, mu0=mu0)
    theta = np.array(history[-1].pop("mu")) if a.batch else best_theta     # final CEM mean (less over-fit)
    # model selection on a separate episode set (validation stays untouched for reporting)
    sel_eps = episode_set(list(range(3000, 3000 + a.select)), a.profile)
    candidates = {"cem_mean": theta}
    if mu0 is not None:
        candidates["prior_fit"] = mu0
    sel = {k: evaluate(NumpyMLP(v), sel_eps) for k, v in candidates.items()}
    chosen = min(sel, key=sel.get)
    theta = candidates[chosen]
    val_learned = evaluate(NumpyMLP(theta), val_eps)
    val_baseline = evaluate(None, val_eps)
    meta = {"objective": "mean makespan (steps) + 50 * unfinished tasks, graph-level PIBT episodes",
            "train_seeds": train_seeds, "val_seeds": val_seeds, "cem": vars(a), "history": history,
            "train_best": best_score, "baseline_train": baseline_train,
            "val_learned": val_learned, "val_baseline": val_baseline,
            "val_improvement_pct": round(100 * (val_baseline - val_learned) / val_baseline, 2),
            "prior": prior_info, "selection": {"scores": sel, "baseline": evaluate(None, sel_eps), "chosen": chosen},
            "train_seconds": round(time.time() - t0, 1)}
    model = to_torch(theta)
    save(model, resolve(a.out), meta)
    np.save(resolve(a.out).with_suffix(".theta.npy"), theta)
    print(json.dumps({k: meta[k] for k in ("baseline_train", "train_best", "val_baseline", "val_learned",
                                            "val_improvement_pct", "train_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
