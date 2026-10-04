"""Learned priority advisor (design doc §10.9, SC-08): model structure, NumPy/PyTorch equivalence, checkpoint I/O,
deterministic optimisation and bounded inference latency."""
import time

import numpy as np
import torch

from fleet.learning.features import N_FEATURES
from fleet.learning.model import SCORE_CLIP, NumpyMLP, PriorityMLP, load, n_params, save, to_torch
from fleet.learning.train import cem, episode_set, evaluate


def test_eq7_structure_and_bounded_output():
    m = PriorityMLP()
    assert [type(l).__name__ for l in m.children()] == ["Linear", "Linear", "Linear"]
    x = torch.randn(64, N_FEATURES) * 100
    y = m(x)
    assert y.shape == (64, 1) and float(y.abs().max()) <= SCORE_CLIP


def test_numpy_mirror_matches_torch():
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1.0, n_params())
    tm, nm = to_torch(theta), NumpyMLP(theta)
    for _ in range(20):
        x = rng.normal(0, 1, N_FEATURES).astype(np.float32)
        a = float(tm(torch.from_numpy(x).unsqueeze(0))[0, 0])
        assert abs(a - nm(x.astype(np.float64))) < 1e-4


def test_checkpoint_roundtrip(tmp_path):
    theta = np.random.default_rng(1).normal(0, 0.5, n_params())
    m = to_torch(theta)
    save(m, tmp_path / "p.pt", {"seed": 1})
    m2, meta = load(tmp_path / "p.pt")
    x = torch.randn(5, N_FEATURES)
    assert torch.allclose(m(x), m2(x)) and meta == {"seed": 1}


def test_cem_is_deterministic_and_never_worse_than_its_start():
    eps = episode_set([1000, 1001])
    (b1, th1), h1 = cem(eps, iters=2, pop=6, elite=2, seed=3, log=lambda s: None)
    (b2, th2), h2 = cem(eps, iters=2, pop=6, elite=2, seed=3, log=lambda s: None)
    assert np.array_equal(th1, th2) and h1 == h2
    # the initial mean (all-zero weights -> constant score) is in the first population, so best <= its score
    assert b1 <= evaluate(NumpyMLP(np.zeros(n_params())), eps) + 1e-9


def test_inference_latency_within_edge_budget():
    m = PriorityMLP().eval()
    torch.set_num_threads(1)
    x = torch.randn(1, N_FEATURES)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        t0 = time.perf_counter()
        for _ in range(500):
            m(x)
    per_call_ms = (time.perf_counter() - t0) / 500 * 1000
    assert per_call_ms < 1.0          # SC-08: far below the 50 ms control period on the host


def test_prior_fit_with_pytorch_reproduces_target():
    from fleet.learning.train import collect_features, fit_prior
    X = collect_features(episode_set([1000, 1001]))
    assert X.shape[1] == N_FEATURES and len(X) > 100
    theta = fit_prior(X, lambda x: -40.0 * float(x[0]), seed=0)
    nm = NumpyMLP(theta)
    err = np.array([nm(x.astype(np.float64)) + 40.0 * float(x[0]) for x in X[:300]])
    target_sd = float(np.std(-40.0 * X[:300, 0]))
    assert np.sqrt(np.mean(err ** 2)) < 0.25 * target_sd     # fitted MLP explains the prior (R^2 > ~0.94)
