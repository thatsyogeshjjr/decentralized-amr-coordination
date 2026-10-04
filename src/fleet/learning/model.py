"""Edge-AI priority model (design doc §10.9, eq. 7):

    h1 = sigma(W1 x + b1),  h2 = sigma(W2 h1 + b2),  p = W3 h2 + b3

PyTorch module for deployment/inference on the robot, plus an exact NumPy mirror used for fast policy-search
rollouts.  The output p (seconds-equivalent) is *added* to the PIBT priority key; it never commands motion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fleet.learning.features import N_FEATURES

HIDDEN = 16
SCORE_CLIP = 150.0    # bounded influence (s-equivalent): can reorder robots, can never dominate deadlock boosts (1e6)


class PriorityMLP(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, hidden: int = HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(n_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.sigmoid(self.l1(x))
        h2 = torch.sigmoid(self.l2(h1))
        return torch.clamp(self.l3(h2), -SCORE_CLIP, SCORE_CLIP)


def n_params(hidden: int = HIDDEN, n_in: int = N_FEATURES) -> int:
    return n_in * hidden + hidden + hidden * hidden + hidden + hidden + 1


def unflatten(theta: np.ndarray, hidden: int = HIDDEN, n_in: int = N_FEATURES):
    i = 0
    out = []
    for shape in [(hidden, n_in), (hidden,), (hidden, hidden), (hidden,), (1, hidden), (1,)]:
        k = int(np.prod(shape))
        out.append(theta[i:i + k].reshape(shape))
        i += k
    return out


class NumpyMLP:
    """Bit-for-bit structure of PriorityMLP evaluated with NumPy (fast for single-sample rollouts)."""

    def __init__(self, theta: np.ndarray):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = unflatten(np.asarray(theta, dtype=np.float64))

    def __call__(self, x: np.ndarray) -> float:
        h1 = 1.0 / (1.0 + np.exp(-(self.W1 @ x + self.b1)))
        h2 = 1.0 / (1.0 + np.exp(-(self.W2 @ h1 + self.b2)))
        return float(np.clip((self.W3 @ h2 + self.b3)[0], -SCORE_CLIP, SCORE_CLIP))


def to_torch(theta: np.ndarray) -> PriorityMLP:
    m = PriorityMLP()
    W1, b1, W2, b2, W3, b3 = unflatten(np.asarray(theta, dtype=np.float32))
    with torch.no_grad():
        m.l1.weight.copy_(torch.from_numpy(W1)); m.l1.bias.copy_(torch.from_numpy(b1))
        m.l2.weight.copy_(torch.from_numpy(W2)); m.l2.bias.copy_(torch.from_numpy(b2))
        m.l3.weight.copy_(torch.from_numpy(W3)); m.l3.bias.copy_(torch.from_numpy(b3))
    return m.eval()


def save(model: PriorityMLP, path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": HIDDEN, "n_in": N_FEATURES, "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load(path: str | Path) -> tuple[PriorityMLP, dict]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    m = PriorityMLP(ck["n_in"], ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["meta"]
