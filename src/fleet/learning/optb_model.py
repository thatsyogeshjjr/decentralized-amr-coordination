"""Option B advisor model: a compact feed-forward regressor of the blocking-impact label (design doc v3 §10.9).

Inputs: the 16 optb-v1 features standardized with train-set statistics.  Output: sigmoid -> s in [0, 1], the
predicted (normalized) number of peers that will be waiting on this robot over the next 10 s.  Hidden sizes are a
model-selection choice (hidden=[] is the linear baseline); the topology is not fixed in advance.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fleet.learning.congestion import N_FEATURES, schema_hash


class OptBNet(nn.Module):
    def __init__(self, hidden: list[int], n_in: int = N_FEATURES):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net((x - self.mu) / self.sd)).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def serialized_bytes(m: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.getbuffer().nbytes


def save(m: OptBNet, hidden: list[int], path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "schema_hash": schema_hash(), "hidden": hidden, "n_params": n_params(m)}
    torch.save({"state_dict": m.state_dict(), "hidden": hidden, "meta": meta}, path)
    meta["file_bytes"] = path.stat().st_size
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load(path: str | Path) -> tuple[OptBNet, dict]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ck["meta"].get("schema_hash") != schema_hash():
        raise ValueError(f"feature schema mismatch: model {ck['meta'].get('schema_hash')} != code {schema_hash()}")
    m = OptBNet(list(ck["hidden"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), {**ck["meta"], "file_bytes": Path(path).stat().st_size}


def predict(m: OptBNet, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return m(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()
