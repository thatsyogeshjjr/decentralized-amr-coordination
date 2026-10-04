"""Option B congestion-aware priority advisor, one instance inside each robot agent (design doc v3 §10.9, eq. 7).

    local observations -> risk estimate s in [0, 1] -> bounded adjustment +ADJ_MAX*s -> PIBT -> safety shield

Larger priority wins (coord.rank).  The adjustment (<= 20 s-equivalent) can only reorder robots whose goal
assignment times differ by less than ADJ_MAX; it never reaches the deadlock boost (1e6), never lifts an idle robot
(-1e9) above one with a goal, and never touches wheel commands.  Any failure falls back to the deterministic priority
(adjustment 0) and is counted by reason: isolated, stale_input, exception, invalid_output.
"""
from __future__ import annotations

import math
import time

import numpy as np

from fleet.common.config import resolve
from fleet.learning.congestion import ADJ_MAX, features_from_snapshot, heuristic_score, snapshot, stale_input


class CongestionAdvisor:
    def __init__(self, cfg: dict, mode: str):
        if mode not in ("heuristic", "learned"):
            raise ValueError(f"priority_advisor must be none|heuristic|learned, got {mode}")
        self.mode = mode
        self.model, self.meta = None, {}
        if mode == "learned":
            import torch
            from fleet.learning.optb_model import load
            torch.set_num_threads(1)
            self.model, self.meta = load(resolve(cfg["fleet"].get("optb_model_path", "models/optb_advisor.pt")))
            self._torch = torch
        self.calls = 0
        self.fallbacks: dict[str, int] = {}
        self.samples: list[float] = []
        self.deltas: list[float] = []
        self._cache: tuple | None = None

    def _fallback(self, why: str) -> float:
        self.fallbacks[why] = self.fallbacks.get(why, 0) + 1
        return 0.0

    def infer(self, x: np.ndarray) -> float:
        if self.mode == "heuristic":
            return heuristic_score(x)
        t0 = time.perf_counter()
        with self._torch.no_grad():
            s = float(self.model(self._torch.from_numpy(x).unsqueeze(0))[0])
        self.samples.append(time.perf_counter() - t0)
        return s

    def score(self, brain, goal: int, dist: dict) -> float:
        key = (goal, brain.node_at, int(brain.now * 2))          # re-evaluated at 2 Hz, like the v2 advisor
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self.calls += 1
        if brain.iso.blocks():
            d = self._fallback("isolated")
        else:
            snap = snapshot(brain, dist)
            if stale_input(snap):
                d = self._fallback("stale_input")
            else:
                try:
                    s = self.infer(features_from_snapshot(snap))
                except Exception:                          # noqa: BLE001 - any inference failure -> deterministic
                    s = None
                if s is None:
                    d = self._fallback("exception")
                elif not math.isfinite(s) or s < -0.05 or s > 1.05:
                    d = self._fallback("invalid_output")
                else:
                    d = ADJ_MAX * min(1.0, max(0.0, s))
        self.deltas.append(d)
        self._cache = (key, d)
        return d

    def stats(self) -> dict:
        a = np.array(self.samples) * 1000 if self.samples else None
        return {"mode": self.mode, "calls": self.calls, "fallbacks": dict(self.fallbacks),
                "fallback_frac": round(sum(self.fallbacks.values()) / max(1, self.calls), 4),
                "delta_mean": round(float(np.mean(self.deltas)), 4) if self.deltas else None,
                "delta_nonzero_frac": round(float(np.mean(np.array(self.deltas) > 0)), 4) if self.deltas else None,
                "infer_p50_ms": round(float(np.percentile(a, 50)), 4) if a is not None else None,
                "infer_p99_ms": round(float(np.percentile(a, 99)), 4) if a is not None else None,
                "model_version": self.meta.get("model_version"), "n_params": self.meta.get("n_params"),
                "file_bytes": self.meta.get("file_bytes")}
