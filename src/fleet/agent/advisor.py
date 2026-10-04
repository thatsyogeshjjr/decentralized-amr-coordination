"""Learning advisor (design doc Fig. 3/4): learned congestion-aware priority score added to the PIBT priority key.

Learning -> priority estimate -> deterministic coordination -> safety -> actuation (eq. 2).  The score is bounded
(model.SCORE_CLIP), cannot override deadlock boosts, and never touches wheel commands.  Inference time is measured
for the edge compute budget (SC-08).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from fleet.common.config import resolve
from fleet.learning.features import feature_vector, local_counts
from fleet.learning.model import load


class LearnedPriorityAdvisor:
    def __init__(self, cfg: dict):
        self.model, self.meta = load(resolve(cfg["fleet"]["model_path"]))
        torch.set_num_threads(1)
        self.samples: list[float] = []
        self.calls = 0
        self._cache: tuple | None = None

    def features(self, brain, goal: int, dist: dict) -> np.ndarray:
        others = {nd for _, (nd, _) in brain.peer_nodes().items() if nd is not None}
        route = brain.preview_path(brain.node_at, dist, 3)
        n1, n2, onr, deg, free = local_counts(brain.map.adj, brain.node_at, others, route)
        return feature_vector(float(dist.get(brain.node_at, 60)), brain.now - brain.goal_since, brain.goals.carrying(),
                              n1, n2, onr, deg, free)

    def score(self, brain, goal: int, dist: dict) -> float:
        # re-evaluate at 2 Hz: the score is advisory and published with state, no need to run every 50 ms tick
        key = (goal, brain.node_at, int(brain.now * 2))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        x = self.features(brain, goal, dist)
        t0 = time.perf_counter()
        with torch.no_grad():
            s = float(self.model(torch.from_numpy(x).unsqueeze(0))[0, 0])
        self.samples.append(time.perf_counter() - t0)
        self.calls += 1
        self._cache = (key, s)
        return s

    def stats(self) -> dict:
        a = np.array(self.samples) * 1000 if self.samples else np.zeros(1)
        return {"calls": self.calls, "infer_mean_ms": round(float(a.mean()), 4), "infer_p99_ms": round(float(np.percentile(a, 99)), 4),
                "model_val_improvement_pct": self.meta.get("val_improvement_pct")}
