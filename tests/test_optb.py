"""Option B advisor (design doc v3 §10.9; preflight P1-P3, P9): frozen feature schema, decision-time features,
blocking-impact label, heuristic, bounded priority adjustment, deterministic fallback, split discipline."""
import json
import math

import numpy as np
import pytest
import torch

from fleet.agent.coord import rank
from fleet.agent.deadlock import BOOST
from fleet.agent.optb_advisor import CongestionAdvisor
from fleet.common.config import DEFAULTS, PROJECT_ROOT, deep_merge
from fleet.learning import congestion as C
from fleet.learning import dataset as D
from fleet.learning.optb_model import OptBNet, load, save
from fleet.world.mapgrid import WarehouseMap

M = WarehouseMap.load(PROJECT_ROOT / "configs/maps/warehouse_small.txt")


def test_schema_is_frozen_16_features_in_fixed_order():
    assert C.FEATURES == ["occ_next1", "occ_next2", "occ_next3", "route_len", "n_hop1", "n_hop2", "overlap_mine",
                          "overlap_next", "target_next", "wait", "conflicts", "degree", "dist_goal", "carrying",
                          "age_mean", "age_max"]
    assert C.N_FEATURES == 16 and C.SCHEMA_VERSION == "optb-v1"
    assert C.schema_hash() == C.schema_hash()


# ------------------------------------------------------------------------------------------- fake robot
class Entry:
    def __init__(self, name, node, held, ts, path=None, target=None, its=None):
        self.name = name
        self.state = {"node": node, "held": held, "ts": ts}
        self.intent = None if path is None else {"path": path, "target_node": target, "ts": ts if its is None else its}


class FakeBrain:
    def __init__(self, node, peers, now=10.0, blocked=False):
        self.map, self.node_at, self.now = M, node, now
        self.table = type("T", (), {"known": lambda s: peers})()
        self.last_acquire_t, self.conflict_times = now - 12.0, [now - 1.0, now - 30.0]
        self.goals = type("G", (), {"carrying": lambda s: True})()
        self.iso = type("I", (), {"blocks": lambda s: blocked})()

    def preview_path(self, start, dist, k=4):
        out, u = [], start
        for _ in range(k):
            nb = [v for v in self.map.adj[u] if dist.get(v, 1e9) < dist.get(u, 1e9)]
            if not nb:
                break
            u = min(nb, key=lambda v: (dist[v], v))
            out.append(u)
        return out


def corridor():
    """A node with a straight route ahead (3 nodes) on the small warehouse."""
    goal = M.stations("drop")[0]
    dist = M.bfs_dist(goal)
    here = max((n for n in M.nodes if 4 <= dist.get(n, 0) <= 8), key=lambda n: (len(M.adj[n]), -n))
    return here, dist


def test_features_from_local_view_and_conservative_missing_values():
    here, dist = corridor()
    b = FakeBrain(here, [])
    x = C.features_from_snapshot(C.snapshot(b, dist))
    f = dict(zip(C.FEATURES, x))
    assert x.dtype == np.float32 and x.shape == (16,)
    assert f["occ_next1"] == f["occ_next2"] == f["occ_next3"] == 0.0 and f["route_len"] == 1.0
    assert f["age_mean"] == f["age_max"] == 1.0                        # no peer known: maximal staleness
    assert f["wait"] == pytest.approx(12 / 60) and f["conflicts"] == pytest.approx(1 / 5) and f["carrying"] == 1.0
    route = b.preview_path(here, dist, 3)
    # a *stale* peer (state 2.5 s old) holding my next node still counts as occupied; its old intent is ignored
    stale = Entry("amr1", route[0], [route[0]], ts=b.now - 2.5, path=[here], target=here, its=b.now - 3.5)
    fresh = Entry("amr2", route[2], [route[2]], ts=b.now - 0.1, path=[route[1], route[0], here], target=route[0])
    f = dict(zip(C.FEATURES, C.features_from_snapshot(C.snapshot(FakeBrain(here, [stale, fresh]), dist))))
    assert f["occ_next1"] == 1.0 and f["occ_next2"] == 0.0 and f["occ_next3"] == 1.0
    assert f["overlap_mine"] == pytest.approx(1 / 3) and f["overlap_next"] == pytest.approx(1 / 3)
    assert f["target_next"] == pytest.approx(1 / 2) and f["age_max"] == pytest.approx(2.5 / 3)
    assert C.stale_input(C.snapshot(FakeBrain(here, [stale, fresh]), dist))
    assert not C.stale_input(C.snapshot(FakeBrain(here, [fresh]), dist))


def rows_for(waiting_paths, held_a=(5,), ticks=range(0, 400, 10)):
    rows = {"a": {k: {"held": list(held_a), "waiting": False, "path": []} for k in ticks}}
    for name, path in waiting_paths.items():
        rows[name] = {k: {"held": [99], "waiting": True, "path": path} for k in ticks}
    return rows


def test_blocking_label_definition():
    lab = {(r, k): y for r, k, y in C.blocking_labels(rows_for({"b": [5, 6, 7]}))}
    assert lab[("a", 0)] == pytest.approx(0.5)            # one peer waiting on me for the whole window -> 1/2
    lab = {(r, k): y for r, k, y in C.blocking_labels(rows_for({"b": [5, 6], "c": [4, 5]}))}
    assert lab[("a", 0)] == pytest.approx(1.0)            # two waiting peers sum (cap 2 -> 1.0)
    lab = {(r, k): y for r, k, y in C.blocking_labels(rows_for({"b": [5, 6]}, held_a=(5, 6)))}
    assert lab[("a", 0)] == pytest.approx(0.5)            # one peer counts at most once per tau
    rows = rows_for({"b": [5]})
    del rows["b"][100]                                    # a missing row inside the window -> sample dropped
    kept = {(r, k) for r, k, _ in C.blocking_labels(rows)}
    assert ("a", 0) not in kept and ("a", 100) in kept
    assert max(k for r, k in kept if r == "a") == 390 - 200   # last 10 s of the log cannot be labelled


def test_labels_use_future_but_features_do_not(tmp_path):
    rows = rows_for({"b": [5]})
    for r in rows.values():
        for k, row in r.items():
            row.update({"tick": k, "x": [k / 1000] * 16, "has_goal": True, "goal_since": 0.0})
    for name, rr in rows.items():
        (tmp_path / f"optb_{name}.jsonl").write_text("\n".join(json.dumps(v) for v in rr.values()))
    X1, y1, idx1, *_ = D.run_samples(tmp_path)
    for name, rr in rows.items():                         # append 20 s of future log
        with open(tmp_path / f"optb_{name}.jsonl", "a") as f:
            for k in range(400, 800, 10):
                f.write("\n" + json.dumps({**rr[0], "tick": k, "x": [9.0] * 16}))
    X2, y2, idx2, *_ = D.run_samples(tmp_path)
    common = {i: x for i, x in zip(idx1, X1)}
    assert all(np.array_equal(common[i], x) for i, x in zip(idx2, X2) if i in common)
    assert len(idx2) > len(idx1)


def test_heuristic_is_bounded_and_monotone():
    x = np.zeros(16, np.float32)
    assert C.heuristic_score(x) == 0.0
    i = C.FEATURES.index
    lo, hi = x.copy(), x.copy()
    lo[i("overlap_next")], hi[i("overlap_next")] = 1 / 3, 2 / 3
    assert C.heuristic_score(lo) < C.heuristic_score(hi) <= 1.0
    assert C.heuristic_score(np.ones(16, np.float32)) == 1.0


def test_adjustment_bounds_and_ordering_effect():
    assert C.ADJ_MAX == 20.0
    # larger priority wins; base priority = -goal_since
    me_base, peer_close, peer_far = -100.0, -90.0, -70.0
    assert rank(me_base + C.ADJ_MAX * 1.0, "amr0") > rank(peer_close, "amr1")   # reorders a 10 s gap
    assert rank(me_base + C.ADJ_MAX * 1.0, "amr0") < rank(peer_far, "amr1")     # never a 30 s gap
    assert C.ADJ_MAX < BOOST and -1e9 + C.ADJ_MAX < -1e6                      # never beats boost / idle gap


class StubModel(torch.nn.Module):
    def __init__(self, out=None, boom=False):
        super().__init__()
        self.out, self.boom = out, boom

    def forward(self, x):
        if self.boom:
            raise RuntimeError("inference failure")
        return torch.full((x.shape[0],), self.out)


def advisor(mode, model=None):
    a = CongestionAdvisor(deep_merge(DEFAULTS, {}), "heuristic")
    if mode == "learned":
        a.mode, a.model, a._torch = "learned", model, torch
    return a


def test_advisor_fallbacks_are_deterministic_and_counted():
    here, dist = corridor()
    route = FakeBrain(here, []).preview_path(here, dist, 3)
    fresh = Entry("amr2", route[1], [route[1]], ts=9.9, path=[route[0], here], target=route[0])
    stale = Entry("amr1", route[0], [route[0]], ts=8.0)
    cases = [(advisor("heuristic"), FakeBrain(here, [fresh], blocked=True), "isolated"),
             (advisor("heuristic"), FakeBrain(here, [fresh, stale]), "stale_input"),
             (advisor("learned", StubModel(boom=True)), FakeBrain(here, [fresh]), "exception"),
             (advisor("learned", StubModel(out=float("nan"))), FakeBrain(here, [fresh]), "invalid_output"),
             (advisor("learned", StubModel(out=1.5)), FakeBrain(here, [fresh]), "invalid_output")]
    for a, b, why in cases:
        assert a.score(b, 0, dist) == 0.0 and a.fallbacks == {why: 1}, why
    a = advisor("learned", StubModel(out=0.75))
    assert a.score(FakeBrain(here, [fresh]), 0, dist) == pytest.approx(15.0) and a.fallbacks == {}
    h = advisor("heuristic")
    d = h.score(FakeBrain(here, [fresh]), 0, dist)
    assert 0.0 < d <= C.ADJ_MAX and math.isfinite(d)


def test_split_matrix_and_disjoint_seeds():
    conds = D.conditions()
    assert sum(len(s) for *_, s in conds) == 210
    assert len(conds) == 13 and ("medium8", "test", D.SEEDS["test"]) in conds
    assert not any(f == "medium8" and s != "test" for f, s, _ in conds)
    for a in D.SEEDS:
        for b in D.SEEDS:
            assert a == b or not set(D.SEEDS[a]) & set(D.SEEDS[b])


def test_model_roundtrip_and_schema_guard(tmp_path):
    m = OptBNet([8])
    p = tmp_path / "m.pt"
    save(m, [8], p, {"model_version": "t"})
    m2, meta = load(p)
    assert meta["n_params"] == 16 * 8 + 8 + 8 + 1 and meta["file_bytes"] > 0
    x = torch.rand(4, 16)
    assert torch.allclose(m(x), m2(x)) and float(m2(x).min()) >= 0 and float(m2(x).max()) <= 1
    ck = torch.load(p, weights_only=True)
    ck["meta"]["schema_hash"] = "other"
    torch.save(ck, p)
    with pytest.raises(ValueError):
        load(p)
