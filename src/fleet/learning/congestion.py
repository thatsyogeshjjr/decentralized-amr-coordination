"""Option B congestion-aware priority advisor: feature schema, handcrafted heuristic and label (design doc v3 §10.9;
frozen in docs/internal_updates/option_b_spec.md, schema optb-v1).

Every feature is computed from the robot's own state and its peer table at decision time (`snapshot`), by the one
function `features_from_snapshot`, which the dataset logger, the trainer, validation and runtime inference all use.
Future observations are used only by `blocking_labels` (offline label generation).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SCHEMA_VERSION = "optb-v1"
FEATURES = ["occ_next1", "occ_next2", "occ_next3", "route_len", "n_hop1", "n_hop2", "overlap_mine", "overlap_next",
            "target_next", "wait", "conflicts", "degree", "dist_goal", "carrying", "age_mean", "age_max"]
N_FEATURES = len(FEATURES)                     # 16
SCALE = {"route_len": 3.0, "n_hop1": 4.0, "n_hop2": 8.0, "overlap_mine": 3.0, "overlap_next": 3.0,
         "target_next": 2.0, "wait": 60.0, "conflicts": 5.0, "degree": 4.0, "dist_goal": 40.0, "age_mean": 3.0,
         "age_max": 3.0}
INTENT_MAX_AGE = 3.0                           # s: older intents are ignored for overlap
FRESH_AGE = 1.0                                # s: = comms.heartbeat_timeout; stale-input fallback threshold
CONFLICT_WINDOW = 10.0                         # s
ADJ_MAX = 20.0                                 # s-equivalent priority adjustment (larger priority wins)
HEURISTIC_W = {"overlap_next": 0.4, "overlap_mine": 0.3, "occ_next1": 0.2, "conflicts": 0.1}
LABEL_H = 10.0                                 # s look-ahead of the blocking-impact label
LABEL_DT = 0.5                                 # s log period (every 10 ticks at 20 Hz)
LABEL_CAP = 2.0                                # peers waiting on me, saturating


def schema_hash() -> str:
    return hashlib.sha1(json.dumps({"v": SCHEMA_VERSION, "f": FEATURES, "s": SCALE}, sort_keys=True).encode()).hexdigest()[:12]


def snapshot(brain, dist: dict | None) -> dict:
    """Decision-time local view (own state + peer table).  Stale / lost peers keep their held nodes occupied."""
    m, now, here = brain.map, brain.now, brain.node_at
    route = brain.preview_path(here, dist, 3) if dist is not None else []
    peers = []
    for e in brain.table.known():
        st, it = e.state, e.intent
        peers.append({"node": st.get("node"), "held": list(st.get("held", [])), "age": now - float(st["ts"]),
                      "path": list(it.get("path", []))[:4] if it and now - float(it["ts"]) <= INTENT_MAX_AGE else [],
                      "target": it.get("target_node") if it and now - float(it["ts"]) <= INTENT_MAX_AGE else None})
    return {"node": here, "route": route, "adj1": sorted(m.adj[here]),
            "adj2": sorted(({w for u in m.adj[here] for w in m.adj[u]} | set(m.adj[here])) - {here}),
            "peers": peers, "has_goal": dist is not None,
            "wait": (now - brain.last_acquire_t) if dist is not None else 0.0,
            "conflicts": sum(1 for t in brain.conflict_times if now - t <= CONFLICT_WINDOW),
            "degree": len(m.adj[here]), "dist_goal": float(dist.get(here, 60)) if dist is not None else 0.0,
            "carrying": bool(brain.goals.carrying())}


def features_from_snapshot(s: dict) -> np.ndarray:
    occ = set()
    for p in s["peers"]:
        occ.update(p["held"])
    route = s["route"]
    occ_k = [1.0 if k < len(route) and route[k] in occ else 0.0 for k in range(3)]
    a1, a2 = set(s["adj1"]), set(s["adj2"])
    nodes = [p["node"] for p in s["peers"] if p["node"] is not None]
    nxt = route[0] if route else None
    raw = {
        "route_len": len(route), "n_hop1": sum(n in a1 for n in nodes), "n_hop2": sum(n in a2 for n in nodes),
        "overlap_mine": sum(s["node"] in p["path"] for p in s["peers"]),
        "overlap_next": sum(nxt is not None and nxt in p["path"] for p in s["peers"]),
        "target_next": sum(nxt is not None and p["target"] == nxt for p in s["peers"]),
        "wait": min(s["wait"], 60.0), "conflicts": s["conflicts"], "degree": s["degree"],
        "dist_goal": min(s["dist_goal"], 40.0),
        "age_mean": min(float(np.mean([p["age"] for p in s["peers"]])), 3.0) if s["peers"] else 3.0,
        "age_max": min(max(p["age"] for p in s["peers"]), 3.0) if s["peers"] else 3.0,
    }
    x = np.empty(N_FEATURES, dtype=np.float32)
    for i, name in enumerate(FEATURES):
        if name.startswith("occ_next"):
            x[i] = occ_k[int(name[-1]) - 1]
        elif name == "carrying":
            x[i] = float(s["carrying"])
        else:
            x[i] = raw[name] / SCALE[name]
    return np.clip(x, 0.0, 1.0)


def stale_input(s: dict) -> bool:
    """A peer within two hops (by its last known node) has state older than the freshness threshold."""
    near = set(s["adj2"]) | {s["node"]}
    return any(p["node"] in near and p["age"] > FRESH_AGE for p in s["peers"])


def heuristic_score(x: np.ndarray) -> float:
    i = FEATURES.index
    v = sum(w * float(x[i(k)]) for k, w in HEURISTIC_W.items())
    return float(min(1.0, max(0.0, v)))


# ------------------------------------------------------------------------------------------------ labels (offline)
def blocking_labels(rows: dict[str, dict[int, dict]], ticks_per_sample: int = 10) -> list[tuple[str, int, float]]:
    """rows[robot][tick] = {"held": [...], "waiting": bool, "path": [...]} sampled every `ticks_per_sample` ticks.

    y_i(t) = min(2, (dt/H) * sum_{tau in (t, t+H]} sum_{j != i} w_j(tau) * [path_j(tau)[:4] & held_i(tau) != {}]) / 2
    A sample is dropped (not imputed) unless every robot that has a row at t has rows at every tau in the window.
    Returns [(robot, tick, y)] for kept samples; the caller counts dropped ones."""
    n_ahead = int(round(LABEL_H / LABEL_DT))
    out = []
    robots = sorted(rows)
    for i in robots:
        for k in sorted(rows[i]):
            present = [j for j in robots if k in rows[j]]
            taus = [k + ticks_per_sample * q for q in range(1, n_ahead + 1)]
            if not all(tau in rows[j] for j in present for tau in taus):
                continue
            acc = 0.0
            for tau in taus:
                held_i = set(rows[i][tau]["held"])
                for j in present:
                    if j == i:
                        continue
                    rj = rows[j][tau]
                    if rj["waiting"] and held_i.intersection(rj["path"][:4]):
                        acc += 1.0                      # robot j counted at most once per tau
            y = min(LABEL_CAP, acc * LABEL_DT / LABEL_H) / LABEL_CAP
            out.append((i, k, y))
    return out
