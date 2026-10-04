"""Scenario configuration: TOML files deep-merged over defaults.  Every run stores its resolved config."""
from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULTS: dict = {
    "run": {
        "name": "run",
        "seed": 0,
        "mode": "accelerated",      # "accelerated" (data/experiments) | "demo" (1x wall clock, pitches only)
        "duration": 300.0,          # max sim seconds
        "control_hz": 20,
        "state_hz": 10,             # PR-03: 5-10 Hz state broadcast
        "port_base": 7100,
        "results_dir": "results",
        "viewer_fps": 30,
        "barrier_timeout": 0.25,    # wall seconds to wait for in-flight peer messages per tick
        "cmd_timeout": 5.0,         # wall seconds before a silent agent is declared dead
        "stop_when_done": True,
    },
    "world": {
        "map": "configs/maps/warehouse_small.txt",
        "cell": 1.5,
        "pose_noise": 0.0,          # std-dev metres of the localisation estimate
        "yaw_noise": 0.0,
        "n_blocks": 4,
        "d_safe": 0.9,              # eq. 8 safety distance between chassis centres
    },
    "fleet": {
        "n_robots": 3,
        "start_yaw": 0.0,
        "planner": "pibt",          # pibt | stop_wait | central (benchmark only)
        "learned_priority": False,         # v2 historical eq.-7 MLP advisor (kept unchanged)
        "model_path": "models/priority_mlp.pt",
        "priority_advisor": "none",        # v3 Option B: none (A) | heuristic (B) | learned (C)
        "optb_model_path": "models/optb_advisor.pt",
        "log_features": False,             # write optb_<robot>.jsonl decision-time feature logs (dataset generation)
    },
    "motion": {
        "v_max": 1.0,
        "w_max": 1.2,
        "a_max": 0.6,
        "alpha_max": 2.0,
        "pos_tol": 0.08,
        "heading_tol": 0.30,
    },
    "safety": {
        "stop_margin": 0.20,        # m of free space kept in front of the chassis
        "brake_decel": 0.8,
        "peer_radius": 1.15,        # chassis-centre distance below which motion toward a peer is vetoed
        "turn_clear": 0.03,
    },
    "battery": {
        "capacity_wh": 1200.0,
        "idle_w": 60.0,
        "drain_scale": 1.0,
        "charge_w": 600.0,
        "low": 0.25,
        "full": 0.90,
        "initial": [1.0],
    },
    "comms": {
        "latency": 0.0,             # one-way delay seconds (sim time)
        "jitter": 0.0,
        "loss": 0.0,                # receive-side drop probability
        "heartbeat_timeout": 1.0,   # sim seconds without state -> peer stale
        "lost_timeout": 3.0,        # sim seconds without state -> peer lost
        # v3 FR-14 impairments (all off by default; v2 behaviour unchanged)
        "burst": {},                # Gilbert-Elliott: {"p_enter": .., "p_exit": .., "loss_bad": 1.0}
        "disconnect": [],           # [{"robot": i, "t": t0, "duration": d}] (sim seconds, both directions)
        "reorder_pub": False,       # allow jitter to reorder the PUB channel (DM always FIFO)
        "trace": False,             # per-message CSV trace per agent (comms_<robot>.csv)
        "isolation_timeout": 30.0,  # isolated (all peers non-fresh) this long -> persistent partition
    },
    "tasks": {
        "count": 0,
        "rate": 0.1,                # tasks / second (Poisson arrivals)
        "dwell": 2.0,               # load / unload seconds
        "bid_window": 0.3,
        "w_dist": 1.0, "w_time": 0.5, "w_cong": 2.0, "w_batt": 5.0, "w_risk": 3.0,
        "list": [],                 # explicit [[release_t, pickup_node_rc, drop_node_rc], ...]
    },
    "planning": {
        "deadlock_window": 6.0,     # T_d
        "deadlock_eps": 0.15,       # eps_p
        "blockage_ttl": 60.0,
        "stopwait_timeout": 12.0,
        "replan_cost_factor": 3.0,
    },
    "faults": {"events": [], "random_blockages": {"rate": 0.0, "slots": 10, "start": 20.0, "spacing": 25.0,
                                                  "duration": 40.0}},
    "dashboard": {"enabled": False, "port": 8050},
    "waypoints": {},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        with open(p, "rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def ports(cfg: dict) -> dict:
    """Static peer endpoint model (design doc App. A): configuration data, not a discovery service."""
    b = int(cfg["run"]["port_base"])
    n = int(cfg["fleet"]["n_robots"])
    return {
        "sim": b,
        "world_pub": b + 2,
        "central_pub": b + 3,          # benchmark-only centralized reference planner
        "robot_pub": [b + 10 + i for i in range(n)],
        "robot_router": [b + 40 + i for i in range(n)],
    }
