"""Run provenance for the experiment orchestrator (design doc v3 FR-13, §16.2; preflight P5).

The orchestrator configures and records experiments only: it never routes robots, assigns tasks, sets priorities or
resolves conflicts.  A manifest per run records what was run, with which code / model, under which fault schedule,
and how the run ended (including failed and incomplete runs).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from fleet.common.config import PROJECT_ROOT, config_hash, resolve

_CODE: str | None = None
_RUNTIME: str | None = None
# files that never execute inside a run (orchestration, analysis, offline training): excluded from the runtime hash
NON_RUNTIME = ("fleet/experiments/", "fleet/learning/dataset.py", "fleet/learning/train.py", "fleet/learning/train_optb.py")


def code_version() -> str:
    """sha1 over every source file under src/ (path + content).  The repository has no VCS, so this is the code id."""
    global _CODE
    if _CODE is None:
        h = hashlib.sha1()
        src = PROJECT_ROOT / "src"
        for p in sorted(src.rglob("*.py")):
            h.update(p.relative_to(src).as_posix().encode())
            h.update(p.read_bytes())
        _CODE = h.hexdigest()[:12]
    return _CODE


def runtime_code_version() -> str:
    """sha1 over the source files that execute during a run (world, agents, comms, dashboard, benchmark, models)."""
    global _RUNTIME
    if _RUNTIME is None:
        h = hashlib.sha1()
        src = PROJECT_ROOT / "src"
        for p in sorted(src.rglob("*.py")):
            rel = p.relative_to(src).as_posix()
            if rel.startswith(NON_RUNTIME):
                continue
            h.update(rel.encode())
            h.update(p.read_bytes())
        _RUNTIME = h.hexdigest()[:12]
    return _RUNTIME


def file_sha1(p: Path) -> str | None:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def model_version(cfg: dict) -> dict | None:
    f = cfg["fleet"]
    paths = []
    if f.get("learned_priority"):
        paths.append(("historical_mlp", f["model_path"]))
    if f.get("priority_advisor") == "learned":
        paths.append(("optb", f.get("optb_model_path", "models/optb_advisor.pt")))
    if not paths:
        return None
    out = {}
    for kind, rel in paths:
        p = resolve(rel)
        meta = p.with_suffix(".json")
        out[kind] = {"path": rel, "sha1": file_sha1(p),
                     "meta_version": json.loads(meta.read_text()).get("model_version") if meta.exists() else None}
    return out


def fault_schedule(cfg: dict) -> dict:
    c = cfg["comms"]
    return {"events": cfg["faults"].get("events", []), "random_blockages": cfg["faults"].get("random_blockages", {}),
            "bypass_claims": bool(cfg["faults"].get("bypass_claims", False)),
            "comms": {k: c.get(k) for k in ("latency", "jitter", "loss", "burst", "disconnect", "reorder_pub")}}


def write_manifest(out: Path, cfg: dict, scenario: str | None, fresh: bool = False, **status) -> dict:
    """fresh=True starts a new record for a new launch but keeps the orchestrator's earlier attempt log."""
    p = out / "manifest.json"
    old = json.loads(p.read_text()) if p.exists() else None
    if fresh and old is not None:
        status = {"attempts": old.get("attempts", []), **status}
        old = None
    m = old if old is not None else {
        "run": cfg["run"]["name"], "seed": cfg["run"]["seed"], "scenario": scenario,
        "config_hash": config_hash(cfg), "code_version": code_version(), "runtime_code_version": runtime_code_version(),
        "model_version": model_version(cfg),
        "fault_schedule": fault_schedule(cfg), "mode": cfg["run"]["mode"], "planner": cfg["fleet"]["planner"],
        "priority_advisor": cfg["fleet"].get("priority_advisor", "none"),
        "learned_priority": bool(cfg["fleet"].get("learned_priority")), "n_robots": cfg["fleet"]["n_robots"],
        "map": cfg["world"]["map"], "start_wall": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "started"}
    m.update(status)
    p.write_text(json.dumps(m, indent=2))
    return m
