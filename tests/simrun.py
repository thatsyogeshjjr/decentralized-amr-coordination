"""Helper for integration tests: run a scenario through the real launcher (visual MuJoCo, accelerated)."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_port = itertools.count(7300 + (os.getpid() % 50) * 40, 100)


def run_scenario(scenario: str, tmp: Path, timeout: float = 900, **sets) -> dict:
    sets.setdefault("run.port_base", next(_port))
    sets.setdefault("run.results_dir", str(tmp).replace("\\", "/"))
    sets.setdefault("run.mode", "accelerated")
    args = [sys.executable, "-m", "fleet.launch", "--scenario", str(ROOT / scenario)]
    for k, v in sets.items():
        args += ["--set", f"{k}={json.dumps(v) if not isinstance(v, str) else v}"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    p = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    from fleet.common.config import load_config
    cfg = load_config(ROOT / scenario)
    name = sets.get("run.name", cfg["run"]["name"])
    seed = sets.get("run.seed", cfg["run"]["seed"])
    out = Path(sets["run.results_dir"]) / name / f"seed_{seed}"
    return json.loads((out / "result.json").read_text())
