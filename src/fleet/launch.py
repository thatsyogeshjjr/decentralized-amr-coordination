"""Launch one run: world server (visual MuJoCo, in this process) + one agent process per robot (+ dashboard).

    python -m fleet.launch --scenario configs/scenarios/demo_warehouse.toml [--mode demo|accelerated] [--seed N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fleet.common.config import load_config, resolve
from fleet.experiments.provenance import write_manifest


def _spawn(args: list[str], log: Path) -> subprocess.Popen:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen([sys.executable, "-m", *args], stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)


def run(cfg: dict, scenario: str | None = None) -> dict:
    from fleet.world.sim_server import SimServer

    out = resolve(cfg["run"]["results_dir"]) / cfg["run"]["name"] / f"seed_{cfg['run']['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, cfg, scenario, fresh=True)
    cfg_path = out / "resolved_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    server = SimServer(cfg)          # binds sockets before agents connect
    procs = []
    if cfg["fleet"]["planner"] == "central":     # benchmark-only reference: a central process owns the plan
        procs.append(("central", _spawn(["fleet.benchmark.central", "--config", str(cfg_path)], out / "central.log")))
    if cfg["dashboard"]["enabled"]:
        procs.append(("dashboard", _spawn(["fleet.dashboard.server", "--config", str(cfg_path)], out / "dashboard.log")))
    for i in range(int(cfg["fleet"]["n_robots"])):
        procs.append((f"amr{i}", _spawn(["fleet.agent.agent", "--config", str(cfg_path), "--robot", str(i)], out / f"agent{i}.log")))
    def on_event(ev: dict) -> None:
        if ev["type"] == "kill_dashboard":
            for name, p in procs:
                if name == "dashboard" and p.poll() is None:
                    p.kill()                      # hard kill: observer disappears mid-run (FR-12 fault)

    server.on_event = on_event
    try:
        result = server.run()
    finally:
        deadline = time.time() + 10
        for name, p in procs:
            if name in ("dashboard", "central"):
                p.terminate()
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
    result["agent_exit_codes"] = {name: p.returncode for name, p in procs}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    t = result["tasks"]
    write_manifest(out, cfg, scenario, status="aborted" if result["aborted"] else "finished",
                   end_wall=time.strftime("%Y-%m-%dT%H:%M:%S"), complete=t["n_delivered"] == t["n_tasks"],
                   agent_exit_codes=result["agent_exit_codes"],
                   outputs=sorted(p.name for p in out.iterdir() if p.is_file()))
    return result


def parse_overrides(items: list[str], over: dict | None = None) -> dict:
    """['a.b=1', 'run.name=fourway__pibt'] -> nested dict; JSON values where valid, otherwise plain strings."""
    over = {} if over is None else over
    for s in items:
        key, val = s.split("=", 1)
        d = over
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        try:
            d[parts[-1]] = json.loads(val)        # numbers, booleans, lists, objects
        except json.JSONDecodeError:
            d[parts[-1]] = val                    # plain strings (names, paths)
    return over


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--mode", choices=["demo", "accelerated"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set comms.loss=0.1")
    a = ap.parse_args(argv)
    over: dict = {}
    if a.mode:
        over.setdefault("run", {})["mode"] = a.mode
    if a.seed is not None:
        over.setdefault("run", {})["seed"] = a.seed
    over = parse_overrides(a.set, over)
    cfg = load_config(a.scenario, over)
    res = run(cfg, a.scenario)
    s, t = res["safety"], res["tasks"]
    print(json.dumps({"out": res.get("out_dir"), "sim_time": res["sim_time"], "rtf": res["real_time_factor"],
                      "collisions": s["collisions"], "robot_contacts": s["robot_contacts"],
                      "static_contacts": s["static_contacts"], "min_pair_dist": s["min_pair_dist"],
                      "delivered": f"{t['n_delivered']}/{t['n_tasks']}", "makespan": t["makespan"],
                      "alive": res["alive"], "aborted": res["aborted"]}, indent=1))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # results are already on disk; skip interpreter teardown, where the GLFW viewer thread can crash on Windows
    os._exit(code)
