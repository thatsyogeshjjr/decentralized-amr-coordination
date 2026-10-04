"""Dashboard (FR-11) serves live fleet state; robot operation continues when it is killed (FR-12, SC-07)."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import zmq

from fleet.common.config import DEFAULTS, deep_merge, ports
from fleet.common.messages import encode, make
from simrun import ROOT, run_scenario


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get(url: str, timeout=5.0) -> bytes:
    return urllib.request.urlopen(url, timeout=timeout).read()


def test_dashboard_serves_state_from_robot_telemetry(tmp_path):
    base = 9500 + (os.getpid() % 30) * 20
    http = free_port()
    cfg = deep_merge(DEFAULTS, {"run": {"port_base": base}, "dashboard": {"port": http}})
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    ctx = zmq.Context.instance()
    pubs = []
    for port in ports(cfg)["robot_pub"]:          # stand-in robots publishing real wire-format telemetry
        s = ctx.socket(zmq.PUB)
        s.bind(f"tcp://127.0.0.1:{port}")
        pubs.append(s)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.Popen([sys.executable, "-m", "fleet.dashboard.server", "--config", str(cfg_path)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 20
        snap = None
        while time.time() < deadline:
            for i, s in enumerate(pubs):
                s.send(encode(make("RobotState", f"amr{i}", 1, 1.0, 1.0, pose=[1.5 * i, 6.0, 0.0], vel=[0, 0],
                                   battery=0.8, task_id="T1" if i == 0 else None, goal_node=None, status="moving",
                                   node=54 + i, held=[54 + i])))
            try:
                snap = json.loads(get(f"http://127.0.0.1:{http}/state"))
                if len(snap["robots"]) == 3:
                    break
            except OSError:
                pass
            time.sleep(0.2)
        assert snap and sorted(snap["robots"]) == ["amr0", "amr1", "amr2"]
        assert snap["robots"]["amr0"]["task_id"] == "T1" and snap["robots"]["amr1"]["battery"] == 0.8   # FR-11 fields
        m = json.loads(get(f"http://127.0.0.1:{http}/map"))
        assert m["W"] == 13 and len(m["robots"]) == 3
        assert b"<canvas" in get(f"http://127.0.0.1:{http}/")
        with urllib.request.urlopen(f"http://127.0.0.1:{http}/events", timeout=5) as r:
            line = r.readline()
            assert line.startswith(b"data: ") and b"amr0" in line
    finally:
        proc.kill()
        for s in pubs:
            s.close(0)


@pytest.mark.sim
def test_fleet_continues_after_dashboard_is_killed(tmp_path):
    r = run_scenario("configs/scenarios/m6_dashboard.toml", tmp_path, **{"dashboard.port": free_port()})
    t = r["tasks"]
    assert t["n_delivered"] == t["n_tasks"]          # coordination unaffected by losing the observer
    assert r["safety"]["collisions"] == 0
    assert any(e.get("type") == "kill_dashboard" for e in r["events"])
    assert t["last_delivery"] > 30.0                   # deliveries happened after the dashboard died at t = 20 s
    seen = json.loads((tmp_path / "m6_dashboard" / "seed_4" / "dashboard_seen.json").read_text())
    assert seen["robots_seen"] == ["amr0", "amr1", "amr2"] and seen["rx"]["RobotState"] > 0
    assert seen["world_t"] is not None and seen["world_t"] <= 21.0     # it stopped observing when killed
    assert r["agent_exit_codes"]["dashboard"] != 0
    assert all(code == 0 for name, code in r["agent_exit_codes"].items() if name != "dashboard")
