"""Fleet dashboard (FR-11): observation only.

Subscribes to the robots' PUB telemetry (RobotState / RobotIntent / BlockageEvent) and the world server's
ground-truth stream; serves a single-page web UI over plain HTTP with Server-Sent Events.  It never sends a single
byte to a robot: robot operation does not depend on it (FR-12, verified by killing it mid-run).

    python -m fleet.dashboard.server --config <resolved_config.json> [--port 8050]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zmq

from fleet.common.config import ports, resolve
from fleet.world.mapgrid import WarehouseMap
from fleet.world.mjcf_builder import ROBOT_COLORS, robot_name

STATIC = Path(__file__).resolve().parent / "static"


class DashboardState:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.world: dict = {}
        self.robots: dict[str, dict] = {}
        self.intents: dict[str, dict] = {}
        self.blockages: dict[int, dict] = {}
        self.rx = {"world": 0, "RobotState": 0, "RobotIntent": 0, "BlockageEvent": 0, "other": 0}
        self.rx_bytes = 0
        self.t0 = time.time()
        m = WarehouseMap.load(resolve(cfg["world"]["map"]), cfg["world"]["cell"])
        self.map = {"rows": m.rows, "cell": m.cell, "H": m.H, "W": m.W,
                    "robots": [{"name": robot_name(i), "color": ROBOT_COLORS[i % len(ROBOT_COLORS)]}
                               for i in range(int(cfg["fleet"]["n_robots"]))]}

    def ingest(self, raw: bytes) -> None:
        msg = json.loads(raw)
        with self.lock:
            self.rx_bytes += len(raw)
            t = msg.get("type")
            if t == "world":
                self.world = msg
                self.rx["world"] += 1
            elif t == "RobotState":
                self.robots[msg["src"]] = msg
                self.rx[t] += 1
            elif t == "RobotIntent":
                self.intents[msg["src"]] = msg
                self.rx[t] += 1
            elif t == "BlockageEvent":
                if msg["blocked"]:
                    self.blockages[msg["node"]] = msg
                else:
                    self.blockages.pop(msg["node"], None)
                self.rx[t] += 1
            else:
                self.rx["other"] += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {"world": self.world, "robots": self.robots, "intents": self.intents,
                    "blockages": list(self.blockages), "rx": dict(self.rx), "rx_bytes": self.rx_bytes,
                    "uptime": round(time.time() - self.t0, 1),
                    "planner": self.cfg["fleet"]["planner"], "learned": bool(self.cfg["fleet"].get("learned_priority")),
                    "mode": self.cfg["run"]["mode"]}


def subscriber(state: DashboardState, stop: threading.Event) -> None:
    cfg = state.cfg
    p = ports(cfg)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://127.0.0.1:{p['world_pub']}")
    for port in p["robot_pub"]:
        sub.connect(f"tcp://127.0.0.1:{port}")
    while not stop.is_set():
        if sub.poll(100):
            try:
                state.ingest(sub.recv(zmq.NOBLOCK))
            except (zmq.Again, ValueError, KeyError):
                pass
    sub.close()


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/map":
                self._send(200, json.dumps(state.map).encode(), "application/json")
            elif self.path == "/state":
                self._send(200, json.dumps(state.snapshot()).encode(), "application/json")
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b"data: " + json.dumps(state.snapshot()).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            else:
                self._send(404, b"not found", "text/plain")
    return Handler


def serve(cfg: dict, port: int, log_path: Path | None = None) -> None:
    state = DashboardState(cfg)
    stop = threading.Event()
    threading.Thread(target=subscriber, args=(state, stop), daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"dashboard: http://127.0.0.1:{port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
            if log_path is not None:      # evidence of what the observer saw (used by the FR-12 test)
                snap = state.snapshot()
                log_path.write_text(json.dumps({"rx": snap["rx"], "rx_bytes": snap["rx_bytes"],
                                                "robots_seen": sorted(snap["robots"]),
                                                "world_t": snap["world"].get("t")}))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        httpd.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    port = a.port or int(cfg["dashboard"]["port"])
    serve(cfg, port, Path(a.config).with_name("dashboard_seen.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
