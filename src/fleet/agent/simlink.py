"""Robot <-> world hardware link: receives this robot's own sensor frame, returns wheel-speed commands.

Control boundary (design doc v3 FR-13, preflight P5): the only runtime input a robot accepts from outside its peer
network is its own sensor frame plus the world clock.  Message types other than sense / stop / kill are rejected, and
any sense field outside the whitelist (e.g. a route, a priority or a task assignment) is stripped and counted, so no
external process - world, orchestrator or dashboard - can inject operational decisions into a running agent.
"""
from __future__ import annotations

import json
from collections import Counter

import zmq

ALLOWED_TYPES = {"sense", "stop", "kill"}
SENSE_KEYS = {"type", "tick", "t", "pose", "vel", "ranges", "battery", "load", "orders", "expect"}
#   pose/vel/ranges/battery/load: own sensors; t/tick: clock; orders: WMS order announcements (not assignments);
#   expect: delivery-barrier message counters (numbers only, never content)


class SimLink:
    def __init__(self, ctx: zmq.Context, name: str, port: int):
        self.sock = ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, name.encode())
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://127.0.0.1:{port}")
        self.name = name
        self.accepted: Counter = Counter()
        self.rejected: Counter = Counter()

    def hello(self, sent: dict | None = None) -> None:
        """`sent`: peer-message counters at the end of the join phase (seeds the delivery barrier for tick 0)."""
        self.sock.send(json.dumps({"type": "hello", "robot": self.name, "sent": sent or {}}).encode())

    def validate(self, msg: dict) -> dict | None:
        if msg.get("type") not in ALLOWED_TYPES:
            self.rejected["type:" + str(msg.get("type"))] += 1
            return None
        if msg["type"] == "sense":
            extra = set(msg) - SENSE_KEYS
            for k in sorted(extra):
                self.rejected["key:" + k] += 1
                del msg[k]
        self.accepted[msg["type"]] += 1
        return msg

    def recv(self, timeout_ms: int = 120000) -> dict | None:
        while self.sock.poll(timeout_ms):
            msg = self.validate(json.loads(self.sock.recv()))
            if msg is not None:
                return msg
        return None

    def stats(self) -> dict:
        return {"accepted": dict(self.accepted), "rejected": dict(self.rejected),
                "link_rejected": sum(self.rejected.values())}

    def send(self, msg: dict) -> None:
        self.sock.send(json.dumps(msg, separators=(",", ":")).encode())

    def close(self) -> None:
        self.sock.close()
