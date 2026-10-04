"""Wire format (design doc §10.2, App. A).  The project owns the schema: ZeroMQ only moves bytes.

Every peer message is a JSON object with an envelope
    type, src, seq (monotonic per sender & channel), ts (sender sim time), ttl (seconds of validity)
plus a type-specific body.  Required body fields are validated on decode; unknown types are rejected.
"""
from __future__ import annotations

import json

SCHEMAS: dict[str, tuple[str, ...]] = {
    # App. A logical message contents
    "RobotState": ("pose", "vel", "battery", "task_id", "goal_node", "status", "node", "held"),
    "RobotIntent": ("route_revision", "path", "target_node", "priority", "valid_until"),
    "TaskAnnouncement": ("task_id", "pickup", "dropoff", "deadline", "priority", "status", "revision"),
    "TaskBid": ("task_id", "estimated_cost", "route_cost", "revision"),
    "TaskConsensus": ("task_id", "winner", "revision", "confidence", "cost"),
    "BlockageEvent": ("node", "source", "severity", "blocked", "event_id"),
    # node-claim / PIBT coordination (DEALER/ROUTER)
    "Claim": ("node", "priority", "claim_id"),
    "ClaimReply": ("node", "claim_id", "granted"),
    "Release": ("nodes",),
    "Push": ("node", "priority", "from_node", "chain", "push_id", "path"),
    "PushReply": ("push_id", "ok"),
    "TaskDone": ("task_id", "revision"),
    # fleet startup (CONOPS phase 1): join beacons until every peer's PUB->SUB path is proven
    "Beacon": ("heard",),
    # benchmark-only centralized reference planner
    "CentralPlan": ("step", "assign"),
}
ENVELOPE = ("type", "src", "seq", "ts", "ttl")


class SchemaError(ValueError):
    pass


def make(mtype: str, src: str, seq: int, ts: float, ttl: float, **body) -> dict:
    if mtype not in SCHEMAS:
        raise SchemaError(f"unknown message type {mtype}")
    missing = [k for k in SCHEMAS[mtype] if k not in body]
    if missing:
        raise SchemaError(f"{mtype} missing fields {missing}")
    return {"type": mtype, "src": src, "seq": seq, "ts": round(ts, 4), "ttl": ttl, **body}


def encode(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode()


def decode(raw: bytes) -> dict:
    msg = json.loads(raw)
    if not isinstance(msg, dict):
        raise SchemaError("message is not an object")
    for k in ENVELOPE:
        if k not in msg:
            raise SchemaError(f"envelope missing {k}")
    mtype = msg["type"]
    if mtype not in SCHEMAS:
        raise SchemaError(f"unknown message type {mtype}")
    missing = [k for k in SCHEMAS[mtype] if k not in msg]
    if missing:
        raise SchemaError(f"{mtype} missing fields {missing}")
    return msg


def expired(msg: dict, now: float) -> bool:
    return msg["ttl"] > 0 and now > msg["ts"] + msg["ttl"]
