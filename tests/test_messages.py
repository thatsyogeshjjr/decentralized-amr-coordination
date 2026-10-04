import pytest

from fleet.common.messages import SCHEMAS, SchemaError, decode, encode, expired, make


def test_roundtrip_all_types():
    for t, fields in SCHEMAS.items():
        body = {f: 1 for f in fields}
        m = make(t, "amr0", 7, 1.23456, 0.5, **body)
        d = decode(encode(m))
        assert d == m and d["seq"] == 7 and d["ts"] == 1.2346


def test_robot_state_carries_fr02_fields():
    # FR-02: pose, velocity, task, battery, route/intent and timestamp are published
    assert {"pose", "vel", "battery", "task_id", "goal_node", "status"} <= set(SCHEMAS["RobotState"])
    assert {"path", "target_node", "priority", "valid_until"} <= set(SCHEMAS["RobotIntent"])


def test_missing_fields_and_unknown_type_rejected():
    with pytest.raises(SchemaError):
        make("RobotState", "a", 1, 0.0, 1.0, pose=[0, 0, 0])
    with pytest.raises(SchemaError):
        decode(b'{"type":"Nope","src":"a","seq":1,"ts":0,"ttl":1}')
    with pytest.raises(SchemaError):
        decode(b'{"type":"TaskBid","src":"a","seq":1,"ts":0}')
    with pytest.raises(ValueError):
        decode(b"not json")


def test_ttl_expiry():
    m = make("Release", "a", 1, 10.0, 0.5, nodes=[1])
    assert not expired(m, 10.4) and expired(m, 10.6)
    assert not expired(make("Release", "a", 1, 10.0, 0.0, nodes=[]), 1e9)  # ttl 0 = no expiry
