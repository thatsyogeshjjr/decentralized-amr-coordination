"""Peer messaging over real ZeroMQ TCP sockets (no simulator involved)."""
import itertools
import multiprocessing as mp
import os
import time

import numpy as np
import pytest
import zmq

from fleet.comms.fault_injector import FaultInjector
from fleet.comms.peer_node import PeerNode
from fleet.comms.peer_table import FRESH, LOST, STALE, PeerTable

NAMES = ["amr0", "amr1", "amr2"]
_ports = itertools.count(9000 + (os.getpid() % 40) * 50, 10)


def make_nodes(n=3, **kw):
    b = next(_ports)
    ctx = zmq.Context.instance()
    names = NAMES[:n]
    pub = [b + i for i in range(n)]
    rtr = [b + 5 + i for i in range(n)]
    nodes = [PeerNode(ctx, nm, names, pub, rtr, **kw) for nm in names]
    t0 = time.time()
    while not all([nd.join_step(10) for nd in nodes]):   # startup handshake, as agents do before the clock starts
        assert time.time() - t0 < 10, "join handshake did not complete"
    for nd in nodes:  # drain beacons still in flight so tests start clean
        nd.join_step(50)
    return nodes


def state_body(i):
    return dict(pose=[i, 0, 0], vel=[0, 0], battery=1.0, task_id=None, goal_node=None, status="idle", node=i, held=[i])


def test_pub_sub_and_dealer_router_delivery():
    nodes = make_nodes()
    try:
        for i, nd in enumerate(nodes):
            nd.publish("RobotState", 0.0, 1.0, **state_body(i))
        nodes[0].send("amr2", "Claim", 0.0, 1.0, node=5, priority=0.7, claim_id="c1")
        exp = {nd.name: nd.sent_counters() for nd in nodes}
        got = {}
        for nd in nodes:
            got[nd.name] = nd.sync(0.05, {k: v for k, v in exp.items() if k != nd.name}, timeout=2.0)
        for nd in nodes:
            srcs = sorted(m["src"] for m in got[nd.name] if m["type"] == "RobotState")
            assert srcs == sorted(p for p in NAMES if p != nd.name)
            assert nd.barrier_timeouts == 0
        claims = [m for m in got["amr2"] if m["type"] == "Claim"]
        assert len(claims) == 1 and claims[0]["src"] == "amr0" and claims[0]["node"] == 5
        assert not any(m["type"] == "Claim" for m in got["amr1"])
    finally:
        for nd in nodes:
            nd.close()


def test_delivery_order_is_deterministic():
    nodes = make_nodes()
    try:
        for k in range(5):
            for i in (2, 1):
                nodes[i].send("amr0", "Release", 0.0, 1.0, nodes=[k])
        exp = {nd.name: nd.sent_counters() for nd in nodes[1:]}
        got = nodes[0].sync(0.05, exp, timeout=2.0)
        keys = [(m["src"], m["seq"]) for m in got]
        assert keys == sorted(keys)
    finally:
        for nd in nodes:
            nd.close()


def test_latency_in_sim_time_and_loss_rate():
    fi = FaultInjector(latency=0.1, loss=0.0)
    fi.offer("pub", {"ts": 1.0, "seq": 1})
    assert fi.deliver(1.05) == []
    assert len(fi.deliver(1.1)) == 1
    fi = FaultInjector(loss=0.2, seed=3)
    for k in range(5000):
        fi.offer("dm", {"ts": 0.0, "seq": k})
    rate = fi.dropped / fi.received
    assert 0.18 < rate < 0.22
    # same seed -> identical drop pattern (reproducibility)
    a, b = FaultInjector(loss=0.3, seed=9), FaultInjector(loss=0.3, seed=9)
    for k in range(200):
        a.offer("pub", {"ts": 0.0, "seq": k})
        b.offer("pub", {"ts": 0.0, "seq": k})
    assert [m["seq"] for m in a.deliver(1)] == [m["seq"] for m in b.deliver(1)]


def test_end_to_end_loss_over_sockets():
    nodes = make_nodes(2, loss=0.25, seed=5)
    try:
        for k in range(800):
            nodes[1].publish("RobotState", 0.0, 0.0, **state_body(1))
        got = nodes[0].sync(0.0, {"amr1": nodes[1].sent_counters()}, timeout=3.0)
        states = [m for m in got if m["type"] == "RobotState"]
        frac = 1 - len(states) / 800
        assert 0.20 < frac < 0.30
        st = nodes[0].stats()
        assert st["rx_offered"] - st["rx_dropped"] == len(got) + nodes[0].fault.in_flight
    finally:
        for nd in nodes:
            nd.close()


def test_peer_table_freshness_and_ordering():
    pt = PeerTable(["amr1"], stale_after=1.0, lost_after=3.0)
    st = lambda seq, ts, ttl=2.0: {"type": "RobotState", "src": "amr1", "seq": seq, "ts": ts, "ttl": ttl, **state_body(1)}
    assert pt.update(st(2, 0.0), 0.1)
    assert not pt.update(st(1, 0.05), 0.1)           # older sequence rejected
    assert not pt.update(st(3, 0.0, ttl=0.5), 0.9)   # TTL expired
    assert pt.refresh(0.5) == [] and pt.entries["amr1"].status == FRESH
    pt.refresh(2.0)
    assert pt.entries["amr1"].status == STALE
    assert pt.refresh(3.5) == ["amr1"] and pt.entries["amr1"].status == LOST
    assert pt.refresh(4.0) == []                    # transition reported once
    assert pt.update(st(4, 5.0), 5.0)
    pt.refresh(5.0)
    assert pt.entries["amr1"].status == FRESH


def _publisher_proc(idx, base, duration, q):
    ctx = zmq.Context()
    node = PeerNode(ctx, NAMES[idx], NAMES, [base + i for i in range(3)], [base + 5 + i for i in range(3)])
    tj = time.time()
    while not node.join_step(20) and time.time() - tj < 15:
        pass
    for _ in range(10):
        node.join_step(20)          # keep beaconing briefly so slower peers also complete
    t0 = time.time()
    counts = {p: 0 for p in node.peers}
    next_pub = t0 + 0.5
    while time.time() - t0 < duration:
        now = time.time()
        if now >= next_pub:
            node.publish("RobotState", now - t0, 1.0, **state_body(idx))
            next_pub += 0.1                      # 10 Hz
        node.poll_raw(5)
        for m in node.take_pending():
            if m["type"] == "RobotState" and 1.5 < m["ts"] < duration - 0.5:
                counts[m["src"]] += 1
    q.put((NAMES[idx], counts))
    node.close()
    ctx.term()


def test_state_rate_across_processes():
    """PR-03: separate OS processes exchange state at 5-10 Hz over TCP (brokerless)."""
    base = next(_ports)
    q = mp.get_context("spawn").Queue()
    duration = 4.0
    procs = [mp.get_context("spawn").Process(target=_publisher_proc, args=(i, base, duration, q)) for i in range(3)]
    for p in procs:
        p.start()
    res = dict(q.get(timeout=60) for _ in procs)
    for p in procs:
        p.join(timeout=10)
    window = duration - 0.5 - 1.5
    for name, counts in res.items():
        for peer, c in counts.items():
            hz = c / window
            assert 5.0 <= hz <= 11.0, (name, peer, hz)
