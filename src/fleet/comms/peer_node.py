"""Brokerless ZeroMQ peer node (design doc §10.2-10.3).

Each robot owns its endpoints; there is no broker and no central relay:
  * PUB  (bind robot_pub[i])     high-rate RobotState / RobotIntent / BlockageEvent dissemination
  * SUB  (connect to every peer's PUB)
  * ROUTER (bind robot_router[i]) inbox for targeted coordination (bids, consensus, claims, pushes)
  * one DEALER per peer (connect to the peer's ROUTER, identity = own name)

Application-level reliability fields: monotonically increasing sequence numbers (per sender for PUB, per link
for DEALER), sender timestamp, message type and TTL.  Faults are injected on the receive path.

Delivery barrier: the world clock forwards, per peer, the highest sequence numbers that peer had sent by the
end of the previous tick (counters only, never content).  `sync()` waits (bounded) until those messages have
physically arrived, then applies fault injection in a deterministic order.  This makes runs reproducible.
"""
from __future__ import annotations

import time
from collections import Counter

import zmq

from fleet.common.messages import SchemaError, decode, encode, make
from fleet.comms.fault_injector import FaultInjector


class PeerNode:
    def __init__(self, ctx: zmq.Context, name: str, names: list[str], pub_ports: list[int], router_ports: list[int],
                 latency: float = 0.0, jitter: float = 0.0, loss: float = 0.0, seed: int = 0, host: str = "127.0.0.1",
                 burst: dict | None = None, disconnect: list[dict] | None = None, reorder_pub: bool = False,
                 trace_path: str | None = None):
        self.name = name
        self.idx = names.index(name)
        self.peers = [p for p in names if p != name]
        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.setsockopt(zmq.SNDHWM, 100000)
        self.pub.bind(f"tcp://{host}:{pub_ports[self.idx]}")
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.setsockopt(zmq.RCVHWM, 100000)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.router = ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(f"tcp://{host}:{router_ports[self.idx]}")
        self.dealers: dict[str, zmq.Socket] = {}
        for p in self.peers:
            j = names.index(p)
            self.sub.connect(f"tcp://{host}:{pub_ports[j]}")
            d = ctx.socket(zmq.DEALER)
            d.setsockopt(zmq.LINGER, 0)
            d.setsockopt(zmq.IDENTITY, name.encode())
            d.setsockopt(zmq.SNDHWM, 10000)
            d.connect(f"tcp://{host}:{router_ports[j]}")
            self.dealers[p] = d
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        self.poller.register(self.router, zmq.POLLIN)
        self.fault = FaultInjector(latency, jitter, loss, seed=seed * 1000 + self.idx, burst=burst,
                                   disconnect=disconnect, name=name, reorder_pub=reorder_pub, trace_path=trace_path)
        self.pub_seq = 0
        self.dm_seq = {p: 0 for p in self.peers}
        self.rx_pub_max = {p: 0 for p in self.peers}
        self.rx_dm_max = {p: 0 for p in self.peers}
        self.tx_bytes = 0
        self.tx_msgs: Counter = Counter()
        self.rx_bytes = 0
        self.decode_errors = 0
        self.foreign_src = 0
        self.barrier_timeouts = 0
        self.barrier_wait = 0.0
        self._pending: list[tuple[str, dict]] = []

    # ------------------------------------------------------------------------------------------------ sending
    def publish(self, mtype: str, now: float, ttl: float, **body) -> dict:
        self.pub_seq += 1
        msg = make(mtype, self.name, self.pub_seq, now, ttl, **body)
        raw = encode(msg)
        self.pub.send(raw)
        self.tx_bytes += len(raw)
        self.tx_msgs[mtype] += 1
        return msg

    def send(self, peer: str, mtype: str, now: float, ttl: float, **body) -> dict:
        self.dm_seq[peer] += 1
        msg = make(mtype, self.name, self.dm_seq[peer], now, ttl, **body)
        raw = encode(msg)
        try:
            self.dealers[peer].send(raw, zmq.NOBLOCK)
        except zmq.Again:          # peer unreachable and queue full: the message is lost, like on a real link
            pass
        self.tx_bytes += len(raw)
        self.tx_msgs[mtype] += 1
        return msg

    def broadcast(self, mtype: str, now: float, ttl: float, peers: list[str] | None = None, **body) -> None:
        for p in (self.peers if peers is None else peers):
            self.send(p, mtype, now, ttl, **body)

    def sent_counters(self) -> dict:
        return {"pub": self.pub_seq, "dm": dict(self.dm_seq)}

    # ------------------------------------------------------------------------------------------------ receiving
    def _pull(self, timeout_ms: int = 0) -> int:
        n = 0
        events = dict(self.poller.poll(timeout_ms))
        while events:
            if self.sub in events:
                while True:
                    try:
                        raw = self.sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    n += self._accept("pub", raw)
            if self.router in events:
                while True:
                    try:
                        _ident, raw = self.router.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    n += self._accept("dm", raw)
            events = dict(self.poller.poll(0))
        return n

    def _accept(self, chan: str, raw: bytes) -> int:
        self.rx_bytes += len(raw)
        try:
            msg = decode(raw)
        except (SchemaError, ValueError):
            self.decode_errors += 1
            return 0
        src = msg["src"]
        if src not in self.rx_pub_max:
            self.foreign_src += 1        # not a configured peer: never processed
            return 0
        if chan == "pub":
            self.rx_pub_max[src] = max(self.rx_pub_max[src], msg["seq"])
        else:
            self.rx_dm_max[src] = max(self.rx_dm_max[src], msg["seq"])
        self._pending.append((chan, msg))
        return 1

    def _satisfied(self, expect: dict) -> bool:
        for p, c in expect.items():
            if p not in self.rx_pub_max:
                continue
            if self.rx_pub_max[p] < int(c.get("pub", 0)):
                return False
            if self.rx_dm_max[p] < int(c.get("dm", {}).get(self.name, 0)):
                return False
        return True

    def sync(self, now: float, expect: dict | None = None, timeout: float = 0.25) -> list[dict]:
        """Wait (bounded) for the peer traffic announced in `expect`, then return messages deliverable at `now`."""
        t0 = time.perf_counter()
        self._pull(0)
        if expect:
            while not self._satisfied(expect):
                left = timeout - (time.perf_counter() - t0)
                if left <= 0:
                    self.barrier_timeouts += 1
                    break
                self._pull(int(max(1, left * 1000)))
        self.barrier_wait += time.perf_counter() - t0
        if expect is not None:
            # deliver exactly the traffic announced for this tick; anything newer (sent by a faster peer during the
            # current tick) is held back to the next tick so delivery does not depend on OS scheduling
            batch, keep = [], []
            for chan, msg in self._pending:
                c = expect.get(msg["src"])
                lim = None if c is None else (int(c.get("pub", 0)) if chan == "pub" else int(c.get("dm", {}).get(self.name, 0)))
                (batch if lim is None or msg["seq"] <= lim else keep).append((chan, msg))
            self._pending = keep
        else:
            batch, self._pending = self._pending, []
        batch.sort(key=lambda cm: (cm[1]["src"], cm[0], cm[1]["seq"]))
        for chan, msg in batch:
            if msg["type"] == "Beacon":
                # join-phase beacons still in flight: their number depends on wall-clock join timing, so they must
                # not consume fault-injection RNG draws (v2 lossy runs were not bit-reproducible because of this)
                continue
            self.fault.offer(chan, msg, now)
        return self.fault.deliver(now)

    def poll_raw(self, timeout_ms: int) -> None:
        """Used during the join phase (before the clock starts)."""
        self._pull(timeout_ms)

    # ------------------------------------------------------------------------------------------------ join
    def join_step(self, timeout_ms: int = 50) -> bool:
        """One beacon round of the startup handshake.  Returns True once this node has heard beacons from every
        peer *and* every peer reports having heard this node (both PUB->SUB directions proven)."""
        if not hasattr(self, "_heard"):
            self._heard: set[str] = set()
            self._heard_me: set[str] = set()
        self.publish("Beacon", 0.0, 0.0, heard=sorted(self._heard))
        self._pull(timeout_ms)
        keep = []
        for chan, m in self._pending:
            if m["type"] == "Beacon":
                self._heard.add(m["src"])
                if self.name in m["heard"]:
                    self._heard_me.add(m["src"])
            else:
                keep.append((chan, m))
        self._pending = keep
        return self._heard >= set(self.peers) and self._heard_me >= set(self.peers)

    def take_pending(self) -> list[dict]:
        batch, self._pending = self._pending, []
        return [m for _, m in batch]

    def stats(self) -> dict:
        return {"tx_bytes": self.tx_bytes, "rx_bytes": self.rx_bytes, "tx_msgs": dict(self.tx_msgs),
                "rx_offered": self.fault.received, "rx_dropped": self.fault.dropped,
                "decode_errors": self.decode_errors, "foreign_src": self.foreign_src, "barrier_timeouts": self.barrier_timeouts,
                "barrier_wait_s": round(self.barrier_wait, 3), "impair": self.fault.summary()}

    def close(self) -> None:
        for s in [self.pub, self.sub, self.router, *self.dealers.values()]:
            s.close(0)
        self.fault.close()
