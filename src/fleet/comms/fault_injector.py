"""Receive-side communication impairment layer (design doc v3 §10.3, FR-14; v2 §16.2 latency / packet loss).

Applied per received message in a deterministic order (sorted by sender, channel, seq) with seeded RNGs, so a given
seed reproduces the same drop and delay pattern.  All times are *simulation* time.

Impairments
  * latency L and jitter J ~ U(0, jitter): a message stamped at sender time ts is due at ts + L + J
  * per-link FIFO (default): due = max(due, due of the previous message on the same (src, channel) link), like TCP;
    `reorder_pub=True` lets jitter reorder the PUB channel only (state/intent are protected by sequence rejection,
    blockage events are idempotent); the DEALER/ROUTER channel is always FIFO (the claim protocol assumes TCP order)
  * independent loss with probability `loss` on the lossy channels
  * burst loss: a Gilbert-Elliott chain per (src, channel) link, stepped once per offered message;
    burst = {"p_enter": G->B, "p_exit": B->G, "loss_bad": drop probability in B (default 1)}
  * temporary disconnection: disconnect = [{"robot": i, "t": t0, "duration": d}]; while t0 <= ts < t0 + d every
    message from or to robot i is dropped (both directions, all channels)

Delivery is quantized to the receiver's control ticks: the lockstep barrier first offers a message sent at tick k at
tick k+1, and `deliver(now)` releases messages with due <= now.  The effective delay of a message is therefore
(first tick time >= max(ts + tick, due)) - ts; configured 0, 10 and 50 ms all deliver after one 50 ms tick.  This is a
simulation impairment model, not a model of real wireless timing.

The historical (v2) behaviour is preserved bit-for-bit when jitter = 0 and burst/disconnect are off: the main RNG is
consumed exactly as before and FIFO ordering is then a no-op.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import numpy as np

TRACE_FIELDS = ["recv_t", "dst", "src", "chan", "type", "seq", "ts", "cfg_latency", "jitter_sample", "cfg_loss",
                "burst_state", "outcome", "due", "delivered_t", "eff_delay"]


class FaultInjector:
    def __init__(self, latency: float = 0.0, jitter: float = 0.0, loss: float = 0.0, seed: int = 0,
                 lossy_channels: tuple[str, ...] = ("pub", "dm"), burst: dict | None = None,
                 disconnect: list[dict] | None = None, name: str = "", reorder_pub: bool = False,
                 trace_path: str | None = None):
        if not 0.0 <= loss < 1.0:
            raise ValueError("loss must be in [0, 1)")
        self.latency, self.jitter, self.loss = float(latency), float(jitter), float(loss)
        self.rng = np.random.default_rng(seed)
        self.burst_rng = np.random.default_rng([seed, 7919])      # separate stream: never perturbs the main RNG
        self.lossy = set(lossy_channels)
        self.burst = dict(burst) if burst else None
        if self.burst:
            for k in ("p_enter", "p_exit"):
                if not 0.0 <= float(self.burst[k]) <= 1.0:
                    raise ValueError(f"burst.{k} must be in [0, 1]")
        self.disconnect = [(f"amr{int(d['robot'])}" if not isinstance(d["robot"], str) else d["robot"],
                            float(d["t"]), float(d["t"]) + float(d["duration"])) for d in (disconnect or [])]
        self.name = name
        self.reorder_pub = bool(reorder_pub)
        self._heap: list[tuple[float, int, dict, str, float]] = []
        self._n = 0
        self.received = 0
        self.dropped = 0
        self.drops_by_reason: Counter = Counter()
        self._bad: dict[tuple, bool] = defaultdict(bool)
        self._last_due: dict[tuple, float] = {}
        self._last_seq: dict[tuple, int] = {}
        self.reorder_count = 0
        self.delivered = 0
        self._age: dict[str, list[float]] = defaultdict(list)    # per message type: delivery - ts (sim s)
        self._jit: list[float] = []
        self._trace = None
        if trace_path:
            self._trace = open(trace_path, "w", buffering=1 << 16)
            self._trace.write(",".join(TRACE_FIELDS) + "\n")

    # ------------------------------------------------------------------------------------------------ helpers
    def disconnected(self, src: str, ts: float) -> bool:
        return any((src == r or self.name == r) and t0 <= ts < t1 for r, t0, t1 in self.disconnect)

    def _row(self, now, msg, chan, jit, bstate, outcome, due="", delivered="", eff="") -> None:
        if self._trace is not None:
            self._trace.write(",".join(str(v) for v in (
                "" if now is None else round(now, 4), self.name, msg.get("src", ""), chan, msg.get("type", ""),
                msg.get("seq", ""), msg.get("ts", ""), self.latency, "" if jit is None else round(jit, 5), self.loss,
                bstate, outcome, "" if due == "" else round(due, 5), delivered, eff)) + "\n")

    # ------------------------------------------------------------------------------------------------ path
    def offer(self, chan: str, msg: dict, now: float | None = None) -> None:
        self.received += 1
        link = (msg.get("src"), chan)
        ts = float(msg["ts"])
        # historical order of RNG use: iid loss draw first, then the jitter draw
        if chan in self.lossy and self.loss > 0 and self.rng.random() < self.loss:
            self._drop("random", now, msg, chan)
            return
        bstate = ""
        if self.burst and chan in self.lossy:
            bad = self._bad[link]
            u = self.burst_rng.random()
            bad = (u >= float(self.burst["p_exit"])) if bad else (u < float(self.burst["p_enter"]))
            self._bad[link] = bad
            bstate = "B" if bad else "G"
            if bad and self.burst_rng.random() < float(self.burst.get("loss_bad", 1.0)):
                self._drop("burst", now, msg, chan, bstate)
                return
        if self.disconnect and self.disconnected(msg.get("src"), ts):
            self._drop("disconnect", now, msg, chan, bstate)
            return
        jit = self.rng.uniform(0, self.jitter) if self.jitter > 0 else 0.0
        due = ts + self.latency + jit
        if not (self.reorder_pub and chan == "pub"):
            due = max(due, self._last_due.get(link, -1e18))
            self._last_due[link] = due
        if self.jitter > 0:
            self._jit.append(jit)
        heapq.heappush(self._heap, (due, self._n, msg, chan, jit))
        self._n += 1
        if self._trace is not None:
            self._row(now, msg, chan, jit, bstate, "queued", due)

    def _drop(self, reason: str, now, msg: dict, chan: str, bstate: str = "") -> None:
        self.dropped += 1
        self.drops_by_reason[reason] += 1
        self._row(now, msg, chan, None, bstate, "drop_" + reason)

    def deliver(self, now: float) -> list[dict]:
        out = []
        while self._heap and self._heap[0][0] <= now + 1e-9:
            due, _, msg, chan, jit = heapq.heappop(self._heap)
            out.append(msg)
            self.delivered += 1
            link = (msg.get("src"), chan)
            seq = msg.get("seq")
            if seq is not None:
                if seq < self._last_seq.get(link, -1):
                    self.reorder_count += 1
                self._last_seq[link] = max(seq, self._last_seq.get(link, -1))
            age = now - float(msg["ts"])
            self._age[msg.get("type", "?")].append(age)
            if self._trace is not None:
                self._row(now, msg, chan, jit, "", "delivered", due, round(now, 4), round(age, 4))
        return out

    @property
    def in_flight(self) -> int:
        return len(self._heap)

    def summary(self) -> dict:
        """Aggregate impairment record (sim-clock values): configured vs measured."""
        ages = {}
        for typ, v in sorted(self._age.items()):
            a = np.asarray(v)
            ages[typ] = {"n": int(a.size), "median_s": round(float(np.median(a)), 4),
                         "p90_s": round(float(np.percentile(a, 90)), 4), "max_s": round(float(a.max()), 4)}
        allv = np.concatenate([np.asarray(v) for v in self._age.values()]) if self._age else np.zeros(0)
        return {
            "configured": {"latency_s": self.latency, "jitter_s": self.jitter, "loss": self.loss,
                           "burst": self.burst, "disconnect": [list(d) for d in self.disconnect],
                           "reorder_pub": self.reorder_pub},
            "offered": self.received, "delivered": self.delivered, "dropped": self.dropped,
            "drops_by_reason": dict(self.drops_by_reason), "reorder_count": self.reorder_count,
            "jitter_sample_mean_s": round(float(np.mean(self._jit)), 5) if self._jit else 0.0,
            "eff_delay_median_s": round(float(np.median(allv)), 4) if allv.size else None,
            "eff_delay_p90_s": round(float(np.percentile(allv, 90)), 4) if allv.size else None,
            "eff_delay_max_s": round(float(allv.max()), 4) if allv.size else None,
            "age_by_type": ages,
        }

    def close(self) -> None:
        if self._trace is not None:
            self._trace.close()
            self._trace = None
