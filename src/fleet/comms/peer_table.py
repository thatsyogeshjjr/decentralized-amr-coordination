"""Peer table (design doc §10.1): newest valid state per known robot, with freshness / loss classification."""
from __future__ import annotations

from dataclasses import dataclass, field

FRESH, STALE, LOST = "fresh", "stale", "lost"


@dataclass
class PeerEntry:
    name: str
    state: dict | None = None
    intent: dict | None = None
    state_seq: int = -1
    intent_seq: int = -1
    last_heard: float = -1e9          # receiver sim time of last accepted message
    status: str = LOST
    lost_since: float | None = None
    rx: int = 0


@dataclass
class PeerTable:
    peers: list[str]
    stale_after: float = 1.0
    lost_after: float = 3.0
    entries: dict[str, PeerEntry] = field(default_factory=dict)
    rejected_stale: int = 0
    rejected_old: int = 0
    outages: list = field(default_factory=list)        # closed peer-outage episodes (FR-14 peer availability)
    _open_out: dict = field(default_factory=dict)      # peer -> open outage record
    n_outages: int = 0

    def __post_init__(self):
        self.entries = {p: PeerEntry(p) for p in self.peers}

    def update(self, msg: dict, now: float) -> bool:
        """Accept a RobotState / RobotIntent if newer than what we hold and not past its TTL."""
        e = self.entries.get(msg["src"])
        if e is None:
            return False
        if msg["ttl"] > 0 and now > msg["ts"] + msg["ttl"]:
            self.rejected_stale += 1
            return False
        if msg["type"] == "RobotState":
            if msg["seq"] <= e.state_seq:
                self.rejected_old += 1
                return False
            e.state, e.state_seq = msg, msg["seq"]
        elif msg["type"] == "RobotIntent":
            if msg["seq"] <= e.intent_seq:
                self.rejected_old += 1
                return False
            e.intent, e.intent_seq = msg, msg["seq"]
        else:
            return False
        e.last_heard = now
        e.rx += 1
        return True

    def heard(self, src: str, now: float) -> None:
        """Any authenticated traffic from a peer counts as a heartbeat."""
        e = self.entries.get(src)
        if e is not None:
            e.last_heard = max(e.last_heard, now)

    def refresh(self, now: float) -> list[str]:
        """Re-classify all peers; returns names that transitioned to LOST in this call."""
        newly_lost = []
        for e in self.entries.values():
            age = now - e.last_heard
            new = FRESH if age <= self.stale_after else STALE if age <= self.lost_after else LOST
            if new == LOST and e.status != LOST and e.state is not None:
                newly_lost.append(e.name)
                e.lost_since = now
            if new != LOST:
                e.lost_since = None
            if e.state is not None:
                o = self._open_out.get(e.name)
                if new != FRESH and o is None:
                    # outage starts when the last accepted message was heard; detected once it exceeds stale_after
                    self._open_out[e.name] = {"peer": e.name, "last_heard": round(e.last_heard, 3),
                                              "detected": round(now, 3), "worst": new}
                elif o is not None and new == LOST:
                    o["worst"] = LOST
                elif o is not None and new == FRESH:
                    o["recovered"] = round(now, 3)
                    o["duration"] = round(now - o["last_heard"], 3)
                    self.outages.append(self._open_out.pop(e.name))
                    self.n_outages += 1
                    del self.outages[:-200]
            e.status = new
        return newly_lost

    def outage_summary(self) -> dict:
        """Closed episodes plus those still open (reported as unrecovered)."""
        return {"closed": self.outages[-50:], "open": list(self._open_out.values()), "n_closed": self.n_outages}

    def fresh(self) -> list[PeerEntry]:
        return [e for e in self.entries.values() if e.status == FRESH and e.state is not None]

    def known(self) -> list[PeerEntry]:
        """Peers with any state, regardless of freshness (for conservative occupancy)."""
        return [e for e in self.entries.values() if e.state is not None]

    def live_names(self) -> list[str]:
        return [e.name for e in self.entries.values() if e.status != LOST]
