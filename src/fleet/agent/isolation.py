"""Communication isolation / reconnection state machine (design doc v3 §10.3; preflight P4 + addendum A).

States
  CONNECTED     at least one peer is fresh (or no peer was ever heard: fleet startup)
  ISOLATED      every peer that has ever been heard is non-fresh at the same time: the robot assumes its *own* link
                is down, makes no new claims / pushes / bids, completes only an already granted move and holds
  PARTITIONED   ISOLATED for `timeout` s: reported as a persistent partition; behaviour stays conservative (hold)
  RECONNECTING  a peer state passed seq/TTL validation again; hold one heartbeat period so every reachable peer's
                state refreshes, then reconcile (release stale nodes, restore/reopen suspended tasks) and resume

`update()` returns the action for the brain: "enter" (cancel pending coordination), "resume" (reconcile) or None.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fleet.comms.peer_table import FRESH, PeerTable

CONNECTED, ISOLATED, PARTITIONED, RECONNECTING = "connected", "isolated", "partitioned", "reconnecting"


@dataclass
class IsolationMonitor:
    timeout: float = 30.0
    hold: float = 1.0
    state: str = CONNECTED
    episodes: list = field(default_factory=list)
    _ep: dict | None = None
    _t_reconnect: float | None = None

    def blocks(self) -> bool:
        """True while new claims, pushes and bids are suspended."""
        return self.state != CONNECTED

    def update(self, now: float, table: PeerTable) -> str | None:
        known = [e for e in table.entries.values() if e.state is not None]
        any_fresh = any(e.status == FRESH for e in known)
        if self.state == CONNECTED:
            if known and not any_fresh:
                self.state = ISOLATED
                self._ep = {"t_isolated": round(now, 3), "t_partitioned": None, "t_reconnect": None,
                            "t_resume": None, "released_nodes": [], "tasks_restored": [], "tasks_reopened": []}
                return "enter"
            return None
        if self.state in (ISOLATED, PARTITIONED):
            if any_fresh:
                self.state = RECONNECTING
                self._t_reconnect = now
                if self._ep["t_reconnect"] is None:
                    self._ep["t_reconnect"] = round(now, 3)
            elif self.state == ISOLATED and now - self._ep["t_isolated"] >= self.timeout:
                self.state = PARTITIONED
                self._ep["t_partitioned"] = round(now, 3)
            return None
        # RECONNECTING
        if not any_fresh:
            self.state = PARTITIONED if self._ep["t_partitioned"] is not None else ISOLATED
            return None
        if now - self._t_reconnect >= self.hold:
            self.state = CONNECTED
            self._ep["t_resume"] = round(now, 3)
            self._ep["outage_s"] = round(now - self._ep["t_isolated"], 3)
            self.episodes.append(self._ep)
            return "resume"
        return None

    def note(self, key: str, values: list) -> None:
        """Attach reconciliation details to the episode being closed (called right after 'resume')."""
        ep = self.episodes[-1] if self.state == CONNECTED and self.episodes else self._ep
        if ep is not None:
            ep[key] = list(values)

    def summary(self) -> dict:
        return {"state": self.state, "episodes": self.episodes[-20:], "n_episodes": len(self.episodes),
                "open": None if self.state == CONNECTED else self._ep}
