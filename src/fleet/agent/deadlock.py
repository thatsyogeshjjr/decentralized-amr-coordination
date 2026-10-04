"""Deadlock detection and recovery (design doc §10.8, FR-07, PR-06).

Detector: a robot with a goal is *stalled* when its distance-to-goal has not improved by more than `eps` within the
window T_d.  This contains eq. (6) (no displacement => no progress) and additionally catches livelock (being pushed
back and forth).  A stalled robot is *deadlocked* when it is in a conflicting neighbourhood where the priority order
is failing: a robot of higher priority within two graph hops is *also* stalled.  (Waiting behind a higher-priority
robot that is making progress is normal PIBT behaviour, not a deadlock.)

Recovery ladder (each step is local, deterministic, no central arbiter):
  1. priority escalation: the deadlocked robot takes a priority boost above all un-boosted robots for
     `boost_time`; each strictly new best distance-to-goal extends it (so it can finish passing), up to
     `boost_max`.  After expiry an equal cooldown forbids re-escalation (the robot it was blocking gets its turn).
     A robot never escalates while a boosted robot is within two hops (no boost ping-pong).
  2. if it escalates `max_escalations` times without reaching the goal, the planner is told to invalidate the
     contested route node for a while (route invalidation -> detour), see `wants_invalidation`.

Episode accounting (design doc v3 §10.8, preflight P8; instrumentation only, the logic above is unchanged):
  deadlock episode  opens at an escalation when none is open (states detected = escalating at that time), counts
                    further escalations, notes rerouting (route invalidation), becomes `recovered` at
                    commit_progress (the legacy resolution metric) and closes at the arrival (distance improved);
                    closes as `timed_out` if not recovered within `episode_timeout` s, as `failed_goal_changed` if the
                    goal changes first; an episode still open at run end is classified `failed_run_end` by analysis
  congestion        stalled (no distance improvement for T_d) while *no* stalled higher-priority robot is within two
                    hops: the robot is queueing behind traffic that moves.  Closes on progress (`progressed`), when
                    it escalates (`became_deadlock`) or when the goal changes / clears.  Never escalated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BOOST = 1e6


@dataclass
class DeadlockMonitor:
    window: float = 6.0
    eps: float = 0.5
    boost_time: float = 8.0
    boost_max: float = 30.0
    boost_start: float = -1.0
    max_escalations: int = 3
    best: float = float("inf")
    last_improve: float = 0.0
    boost_until: float = -1.0
    escalations_this_goal: int = 0
    escalations: int = 0
    detections: list = field(default_factory=list)       # (t_detect) for each escalation
    recoveries: list = field(default_factory=list)       # s from detection to resolution (progress move committed)
    arrivals: list = field(default_factory=list)         # s from detection to the progress move being completed
    _open_commit: float | None = None
    _open_detection: float | None = None
    goal_key: object = None
    episode_timeout: float = 60.0
    episodes: list = field(default_factory=list)         # closed deadlock episodes
    congestion: list = field(default_factory=list)       # closed congestion episodes
    n_episodes: int = 0
    n_congestion: int = 0
    _ep: dict | None = None
    _cg: dict | None = None

    # -- episode bookkeeping (instrumentation) ---------------------------------------------------------------------
    def _close_ep(self, now: float, outcome: str) -> None:
        ep = self._ep
        ep["outcome"], ep["t_closed"] = outcome, round(now, 3)
        self.episodes.append(ep)
        del self.episodes[:-200]
        self._ep = None

    def _close_cg(self, now: float, outcome: str) -> None:
        cg = self._cg
        cg["outcome"], cg["t_end"], cg["duration"] = outcome, round(now, 3), round(now - cg["t_stall_start"], 3)
        self.congestion.append(cg)
        del self.congestion[:-200]
        self._cg = None

    def note_reroute(self, now: float) -> None:
        if self._ep is not None and self._ep["t_rerouting"] is None:
            self._ep["t_rerouting"] = round(now, 3)

    def episode_summary(self) -> dict:
        return {"episodes": self.episodes[-50:], "n_episodes": self.n_episodes, "open": self._ep,
                "congestion": self.congestion[-50:], "n_congestion": self.n_congestion, "congestion_open": self._cg,
                "episode_timeout": self.episode_timeout}

    def reset_goal(self, now: float, goal_key: object) -> None:
        if self._ep is not None:
            self._close_ep(now, "recovered" if self._ep["t_recovered"] is not None else "failed_goal_changed")
        if self._cg is not None:
            self._close_cg(now, "goal_changed")
        self.goal_key = goal_key
        self.best = float("inf")
        self.last_improve = now
        self.boost_until = -1.0
        self.escalations_this_goal = 0
        self._open_detection = None

    def boosted(self, now: float) -> bool:
        return now < self.boost_until

    def stalled(self, now: float) -> bool:
        return self.best < float("inf") and now - self.last_improve >= self.window

    def update(self, now: float, dist: float | None, higher_near: bool, boosted_near: bool) -> bool:
        """dist: current distance-to-goal (None = no goal).  higher_near: a higher-priority robot within two hops
        that is itself stalled.  Returns True if a new escalation happened."""
        if self._ep is not None and self._ep["t_recovered"] is None and \
                now - self._ep["t_detected"] > self.episode_timeout:
            self._close_ep(now, "timed_out")
        if dist is None:
            if self._ep is not None:
                self._close_ep(now, "recovered" if self._ep["t_recovered"] is not None else "failed_goal_changed")
            if self._cg is not None:
                self._close_cg(now, "goal_cleared")
            self.best = float("inf")
            self.last_improve = now
            self._open_detection = None
            return False
        if dist < self.best - self.eps:
            if self._ep is not None:
                if self._ep["t_recovered"] is None:
                    self._ep["t_recovered"] = round(now, 3)
                self._ep["t_arrived"] = round(now, 3)
                self._close_ep(now, "recovered")
            if self._cg is not None:
                self._close_cg(now, "progressed")
            self.best = dist
            self.last_improve = now
            if self.boosted(now):
                self.boost_until = min(self.boost_start + self.boost_max, now + self.boost_time)
            if self._open_detection is not None:
                if self._open_commit is None:
                    self.recoveries.append(round(now - self._open_detection, 3))
                self.arrivals.append(round(now - self._open_detection, 3))
                self._open_detection = None
                self._open_commit = None
        stalled = now - self.last_improve >= self.window
        cooling = now < self.boost_until + self.boost_time
        if stalled and not higher_near and self._cg is None and self._ep is None:
            self.n_congestion += 1
            self._cg = {"id": self.n_congestion, "t_stall_start": round(self.last_improve, 3), "t_detected": round(now, 3)}
        if stalled and not cooling and higher_near and not boosted_near:
            if self._cg is not None:
                self._close_cg(now, "became_deadlock")
            if self._ep is None:
                self.n_episodes += 1
                self._ep = {"id": self.n_episodes, "t_detected": round(now, 3), "t_escalating": round(now, 3),
                            "n_escalations": 0, "t_rerouting": None, "t_recovered": None, "t_arrived": None}
            self._ep["n_escalations"] += 1
            self.boost_start = now
            self.boost_until = now + self.boost_time
            self.last_improve = now
            self.best = dist
            self.escalations += 1
            self.escalations_this_goal += 1
            self.detections.append(round(now, 3))
            if self._open_detection is None:
                self._open_detection = now
            return True
        return False

    def commit_progress(self, now: float) -> None:
        """The robot acquired a node that reduces its distance-to-goal: the deadlock is resolved."""
        if self._open_detection is not None and self._open_commit is None:
            self._open_commit = now
            self.recoveries.append(round(now - self._open_detection, 3))
        if self._ep is not None and self._ep["t_recovered"] is None:
            self._ep["t_recovered"] = round(now, 3)

    @property
    def wants_invalidation(self) -> bool:
        return self.escalations_this_goal >= self.max_escalations
