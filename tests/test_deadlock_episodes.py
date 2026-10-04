"""Deadlock / congestion episode accounting (design doc v3 §10.8, preflight P8).  Drives DeadlockMonitor directly
with synthetic distance-to-goal traces; the legacy recoveries/arrivals lists must be unaffected."""
import pytest

from fleet.agent.deadlock import DeadlockMonitor

DT = 0.05


def run(dm, t0, t1, dist, higher=False, boosted=False):
    t = t0
    while t < t1 - 1e-9:
        dm.update(round(t, 3), dist, higher, boosted)
        t += DT
    return round(t, 3)


def fresh():
    dm = DeadlockMonitor(window=6.0, eps=0.5, boost_time=8.0)
    dm.reset_goal(0.0, "g1")
    return dm


def test_deadlock_episode_states_and_recovery():
    dm = fresh()
    t = run(dm, 0.0, 8.0, 10.0, higher=True)              # stalled with a stalled higher-priority neighbour
    # legacy rule: first escalation needs stall >= T_d (6 s) and the initial cooldown (boost_until + boost_time = 7 s)
    assert dm.n_episodes == 1 and dm._ep["t_detected"] == pytest.approx(7.0, abs=DT)
    assert dm._ep["n_escalations"] == 1 and dm.n_congestion == 0
    dm.note_reroute(t)
    dm.commit_progress(t + 1.0)                           # acquired a node closer to the goal
    dm.update(t + 2.0, 9.0, False, False)                 # ... and arrived there
    ep, = dm.episodes
    assert ep["outcome"] == "recovered" and ep["t_rerouting"] == t
    assert ep["t_recovered"] == pytest.approx(t + 1.0) and ep["t_arrived"] == pytest.approx(t + 2.0)
    # legacy metrics unchanged
    assert dm.recoveries == [pytest.approx(t + 1.0 - ep["t_detected"], abs=1e-3)]
    assert dm.arrivals == [pytest.approx(t + 2.0 - ep["t_detected"], abs=1e-3)]


def test_repeated_episodes_get_distinct_ids():
    dm = fresh()
    t = run(dm, 0.0, 8.0, 10.0, higher=True)
    dm.update(t, 9.0, True, False)                        # progress: episode 1 recovered
    t = run(dm, t + DT, t + 30.0, 9.0, higher=True)       # stalled again long after the cooldown
    assert dm.n_episodes == 2 and [e["id"] for e in dm.episodes] == [1] and dm._ep["id"] == 2


def test_timeout_boundary():
    dm = fresh()
    run(dm, 0.0, 7.05, 10.0, higher=True)
    td = dm._ep["t_detected"]
    dm.update(round(td + 59.9, 3), 10.0, False, False)
    assert dm._ep is not None                             # 59.9 s: still open
    dm.update(round(td + 60.1, 3), 10.0, False, False)
    assert dm._ep is None and dm.episodes[0]["outcome"] == "timed_out"
    dm.update(round(td + 61.0, 3), 5.0, False, False)     # late progress does not reopen / re-close it
    assert len(dm.episodes) == 1


def test_goal_change_fails_open_episode_and_run_end_reports_open():
    dm = fresh()
    run(dm, 0.0, 8.0, 10.0, higher=True)
    dm.reset_goal(8.0, "g2")
    assert dm.episodes[0]["outcome"] == "failed_goal_changed"
    run(dm, 8.0, 24.0, 10.0, higher=True)
    s = dm.episode_summary()
    assert s["open"] is not None and s["open"]["id"] == 2  # analysis classifies this as failed_run_end


def test_congestion_vs_deadlock_classification():
    dm = fresh()
    t = run(dm, 0.0, 10.0, 10.0, higher=False)            # queueing behind a robot that is progressing
    assert dm.n_congestion == 1 and dm.n_episodes == 0 and dm.escalations == 0
    dm.update(t, 9.0, False, False)
    c, = dm.congestion
    assert c["outcome"] == "progressed" and c["duration"] == pytest.approx(t - 0.0, abs=DT)
    # congestion that turns into mutual conflict is closed as became_deadlock and opens a deadlock episode
    t = run(dm, t + DT, t + 7.0, 9.0, higher=False)
    run(dm, t, t + 1.0, 9.0, higher=True)
    assert dm.congestion[-1]["outcome"] == "became_deadlock" and dm.n_episodes == 1
