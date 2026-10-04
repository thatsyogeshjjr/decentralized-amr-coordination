"""Isolation / reconnection state machine and task reconciliation (preflight P4 + addendum A)."""
import pytest

from fleet.agent.isolation import CONNECTED, ISOLATED, PARTITIONED, RECONNECTING, IsolationMonitor
from fleet.comms.peer_table import PeerTable
from test_task_allocation import Bus, orders, owners


def st(src, seq, ts):
    return {"type": "RobotState", "src": src, "seq": seq, "ts": ts, "ttl": 1.0}


def drive(mon, pt, now):
    pt.refresh(now)
    return mon.update(now, pt)


def test_startup_does_not_trigger_isolation():
    pt, mon = PeerTable(["amr1", "amr2"]), IsolationMonitor()
    for k in range(100):
        assert drive(mon, pt, k * 0.05) is None
    assert mon.state == CONNECTED


def test_full_cycle_isolated_reconnecting_connected():
    pt, mon = PeerTable(["amr1", "amr2"]), IsolationMonitor(timeout=30.0, hold=1.0)
    pt.update(st("amr1", 1, 0.0), 0.0)
    pt.update(st("amr2", 1, 0.0), 0.0)
    assert drive(mon, pt, 0.5) is None and mon.state == CONNECTED
    pt.update(st("amr2", 2, 0.6), 0.6)
    assert drive(mon, pt, 1.2) is None and mon.state == CONNECTED     # amr2 still fresh: not isolated
    assert drive(mon, pt, 1.65) == "enter" and mon.state == ISOLATED  # every heard peer non-fresh
    pt.update(st("amr1", 2, 10.0), 10.0)
    assert drive(mon, pt, 10.0) is None and mon.state == RECONNECTING
    assert drive(mon, pt, 10.5) is None and mon.state == RECONNECTING  # holds one heartbeat period
    pt.update(st("amr1", 3, 10.9), 10.9)
    assert drive(mon, pt, 11.0) == "resume" and mon.state == CONNECTED
    ep = mon.summary()["episodes"][0]
    assert ep["t_isolated"] == 1.65 and ep["t_reconnect"] == 10.0 and ep["t_resume"] == 11.0
    assert ep["t_partitioned"] is None and ep["outage_s"] == pytest.approx(9.35)


def test_timeout_reports_partition_and_relapse_during_reconnect():
    pt, mon = PeerTable(["amr1"]), IsolationMonitor(timeout=30.0, hold=1.0)
    pt.update(st("amr1", 1, 0.0), 0.0)
    drive(mon, pt, 0.5)
    assert drive(mon, pt, 1.1) == "enter"
    assert drive(mon, pt, 31.0) is None and mon.state == ISOLATED      # 29.9 s: not yet
    assert drive(mon, pt, 31.2) is None and mon.state == PARTITIONED
    pt.update(st("amr1", 2, 40.0), 40.0)
    assert drive(mon, pt, 40.0) is None and mon.state == RECONNECTING
    assert drive(mon, pt, 41.5) is None and mon.state == PARTITIONED   # peer went silent again before resume
    pt.update(st("amr1", 3, 50.0), 50.0)
    drive(mon, pt, 50.0)
    pt.update(st("amr1", 4, 50.9), 50.9)
    assert drive(mon, pt, 51.0) == "resume"
    ep = mon.summary()["episodes"][0]
    assert ep["t_partitioned"] == 31.2 and ep["t_reconnect"] == 40.0 and ep["t_resume"] == 51.0
    assert mon.blocks() is False


# ------------------------------------------------------------------------------------------- task reconciliation
class Iso:
    def __init__(self, blocked):
        self.b = blocked

    def blocks(self):
        return self.b


def assigned_bus():
    bus = Bus()
    bus.step(orders=orders(3))
    for _ in range(30):
        bus.step()
    return bus


def test_isolated_robot_suspends_instead_of_reassigning_and_restores_on_reconnect():
    bus = assigned_bus()
    me = bus.brains["amr0"]
    peer_task = owners(bus)["amr1"]
    me.iso = Iso(True)
    me.tm.on_peer_lost("amr1", bus.now)
    t = me.tm.tasks[peer_task]
    assert t.status == "suspended" and t.revision == 0 and me.tm.reassigned_from_lost == []
    me.tm.tick(bus.now, me)                     # isolated: must not bid on it
    assert me.tm.my_bid is None

    class E:
        status = "fresh"
    me.table = type("T", (), {"entries": {"amr1": E()}})()
    me.iso = Iso(False)
    restored, reopened = me.tm.on_reconnect(bus.now)
    assert restored == [peer_task] and reopened == [] and t.status == "assigned" and t.winner == "amr1"


def test_suspended_task_of_still_missing_peer_is_reopened_on_reconnect():
    bus = assigned_bus()
    me = bus.brains["amr0"]
    peer_task = owners(bus)["amr1"]
    me.iso = Iso(True)
    me.tm.on_peer_lost("amr1", bus.now)

    class E:
        status = "lost"
    me.table = type("T", (), {"entries": {"amr1": E()}})()
    me.iso = Iso(False)
    restored, reopened = me.tm.on_reconnect(bus.now)
    assert reopened == [peer_task] and me.tm.tasks[peer_task].status == "open" and me.tm.tasks[peer_task].revision == 1


def test_carrier_reasserts_and_reassigned_copy_is_dropped():
    bus = assigned_bus()
    carrier = bus.brains["amr1"]
    tid = owners(bus)["amr1"]
    carrier.tm.phase, carrier.load = "drop", True          # picked up: payload sensor confirms
    # the other robots lost amr1 (its link was down) and reassign the task at revision 1
    bus.alive.discard("amr1")
    for n in ("amr0", "amr2"):
        bus.brains[n].tm.on_peer_lost("amr1", bus.now)
    bus.brains["amr2"].tm.mine.status = "done"
    bus.brains["amr2"].tm.mine, bus.brains["amr2"].tm.phase = None, None
    for _ in range(30):
        bus.step()
    assert owners(bus)["amr2"] == tid                       # reassigned copy
    bus.alive.add("amr1")                                   # reconnect: consensus rev 1 reaches the carrier
    for _ in range(60):
        bus.step()
    own = owners(bus)
    assert own["amr1"] == tid and own["amr2"] != tid
    assert carrier.tm.reasserted and carrier.tm.tasks[tid].winner == "amr1"
    assert all(b.tm.tasks[tid].winner == "amr1" for b in bus.brains.values())


def test_non_carrier_drops_reassigned_task_on_reconnect():
    bus = assigned_bus()
    me = bus.brains["amr1"]
    tid = owners(bus)["amr1"]                               # still in pickup phase: not carrying
    bus.alive.discard("amr1")
    for n in ("amr0", "amr2"):
        bus.brains[n].tm.on_peer_lost("amr1", bus.now)
    bus.brains["amr2"].tm.mine.status = "done"
    bus.brains["amr2"].tm.mine, bus.brains["amr2"].tm.phase = None, None
    for _ in range(30):
        bus.step()
    bus.alive.add("amr1")
    for _ in range(60):
        bus.step()
    own = owners(bus)
    assert own["amr2"] == tid and own["amr1"] != tid and not me.tm.reasserted


def test_payload_sensor_releases_failed_pickup_and_blocks_false_done():
    bus = assigned_bus()
    me = bus.brains["amr0"]
    tid = owners(bus)["amr0"]
    me.tm.phase, me.load = "drop", False                    # pickup report was rejected: no load on board
    me.tm.tick(bus.now, me)
    assert me.tm.mine is None and me.tm.pickup_failed[0]["task"] == tid
    assert me.tm.tasks[tid].status == "open" and me.tm.tasks[tid].revision == 1
    # a carrier ignores TaskDone for the order it physically holds
    bus2 = assigned_bus()
    c = bus2.brains["amr1"]
    tid2 = owners(bus2)["amr1"]
    c.tm.phase, c.load = "drop", True
    from fleet.common.messages import make
    c.tm.handle(make("TaskDone", "amr2", 1, bus2.now, 0.0, task_id=tid2, revision=5), bus2.now)
    assert c.tm.mine is not None and c.tm.ignored_done == 1
