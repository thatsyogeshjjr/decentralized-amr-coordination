"""Decentralized auction / consensus (FR-04, FR-09, PR-05) with three TaskManagers exchanging real message dicts
in memory (no simulator).  The fake brain supplies only what the task layer reads from its robot."""
import random

import pytest

from fleet.agent.task_manager import TaskManager
from fleet.common.config import DEFAULTS, PROJECT_ROOT, deep_merge
from fleet.common.messages import make
from fleet.world.mapgrid import WarehouseMap

M = WarehouseMap.load(PROJECT_ROOT / "configs/maps/warehouse_small.txt")
NAMES = ["amr0", "amr1", "amr2"]


class FakeView:
    def __init__(self, bus, name):
        self.bus, self.name = bus, name

    def occupied(self):
        return {b.node_at for n, b in self.bus.brains.items() if n != self.name}


class FakeBrain:
    def __init__(self, bus, name, node):
        self.bus, self.name, self.node_at, self.move = bus, name, node, None
        self.map, self.battery = M, 1.0
        self.view = FakeView(bus, name)

    def dist_field(self, goal):
        return M.bfs_dist(goal)

    def blocked_set(self):
        return set()

    def peer_nodes(self):
        return {n: (b.node_at, {"node": b.node_at, "goal_node": b.tm.current()}) for n, b in self.bus.brains.items()
                if n != self.name and n in self.bus.alive}

    def live_peers(self):
        return [n for n in NAMES if n != self.name and n in self.bus.alive]

    def _broadcast(self, peers, mtype, ttl, **body):
        for p in peers:
            self.bus.post(p, make(mtype, self.name, 0, self.bus.now, ttl, **body))


class Bus:
    def __init__(self, loss=0.0, seed=0):
        self.now, self.loss, self.rng = 0.0, loss, random.Random(seed)
        self.queue: list[tuple[str, dict]] = []
        self.alive = set(NAMES)
        cfg = deep_merge(DEFAULTS, {"tasks": {"bid_window": 0.3}})
        self.brains = {}
        for i, n in enumerate(NAMES):
            b = FakeBrain(self, n, M.starts[i])
            b.tm = TaskManager(cfg, n, M, b)
            self.brains[n] = b

    def post(self, to, msg):
        if self.rng.random() >= self.loss:
            self.queue.append((to, msg))

    def step(self, dt=0.05, orders=()):
        q, self.queue = self.queue, []
        for to, m in q:
            if to in self.alive:
                self.brains[to].tm.handle(m, self.now)
        for n in sorted(self.alive):
            self.brains[n].tm.on_orders(list(orders), self.now)
            self.brains[n].tm.tick(self.now, self.brains[n])
        self.now += dt


def orders(k, t=0.0):
    P, D = M.stations("pickup"), M.stations("drop")
    return [{"task_id": f"T{i}", "pickup": P[i % 3], "dropoff": D[(i + 1) % 3], "release": t} for i in range(k)]


def owners(bus):
    return {n: (b.tm.mine.task_id if b.tm.mine else None) for n, b in bus.brains.items() if n in bus.alive}


def test_simultaneous_tasks_get_unique_winners_quickly():
    bus = Bus()
    bus.step(orders=orders(3))
    for _ in range(30):
        bus.step()
    own = owners(bus)
    assert sorted(v for v in own.values()) == ["T0", "T1", "T2"]          # each robot exactly one, all distinct
    for b in bus.brains.values():
        assert all(lat < 1.0 for lat in b.tm.alloc_latency)              # PR-05
        # replicated tables agree on every winner
        for t in b.tm.tasks.values():
            assert t.status == "assigned" and own[t.winner] == t.task_id


def test_more_tasks_than_robots_leaves_queue_open():
    bus = Bus()
    bus.step(orders=orders(5))
    for _ in range(40):
        bus.step()
    own = owners(bus)
    assert len(set(own.values())) == 3 and None not in own.values()
    t0 = bus.brains["amr0"].tm.tasks
    assert sum(t.status == "open" for t in t0.values()) == 2


@pytest.mark.parametrize("seed", range(5))
def test_consensus_converges_under_message_loss(seed):
    bus = Bus(loss=0.3, seed=seed)
    bus.step(orders=orders(3))
    for _ in range(200):
        bus.step()
    own = owners(bus)
    held = [v for v in own.values() if v]
    assert len(held) == len(set(held)), f"duplicate assignment {own}"
    assert len(held) == 3


def test_lost_peer_task_is_reassigned():
    bus = Bus()
    bus.step(orders=orders(3))
    for _ in range(30):
        bus.step()
    victim = "amr1"
    lost_task = owners(bus)[victim]
    bus.alive.discard(victim)
    bus.step(orders=orders(4)[3:])          # a 4th order arrives too; survivors are busy until they finish
    for n in bus.alive:
        bus.brains[n].tm.on_peer_lost(victim, bus.now)
    t = bus.brains["amr0"].tm.tasks[lost_task]
    assert t.status == "open" and t.revision == 1
    # a survivor finishes its current task -> becomes free -> must pick the re-opened task or the new one
    survivor = bus.brains["amr2"]
    survivor.tm.mine.status = "done"
    survivor.tm.mine, survivor.tm.phase = None, None
    for _ in range(40):
        bus.step()
    assert owners(bus)["amr2"] in (lost_task, "T3")


def test_release_reopens_with_new_revision_and_excludes_releaser():
    bus = Bus()
    bus.step(orders=orders(1))
    for _ in range(30):
        bus.step()
    holder = next(n for n, v in owners(bus).items() if v == "T0")
    bus.brains[holder].tm.release_current(bus.now, "unreachable")
    for _ in range(40):
        bus.step()
    new_holder = next(n for n, v in owners(bus).items() if v == "T0")
    assert new_holder != holder
    assert bus.brains[new_holder].tm.tasks["T0"].revision == 1
