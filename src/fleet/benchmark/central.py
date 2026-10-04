"""BENCHMARK ONLY - centralized reference planner (design doc App. B, App. D configuration 2).

A single process owns the whole fleet plan: it assigns orders greedily (nearest idle robot) and computes
synchronous PIBT steps for all robots; robots in `planner = "central"` mode only execute the moves it assigns.
It is used solely to measure coordination overhead against the decentralized architecture and is never part of
the proposed system.  For a fair comparison it runs in simulated time: it joins the world clock's lockstep like a
robot (same delivery barrier), so its reaction time is not inflated by the accelerated wall clock.

    python -m fleet.benchmark.central --config <resolved_config.json>
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import zmq

from fleet.agent.pibt import pibt_step
from fleet.agent.simlink import SimLink
from fleet.common.config import ports, resolve
from fleet.comms.peer_node import PeerNode
from fleet.learning.gridsim import intended_path
from fleet.world.mapgrid import WarehouseMap
from fleet.world.mjcf_builder import robot_name

NAME = "central"


def central_names(cfg: dict) -> tuple[list[str], list[int], list[int]]:
    p = ports(cfg)
    n = int(cfg["fleet"]["n_robots"])
    names = [robot_name(i) for i in range(n)] + [NAME]
    return names, p["robot_pub"] + [p["central_pub"]], p["robot_router"] + [p["central_pub"] + 1]


class CentralPlanner:
    def __init__(self, cfg: dict, ctx: zmq.Context):
        self.cfg = cfg
        self.map = WarehouseMap.load(resolve(cfg["world"]["map"]), cfg["world"]["cell"])
        names, pubs, rtrs = central_names(cfg)
        self.robots = names[:-1]
        self.net = PeerNode(ctx, NAME, names, pubs, rtrs, seed=int(cfg["run"]["seed"]))
        self.states: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.assigned: dict[str, str] = {}        # robot -> task id
        self.assigned_at: dict[str, int] = {}
        self.done: set[str] = set()
        self.step = 0
        self.plan: dict = {}
        self.tb = np.random.default_rng(int(cfg["run"]["seed"])).random((len(self.robots), max(self.map.pos) + 1))
        self._dist: dict[int, dict] = {}
        self.steps_planned = 0

    def D(self, g: int) -> dict:
        if g not in self._dist:
            self._dist[g] = self.map.bfs_dist(g)
        return self._dist[g]

    def tick(self, sense: dict) -> dict:
        now = float(sense["t"])
        for o in sense.get("orders", []):
            self.orders[o["task_id"]] = o
        for m in self.net.sync(now, sense.get("expect"), float(self.cfg["run"]["barrier_timeout"])):
            if m["type"] == "RobotState" and (m["src"] not in self.states or m["seq"] > self.states[m["src"]]["seq"]):
                self.states[m["src"]] = m
                self.done.update(m.get("done", []))
        alive = [r for r in self.robots if r in sense.get("alive", self.robots)]
        ready = all(r in self.states and self.states[r].get("cstep", -1) >= self.step and
                    self.states[r].get("moving_to") is None for r in alive)
        if ready and alive:
            self._plan(alive)
        if self.plan and (self.step != getattr(self, "_last_pub_step", -1) or int(now * 20) % 10 == 0):
            self.net.publish("CentralPlan", now, 5.0, step=self.step, assign=self.plan)
            self._last_pub_step = self.step
        return {"type": "cmd", "tick": sense["tick"], "wl": 0.0, "wr": 0.0, "sent": self.net.sent_counters(),
                "status": {}, "done": False}

    def _plan(self, alive: list[str]) -> None:
        # order assignment: nearest idle robot, in release order
        for r in list(self.assigned):
            if self.assigned[r] in self.done or r not in alive:
                del self.assigned[r]
        busy = set(self.assigned.values())
        open_orders = [o for o in self.orders.values() if o["task_id"] not in self.done and o["task_id"] not in busy]
        for o in sorted(open_orders, key=lambda o: (o["release"], o["task_id"])):
            idle = [r for r in alive if r not in self.assigned]
            if not idle:
                break
            r = min(idle, key=lambda r: (self.D(o["pickup"]).get(self.states[r]["node"], 1e9), r))
            self.assigned[r] = o["task_id"]
            self.assigned_at[r] = self.step
        pos, prio, dist, paths = [], [], [], []
        idx = {r: i for i, r in enumerate(self.robots)}
        goals = {}
        for r in alive:
            st = self.states[r]
            tid = self.assigned.get(r)
            goal = None
            if tid is not None:
                o = self.orders[tid]
                goal = o["dropoff"] if st.get("carrying") else o["pickup"]
            goals[r] = goal
            pos.append(st["node"])
            if st.get("dwelling"):
                prio.append(1e12)
                dist.append(lambda v, p=st["node"]: 0.0 if v == p else 1e9)
                paths.append([])
            else:
                prio.append((self.step - self.assigned_at.get(r, self.step)) * 2.0 + idx[r] * 1e-3 if goal is not None else -1.0)
                dist.append((lambda v, g=goal: self.D(g).get(v, 1e9)) if goal is not None else (lambda v: 0.0))
                paths.append(intended_path(self.map, st["node"], self.D(goal)) if goal is not None else [])
        nxt = pibt_step(self.map, pos, prio, dist, tiebreak=lambda i, v: self.tb[idx[alive[i]], v], paths=paths)
        self.step += 1
        self.steps_planned += 1
        self.plan = {r: {"next": int(nxt[i]), "goal": goals[r],
                         "task": (dict(self.orders[self.assigned[r]]) if r in self.assigned else None)}
                     for i, r in enumerate(alive)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(open(a.config).read())
    ctx = zmq.Context.instance()
    link = SimLink(ctx, NAME, ports(cfg)["sim"])
    cp = CentralPlanner(cfg, ctx)
    while not cp.net.join_step(20):
        pass
    link.hello(cp.net.sent_counters())
    msg = None
    while msg is None:
        msg = link.recv(timeout_ms=20)
        if msg is None:
            cp.net.join_step(20)
    while True:
        if msg is None:
            msg = link.recv()
        if msg is None or msg["type"] == "stop":
            break
        if msg["type"] == "sense":
            link.send(cp.tick(msg))
        msg = None
    cp.net.close()
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
