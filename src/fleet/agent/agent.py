"""Robot agent process (one per AMR).  Runs entirely on local information plus peer messages.

    python -m fleet.agent.agent --config <resolved_config.json> --robot <index>
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time

import zmq

from fleet.common.config import ports, resolve
from fleet.world.mapgrid import WarehouseMap
from fleet.world.mjcf_builder import robot_name


def make_brain(cfg: dict, idx: int, name: str, wmap: WarehouseMap, ctx: zmq.Context):
    planner = cfg["fleet"]["planner"]
    if planner == "scripted":
        from fleet.agent.scripted import ScriptedBrain
        return ScriptedBrain(cfg, idx, name, wmap)
    from fleet.agent.brain import FleetBrain
    return FleetBrain(cfg, idx, name, wmap, ctx)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--robot", type=int, required=True)
    a = ap.parse_args(argv)
    cfg = json.loads(open(a.config).read())
    # startup watchdog: if initialisation/join stalls, dump every thread's stack to the agent log for diagnosis
    faulthandler.enable(file=sys.stdout)
    faulthandler.dump_traceback_later(90, repeat=False, file=sys.stdout)
    idx = a.robot
    name = robot_name(idx)
    wmap = WarehouseMap.load(resolve(cfg["world"]["map"]), cfg["world"]["cell"])
    ctx = zmq.Context.instance()
    from fleet.agent.simlink import SimLink
    link = SimLink(ctx, name, ports(cfg)["sim"])
    brain = make_brain(cfg, idx, name, wmap, ctx)
    # CONOPS phase 1 (fleet startup): join the peer network before reporting ready to the world clock, and keep
    # beaconing until the clock starts so that slower peers can complete their own join.
    join = getattr(brain, "join_step", lambda: True)
    t0 = time.time()
    while not join():
        if time.time() - t0 > 300:          # generous: startup can be slow when many agents import torch at once
            print(f"{name}: join timeout", flush=True)
            return 2
    faulthandler.cancel_dump_traceback_later()
    counters = getattr(getattr(brain, "net", None), "sent_counters", lambda: {})
    link.hello(counters())
    msg = None
    while msg is None:
        msg = link.recv(timeout_ms=20)
        if msg is None:
            join()   # keep beaconing so slower peers can finish their own join
    while True:
        if msg is None:
            msg = link.recv()
        if msg is None or msg["type"] == "stop":
            break
        if msg["type"] == "kill":
            os._exit(0)          # injected crash: no goodbye, no flush, sockets vanish
        if msg["type"] != "sense":
            continue
        out = brain.step(msg)
        if out.get("stats") is not None:
            out["stats"]["link"] = link.stats()
        wl, wr = out["wheels"]
        link.send({"type": "cmd", "tick": msg["tick"], "wl": wl, "wr": wr, "done": out.get("done", False),
                   "status": out.get("status", {}), "sent": out.get("sent", {}),
                   "task_events": out.get("task_events", []), "stats": out.get("stats")})
        msg = None
    if hasattr(brain, "close"):
        brain.close()
    link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
