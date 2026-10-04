"""Goal sources for the motion/coordination layer.

GoalSequence: fixed list of goal cells per robot from the scenario (conflict scenarios, deadlock cases).
The task-allocation layer (task_manager.TaskManager) implements the same interface for pickup/delivery work.
"""
from __future__ import annotations

from fleet.world.mapgrid import WarehouseMap


class GoalSequence:
    def __init__(self, cfg: dict, name: str, m: WarehouseMap):
        self.goals = [r * m.W + c for r, c in cfg.get("goals", {}).get(name, [])]
        self.k = 0
        self.dwell = float(cfg["tasks"]["dwell"])
        self.reached_t: list[float] = []

    def current(self) -> int | None:
        return self.goals[self.k] if self.k < len(self.goals) else None

    def goal_key(self):
        return ("seq", self.k)

    def label(self) -> str:
        return f"goal {self.k + 1}/{len(self.goals)}" if self.current() is not None else "idle"

    def on_reached(self, now: float) -> float:
        """Called when stationary on the current goal. Returns dwell seconds."""
        return self.dwell

    def on_dwell_done(self, now: float) -> list[dict]:
        self.reached_t.append(round(now, 3))
        self.k += 1
        return []

    @property
    def finished(self) -> bool:
        return self.k >= len(self.goals)

    # task-layer hooks (no-ops for fixed goal sequences)
    def handle(self, msg: dict, now: float) -> None:
        pass

    def tick(self, now: float, ctx) -> None:
        pass

    def on_peer_lost(self, name: str, now: float) -> None:
        pass

    def release_current(self, now: float, reason: str) -> None:
        pass

    def carrying(self) -> bool:
        return False

    def task_id(self):
        return None

    def stats(self) -> dict:
        return {"reached": self.reached_t}


class CentralGoals:
    """BENCHMARK ONLY: goal source for `planner = "central"` - the task comes from the central reference plan."""

    def __init__(self, cfg: dict, name: str, m: WarehouseMap):
        self.name, self.dwell = name, float(cfg["tasks"]["dwell"])
        self.task: dict | None = None
        self._carrying = False
        self.completed: list[dict] = []
        self.events: list[dict] = []

    def set_plan(self, entry: dict) -> None:
        t = entry.get("task")
        if t is not None and (self.task is None or t["task_id"] != self.task["task_id"]) and not self._carrying:
            self.task = t

    def current(self) -> int | None:
        if self.task is None:
            return None
        return self.task["dropoff"] if self._carrying else self.task["pickup"]

    def goal_key(self):
        return (self.task["task_id"] if self.task else None, self._carrying)

    def label(self) -> str:
        return f"{self.task['task_id']}:{'drop' if self._carrying else 'pickup'}" if self.task else "idle"

    def on_reached(self, now: float) -> float:
        return self.dwell

    def drain_events(self) -> list[dict]:
        ev, self.events = self.events, []
        return ev

    def on_dwell_done(self, now: float) -> list[dict]:
        if self.task is None:
            return []
        if not self._carrying:
            self.events.append({"task_id": self.task["task_id"], "phase": "picked"})
            self._carrying = True
        else:
            self.events.append({"task_id": self.task["task_id"], "phase": "delivered"})
            self.completed.append({"task": self.task["task_id"], "t": round(now, 3)})
            self.task, self._carrying = None, False
        return self.drain_events()

    @property
    def finished(self) -> bool:
        return False

    def handle(self, msg: dict, now: float) -> None:
        pass

    def tick(self, now: float, ctx) -> None:
        pass

    def on_peer_lost(self, name: str, now: float) -> None:
        pass

    def release_current(self, now: float, reason: str) -> None:
        pass

    def carrying(self) -> bool:
        return self._carrying

    def task_id(self):
        return self.task["task_id"] if self.task else None

    def stats(self) -> dict:
        return {"completed": self.completed, "alloc_latency": [], "released": [], "reassigned_from_lost": []}
