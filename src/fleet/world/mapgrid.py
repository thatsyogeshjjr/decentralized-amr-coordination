"""Warehouse map: ASCII layout -> traversability graph G = (V, E)  (design doc §10.5).

Legend
  '#'  storage rack (impassable)
  '.'  aisle floor
  'P'  pickup station        'D'  drop-off / packing station
  'C'  charging station      'R'  rest / parking alcove   '0'-'9' robot start cell (aisle)

Cell (r, c) maps to world x = c * cell, y = (H - 1 - r) * cell, so the first text line is the
north (+y) edge of the warehouse.  Nodes are integer ids r * W + c.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

PASSABLE = set(".PDCR0123456789")
STATION_KINDS = {"P": "pickup", "D": "drop", "C": "charger", "R": "park"}


@dataclass
class WarehouseMap:
    rows: list[str]
    cell: float = 1.5
    H: int = 0
    W: int = 0
    pos: dict[int, tuple[float, float]] = field(default_factory=dict)
    kind: dict[int, str] = field(default_factory=dict)
    adj: dict[int, list[int]] = field(default_factory=dict)
    starts: dict[int, int] = field(default_factory=dict)  # robot index -> node
    racks: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str, cell: float = 1.5) -> "WarehouseMap":
        rows = [ln.rstrip() for ln in text.strip("\n").splitlines() if ln.strip() and not ln.lstrip().startswith(";")]
        W = max(len(r) for r in rows)
        rows = [r.ljust(W, "#") for r in rows]
        m = cls(rows=rows, cell=cell, H=len(rows), W=W)
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch in PASSABLE:
                    n = r * W + c
                    m.pos[n] = (c * cell, (m.H - 1 - r) * cell)
                    m.kind[n] = STATION_KINDS.get(ch, "aisle")
                    if ch.isdigit():
                        m.starts[int(ch)] = n
                elif ch == "#":
                    m.racks.append((r, c))
                else:
                    raise ValueError(f"unknown map symbol {ch!r} at row {r} col {c}")
        for n in m.pos:
            r, c = divmod(n, W)
            nb = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m.H and 0 <= cc < W and (rr * W + cc) in m.pos:
                    nb.append(rr * W + cc)
            m.adj[n] = nb
        return m

    @classmethod
    def load(cls, path: str | Path, cell: float = 1.5) -> "WarehouseMap":
        return cls.from_text(Path(path).read_text(), cell)

    # ------------------------------------------------------------------ queries
    @property
    def nodes(self) -> list[int]:
        return sorted(self.pos)

    def rc(self, n: int) -> tuple[int, int]:
        return divmod(n, self.W)

    def stations(self, kind: str) -> list[int]:
        return sorted(n for n, k in self.kind.items() if k == kind)

    def rack_world(self, r: int, c: int) -> tuple[float, float]:
        return c * self.cell, (self.H - 1 - r) * self.cell

    def edges(self) -> list[tuple[int, int]]:
        return [(u, v) for u in self.nodes for v in self.adj[u] if u < v]

    def nearest_node(self, x: float, y: float) -> int:
        c = int(round(x / self.cell))
        r = self.H - 1 - int(round(y / self.cell))
        n = r * self.W + c
        if n in self.pos and 0 <= r < self.H and 0 <= c < self.W:
            return n
        return min(self.pos, key=lambda k: (self.pos[k][0] - x) ** 2 + (self.pos[k][1] - y) ** 2)

    def bfs_dist(self, goal: int, blocked: set[int] | frozenset[int] = frozenset()) -> dict[int, int]:
        """Hop distance from every reachable node to `goal`, ignoring `blocked` nodes."""
        if goal in blocked:
            return {}
        dist = {goal: 0}
        q = deque([goal])
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in dist and v not in blocked:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def extent(self) -> tuple[float, float]:
        return (self.W - 1) * self.cell, (self.H - 1) * self.cell
