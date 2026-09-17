"""Object permanence and stable names. Code remembers; Jev is never asked to.

Entities keep a letter suffix for life ("object A"), an EMA-smoothed position in base frame,
last_seen, and an in/out-of-view status. Names can be overridden by the operator ("A" -> "paper cup").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from robojev.perception.detect import Detection


def letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


@dataclass
class Entity:
    id: str                      # "A"
    xyz: np.ndarray              # base frame, footprint centre on the table
    height: float
    width: float
    color: str
    first_seen: float
    last_seen: float
    seen_count: int = 1
    history: list = field(default_factory=list)   # (t, x, y) for motion estimate
    name: str | None = None      # operator override, e.g. "paper cup"

    def kind(self) -> str:
        """A shape-based guess; a depth camera cannot know what a thing is, only its silhouette."""
        h, w = self.height, self.width
        if h >= 0.06 and w <= 0.13 and h > 0.8 * w:
            return "cup-like object"
        if h < 0.04 and w >= 0.08:
            return "flat object"
        if h < 0.06 and w < 0.08:
            return "small object"
        return "boxy object"

    def label(self) -> str:
        return f"{self.name} {self.id}" if self.name else f"{self.color} {self.kind()} {self.id}"

    def describe(self) -> str:
        shape = {"cup-like object": "upright, taller than wide, like a cup, can or bottle",
                 "flat object": "flat and wide, like a phone, book or pad",
                 "small object": "small, like a block or ball"}.get(self.kind(), "box-shaped")
        return f"{self.color}, {self.height*100:.0f} cm tall, {self.width*100:.0f} cm wide; {shape}"

    def velocity(self, window_s: float = 1.0) -> float:
        h = [p for p in self.history if p[0] > self.last_seen - window_s]
        if len(h) < 2:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = h[0], h[-1]
        dt = t1 - t0
        return float(np.hypot(x1 - x0, y1 - y0) / dt) if dt > 0.2 else 0.0


class Tracker:
    def __init__(self, match_radius: float = 0.06, ema: float = 0.5, ttl_s: float = 30.0, out_of_view_s: float = 1.0):
        self.match_radius, self.ema, self.ttl, self.oov = match_radius, ema, ttl_s, out_of_view_s
        self.entities: dict[str, Entity] = {}
        self._n = 0
        self.names: dict[str, str] = {}

    def update(self, dets: list[Detection], now: float | None = None) -> None:
        now = now or time.time()
        unmatched = list(dets)
        # greedy nearest matching
        for e in sorted(self.entities.values(), key=lambda e: -e.seen_count):
            if not unmatched:
                break
            d = min(unmatched, key=lambda d: np.hypot(d.base_xyz[0] - e.xyz[0], d.base_xyz[1] - e.xyz[1]))
            if np.hypot(d.base_xyz[0] - e.xyz[0], d.base_xyz[1] - e.xyz[1]) <= self.match_radius:
                unmatched.remove(d)
                a = self.ema
                e.xyz = a * np.asarray(d.base_xyz) + (1 - a) * e.xyz
                e.height = a * d.height + (1 - a) * e.height
                e.width = a * d.width + (1 - a) * e.width
                e.color = d.color_name if e.seen_count < 5 else e.color
                e.last_seen, e.seen_count = now, e.seen_count + 1
                e.history.append((now, float(e.xyz[0]), float(e.xyz[1])))
                e.history = e.history[-40:]
        for d in unmatched:
            eid = letter(self._n); self._n += 1
            self.entities[eid] = Entity(eid, np.asarray(d.base_xyz, float), d.height, d.width, d.color_name, now, now,
                                        history=[(now, d.base_xyz[0], d.base_xyz[1])], name=self.names.get(eid))
        for eid in [k for k, e in self.entities.items() if now - e.last_seen > self.ttl]:
            del self.entities[eid]

    def set_name(self, eid: str, name: str | None):
        self.names[eid] = name
        if eid in self.entities:
            self.entities[eid].name = name

    def in_view(self, e: Entity, now: float) -> bool:
        return now - e.last_seen < self.oov

    def stable(self, min_seen: int = 3) -> list[Entity]:
        return [e for e in self.entities.values() if e.seen_count >= min_seen]
