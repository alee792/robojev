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
    dims: list = field(default_factory=list)      # (h, w, color) per sighting until frozen
    frozen: bool = False         # description frozen after CONFIRM sightings (labels must not flicker)
    last_speed: float = 0.0      # speed at the last sighting: a thing that was moving when it vanished has left

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

    def velocity(self, window_s: float = 1.5) -> float:
        """Speed from the displacement between the medians of the older and newer halves of the
        recent history, so two cameras disagreeing by 3 cm does not read as motion."""
        h = [p for p in self.history if p[0] > self.last_seen - window_s]
        if len(h) < 6:
            return 0.0
        a, b = h[: len(h) // 2], h[len(h) // 2:]
        ax, ay = np.median([p[1] for p in a]), np.median([p[2] for p in a])
        bx, by = np.median([p[1] for p in b]), np.median([p[2] for p in b])
        dt = np.median([p[0] for p in b]) - np.median([p[0] for p in a])
        d = float(np.hypot(bx - ax, by - ay))
        return d / dt if dt > 0.3 and d > 0.03 else 0.0


CONFIRM = 10          # sightings before an entity's description is frozen
MAX_NEW_WIDTH = 0.16  # m: wider detections are merged neighbours, not new objects (tabletop scale)
STATIC_TTL = 180.0    # s: a static object out of view (occluded by the arm) stays remembered this long
UNCONFIRMED_TTL = 2.0  # s: an entity with fewer than CONFIRM sightings that stops being seen is a phantom


class Tracker:
    def __init__(self, match_radius: float = 0.07, ema: float = 0.5, ttl_s: float = 30.0, out_of_view_s: float = 1.0):
        self.match_radius, self.ema, self.ttl, self.oov = match_radius, ema, ttl_s, out_of_view_s
        self.entities: dict[str, Entity] = {}
        self._n = 0
        self.names: dict[str, str] = {}

    def update(self, dets: list[Detection], now: float | None = None, carried_xy=None) -> None:
        """`carried_xy`: the gripper position while it holds something; detections there are the
        carried object leaking past the mask and must not become new tracks (run 6: a phantom 2 cm
        from the gripper made the policy flee into a corner)."""
        now = now or time.time()
        unmatched = list(dets)
        # greedy nearest matching
        for e in sorted(self.entities.values(), key=lambda e: -e.seen_count):
            if not unmatched:
                break
            # match on position AND height, so a flat object sliding under a tall one's radius
            # cannot drag the tall one's track (sim run 3)
            cands = [d for d in unmatched if abs(d.height - e.height) < 0.04 or not e.frozen and abs(d.height - e.height) < 0.06]
            if not cands:
                continue
            d = min(cands, key=lambda d: np.hypot(d.base_xyz[0] - e.xyz[0], d.base_xyz[1] - e.xyz[1]))
            if np.hypot(d.base_xyz[0] - e.xyz[0], d.base_xyz[1] - e.xyz[1]) <= self.match_radius:
                unmatched.remove(d)
                a = self.ema
                if (e.frozen and d.width > 1.6 * e.width) or d.partial:
                    a = 0.1   # a merged blob, or one cut by the image border, has a biased centre: barely trust it
                e.xyz = a * np.asarray(d.base_xyz) + (1 - a) * e.xyz
                if not e.frozen:
                    e.dims.append((d.height, d.width, d.color_name))
                    hs, ws, cs = zip(*e.dims)
                    e.height, e.width = float(np.median(hs)), float(np.median(ws))
                    e.color = max(set(cs), key=cs.count)
                    if len(e.dims) >= CONFIRM:
                        e.frozen = True
                e.last_seen, e.seen_count = now, e.seen_count + 1
                e.history.append((now, float(e.xyz[0]), float(e.xyz[1])))
                e.history = e.history[-40:]
                e.last_speed = e.velocity()
        for d in unmatched:
            if d.width > MAX_NEW_WIDTH or d.partial:
                continue   # merged blobs and border-cut blobs must not become objects
            if carried_xy is not None and np.hypot(d.base_xyz[0] - carried_xy[0], d.base_xyz[1] - carried_xy[1]) < 0.15:
                continue
            eid = letter(self._n); self._n += 1
            self.entities[eid] = Entity(eid, np.asarray(d.base_xyz, float), d.height, d.width, d.color_name, now, now,
                                        history=[(now, d.base_xyz[0], d.base_xyz[1])], name=self.names.get(eid),
                                        dims=[(d.height, d.width, d.color_name)])
        # merge duplicates (two cameras, or a split blob): the younger one folds into the older
        ents = sorted(self.entities.values(), key=lambda e: e.first_seen)
        for i, a_ in enumerate(ents):
            for b_ in ents[i + 1:]:
                if b_.id not in self.entities or a_.id not in self.entities:
                    continue
                if np.hypot(*(a_.xyz[:2] - b_.xyz[:2])) < self.match_radius and abs(a_.height - b_.height) < 0.04:
                    a_.seen_count += b_.seen_count
                    a_.last_seen = max(a_.last_seen, b_.last_seen)
                    del self.entities[b_.id]
        # object permanence: a static object the arm is occluding stays remembered for a long time
        # (Doom's remembered[]); something that was moving when last seen has probably left
        for eid in [k for k, e in self.entities.items()
                    if now - e.last_seen > (UNCONFIRMED_TTL if e.seen_count < CONFIRM else (self.ttl if e.last_speed > 0.02 else STATIC_TTL))]:
            del self.entities[eid]

    def pin(self, label: str, xy, now: float | None = None) -> None:
        """Code-side object permanence for a carried object: its track follows the gripper."""
        now = now or time.time()
        for e in self.entities.values():
            if e.label() == label:
                e.xyz = np.array([xy[0], xy[1], e.xyz[2]], float)
                e.last_seen = now
                e.history.append((now, float(xy[0]), float(xy[1]))); e.history = e.history[-40:]
                return

    def set_name(self, eid: str, name: str | None):
        self.names[eid] = name
        if eid in self.entities:
            self.entities[eid].name = name

    def in_view(self, e: Entity, now: float) -> bool:
        return now - e.last_seen < self.oov

    def stable(self, min_seen: int = CONFIRM) -> list[Entity]:
        """Confirmed tracks: enough sightings, spread over at least a second (a burst of fragments
        from one pose does not make an object)."""
        return [e for e in self.entities.values() if e.seen_count >= min_seen and e.last_seen - e.first_seen >= 1.0]
