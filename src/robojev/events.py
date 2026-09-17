"""Change detection: fire a Jev request only when something material changed.

At 10 Hz most answers arrive after the next tick has already started (08 §E2), so a purely clocked
loop spends its whole token budget re-asking about a scene that has not moved. The tick itself stays
at 10 Hz -- compose, command and log every tick -- but a request is sent only when the situation the
last *sent* request described has actually changed, or when `max_silence_s` has elapsed.

What counts as material (everything else is deliberately ignored, especially anything that ticks by
itself like primitive age or the ladder):
  * an entity appeared or disappeared, was relabelled, or moved more than `move_m`
  * the arm's phase facts changed: gripper_state, holding_label, above_label,
    primitive name / subject / status / last_result
  * the standing orders or the user's task text changed
  * the set of primitive keys offered to Jev changed
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Snapshot:
    """What the last sent request described. Entities are keyed by track id, not label."""
    entities: dict = field(default_factory=dict)      # id -> (label, x, y, z)
    phase: tuple = ()
    orders: tuple = ()
    task: str = ""
    offered: tuple = ()


def phase_facts(world) -> tuple:
    """The arm-phase part of the signature. `prim.age_s` is excluded on purpose: it changes every
    tick and would make every tick look material."""
    p = world.prim or {}
    return (world.gripper_state, world.holding_label, world.above_label,
            p.get("name"), p.get("subject"), p.get("status"), p.get("last_result"))


def snapshot_of(world, offered_keys) -> Snapshot:
    return Snapshot(
        entities={e.id: (e.label, float(e.xyz[0]), float(e.xyz[1]), float(e.xyz[2])) for e in world.entities},
        phase=phase_facts(world),
        orders=tuple(world.orders or ()),
        task=world.user_task or "",
        offered=tuple(sorted(offered_keys or ())),
    )


class ChangeDetector:
    """Decides whether this tick deserves a request. Only an actually *sent* request moves the
    baseline, so a tick skipped for any other reason (in-flight cap, pause) leaves the pending
    change pending."""

    def __init__(self, move_m: float = 0.02, max_silence_s: float = 1.0):
        self.move_m = move_m
        self.max_silence_s = max_silence_s
        self.last: Snapshot | None = None
        self.last_sent_t: float | None = None
        self.retry = False          # the last request failed or was dropped stale: ask again at once
        self._pending: Snapshot | None = None

    def should_ask(self, world, offered_keys, now: float) -> tuple[bool, str]:
        snap = snapshot_of(world, offered_keys)
        self._pending = snap
        if self.last is None or self.last_sent_t is None:
            return True, "first request"
        if self.retry:
            # a sent request that came back an error, a timeout or too stale left the situation
            # unanswered: re-ask on the next tick instead of waiting out max_silence_s
            return True, "retry after failed request"
        why = self._diff(self.last, snap)
        if why:
            return True, why
        if now - self.last_sent_t >= self.max_silence_s:
            return True, "max silence"
        return False, "no change"

    def mark_sent(self, now: float) -> None:
        if self._pending is not None:
            self.last = self._pending
        self.last_sent_t = now
        self.retry = False

    def _diff(self, a: Snapshot, b: Snapshot) -> str | None:
        gone = set(a.entities) - set(b.entities)
        if gone:
            return f"entity disappeared: {sorted(gone)[0]}"
        new = set(b.entities) - set(a.entities)
        if new:
            return f"entity appeared: {sorted(new)[0]}"
        for eid, (label, x, y, z) in b.entities.items():
            olabel, ox, oy, oz = a.entities[eid]
            if label != olabel:
                return f"entity relabelled: {olabel} -> {label}"
            if math.dist((x, y, z), (ox, oy, oz)) > self.move_m:
                return f"entity moved: {label}"
        if a.phase != b.phase:
            return "phase facts changed"
        if a.orders != b.orders:
            return "orders changed"
        if a.task != b.task:
            return "task changed"
        if a.offered != b.offered:
            return "offered primitives changed"
        return None
