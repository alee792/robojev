"""Changes the robot didn't cause (docs/v2.md "Rules that hold everywhere").

Code keeps a simple expectation of what the robot itself moves: the held object travels with the
gripper, an object the gripper just released or pushed moves, and objects within a few cm of a low
gripper may shift. Everything else that moves is an outside change and becomes a scene_change event.

Positions are compared with where each object was last known to rest (not the previous frame), and
a small shift must persist for `confirm_ticks` frames, so perception jitter does not become events.

Hands are reported on transitions only (appeared, gone, its distance band from the gripper changed,
it started or stopped coming closer, it entered or left the arm's path, it was held out or withdrawn),
so a hand costs a handful of decisions, not one per tick.
"""

from __future__ import annotations

from .data import WorldState, dxy, seg_dist

MOVE_CM = 2.0          # smaller moves are perception jitter
NEAR_GRIPPER_CM = 5.0  # objects this close to a low gripper may shift when it grasps or releases


def hand_band(d: float) -> str:
    if d < 10:
        return "very close"
    if d < 25:
        return "near"
    if d < 45:
        return "mid-range"
    return "far"


def hand_facts(state: WorldState, path_to: tuple | None, prev_d: float | None = None) -> dict | None:
    """Code-computed facts about the hand relative to the gripper and the arm's remaining path."""
    h, a = state.hand, state.arm
    if h is None:
        return None
    d = ((h.x - a.x) ** 2 + (h.y - a.y) ** 2 + (h.z - a.z) ** 2) ** 0.5
    nxt = ((h.x + h.vx - a.x) ** 2 + (h.y + h.vy - a.y) ** 2 + (h.z - a.z) ** 2) ** 0.5
    in_path = path_to is not None and seg_dist((h.x, h.y), (a.x, a.y), path_to) < 8.0 and dxy((a.x, a.y), path_to) > 1.0
    return {"distance_cm": round(d), "band": hand_band(d), "approaching": nxt < d - 0.5,
            "in_path": bool(in_path), "held_out": h.held_out}


class ChangeDetector:
    def __init__(self, confirm_ticks: int = 2):
        self.prev: WorldState | None = None
        self.hand_key = None
        self.missing: dict[str, int] = {}
        self.ref: dict[str, tuple] = {}       # where each object was last known to rest (x, y, where)
        self.cand: dict[str, int] = {}        # ticks an object has been away from its reference
        self.confirm = confirm_ticks

    def update(self, state: WorldState, path_to: tuple | None = None, robot_touched: set | None = None) -> list[dict]:
        """-> changes, each {"what": "object_moved" | "object_gone" | "object_appeared" | "hand", ...}."""
        prev, out = self.prev, []
        self.prev = state
        if prev is None:
            self.hand_key = self._hkey(state, path_to)
            self.ref = {o.id: (o.x, o.y, o.where) for o in state.objects.values()}
            return out
        touched = set(robot_touched or ())
        for arm in (prev.arm, state.arm):
            if arm.holding:
                touched.add(arm.holding)
        low = [a for a in (prev.arm, state.arm) if a.z < state.geometry.low_cm]
        for oid, o in state.objects.items():
            p = prev.objects.get(oid)
            if p is None:
                if oid not in self.missing:
                    out.append({"what": "object_appeared", "object": oid, "to": o.where})
                self.missing.pop(oid, None)
                continue
            self.missing.pop(oid, None)
            ref = self.ref.get(oid, (p.x, p.y, p.where))
            if oid in touched or o.where == "gripper" or \
                    any(dxy((a.x, a.y), (p.x, p.y)) < NEAR_GRIPPER_CM or dxy((a.x, a.y), (o.x, o.y)) < NEAR_GRIPPER_CM for a in low):
                self.ref[oid] = (o.x, o.y, o.where)          # the robot's own doing: expected, re-anchor
                self.cand.pop(oid, None)
                continue
            moved = dxy(ref[:2], (o.x, o.y))
            placed = ref[2] != o.where and "gripper" not in (ref[2], o.where)
            if moved > MOVE_CM or placed:
                self.cand[oid] = self.cand.get(oid, 0) + 1
                if placed or self.cand[oid] >= self.confirm:   # a change of place is certain; a small shift must persist
                    out.append({"what": "object_moved", "object": oid, "from": ref[2], "to": o.where,
                                "from_xy": (round(ref[0]), round(ref[1])), "to_xy": (round(o.x), round(o.y)), "moved_cm": round(moved)})
                    self.ref[oid] = (o.x, o.y, o.where)
                    self.cand.pop(oid, None)
            else:
                self.cand.pop(oid, None)
        for oid in prev.objects:
            if oid not in state.objects and oid not in touched:
                self.missing[oid] = self.missing.get(oid, 0) + 1
                if self.missing[oid] == 3:   # gone for 3 ticks, not a perception drop-out
                    out.append({"what": "object_gone", "object": oid, "from": prev.objects[oid].where})
        key = self._hkey(state, path_to)
        if key != self.hand_key:
            old = self.hand_key
            self.hand_key = key
            out.append({"what": "hand", "before": old, "now": key})
        return out

    @staticmethod
    def _hkey(state: WorldState, path_to):
        f = hand_facts(state, path_to)
        if f is None:
            return None
        return (f["band"], f["approaching"], f["in_path"], f["held_out"])
