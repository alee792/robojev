"""The safety filter: code that clips every command, independently of every model.

  - STOP (button or typed "stop") is handled by the harness before anything reaches here.
  - Workspace: targets are clipped to the workspace box.
  - Hands: a hand within `hand_slow_cm` caps the speed at slow; no step may move toward a hand closer
    than `hand_stop_cm` (the code stop distance: it prevents a collision; keeping clear is the
    right-now answer's job).
  - Constraints: never grasp or fingertip-push a dont_touch object, never lower the gripper within its
    footprint of one, never release an object inside a place a keep_out_of constraint forbids.
Returns the (possibly clipped) command, or None and the reason it was refused.
"""

from __future__ import annotations

import math

from .data import Command, WorldState, dxy


class SafetyFilter:
    def __init__(self):
        self.refusals = 0
        self.floor_stops = 0
        self.dont_touch: set = set()
        self.keep_out: list = []

    def set_constraints(self, plan: dict | None):
        from .plan import forbidden, keep_out
        self.dont_touch = forbidden(plan) if plan else set()
        self.keep_out = keep_out(plan) if plan else []

    def clip(self, state: WorldState, cmd: Command | None, normal_speed: float) -> tuple[Command | None, str | None]:
        if cmd is None or cmd.kind == "wait":
            return cmd, None
        g, arm = state.geometry, state.arm
        if cmd.kind == "grip":
            for f in self.dont_touch:
                o = state.objects.get(f)
                if o and o.where != "gripper" and dxy((arm.x, arm.y), (o.x, o.y)) < g.finger_cm and arm.z < g.low_cm:
                    return self._refuse(f"would touch {o.name}, which must not be touched")
            return cmd, None
        if cmd.kind == "release":
            held = arm.holding
            for ko, kp in self.keep_out:
                if held == ko:
                    p = state.places.get(kp)
                    members = p.members if p is not None and p.kind == "group" else (kp,)
                    for m in members:
                        q = state.places.get(m)
                        if q is not None and dxy((arm.x, arm.y), (q.x, q.y)) <= q.radius:
                            return self._refuse(f"would put {state.objects[held].name} into {q.name}, which a constraint forbids")
            return cmd, None
        x0, x1, y0, y1, z0, z1 = g.workspace
        tx, ty, tz = cmd.target
        tx, ty, tz = min(x1, max(x0, tx)), min(y1, max(y0, ty)), min(z1, max(z0, tz))
        v = cmd.speed or normal_speed
        h = state.hand
        cur = (arm.x, arm.y, arm.z)
        if h is not None and math.dist(cur, (h.x, h.y, h.z)) < g.hand_slow_cm:
            v = min(v, g.slow_speed)
        dx, dy = tx - arm.x, ty - arm.y
        dh = math.hypot(dx, dy)
        s = 1.0 if dh <= v else v / dh
        dz = max(-3.0, min(3.0, tz - arm.z))
        new = (arm.x + dx * s, arm.y + dy * s, arm.z + dz)
        if h is not None:
            hp = (h.x, h.y, h.z)
            dn, dc = math.dist(new, hp), math.dist(cur, hp)
            if dn < dc and dn < g.hand_stop_cm:
                self.floor_stops += 1
                return self._refuse("a person's hand is too close (code stop distance)")
        if new[2] < g.low_cm:
            foot = g.tip_cm if cmd.fingertip else g.finger_cm
            for f in self.dont_touch:
                o = state.objects.get(f)
                if o and o.where != "gripper" and dxy(new, (o.x, o.y)) < foot:
                    return self._refuse(f"would touch {o.name}, which must not be touched")
            if cmd.drag in self.dont_touch:
                return self._refuse("would push an object that must not be touched")
        return Command("move", new, v, cmd.fingertip, cmd.drag), None

    def _refuse(self, why: str):
        self.refusals += 1
        return None, why
