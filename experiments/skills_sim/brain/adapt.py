"""protocol.WorldState (metres, many arms, many hands) -> e12v2's core WorldState (cm, one arm, one hand).

e12v2's pure pieces (decision text and distance bands, the plan validator, done checks, the change
detector) were written in cm against their own WorldState. Converting once, at the protocol boundary,
keeps them unchanged: nothing past this module sees a metre, nothing before it sees a cm.
"""
from __future__ import annotations

from e12v2.core.data import ArmObs, Geometry, HandObs, Obj, Place, WorldState

from ..protocol import ArmSpec
from ..protocol import WorldState as RobotWorld

CM = 100.0


def geometry(arm: ArmSpec, lookahead_s: float) -> Geometry:
    """The numbers e12v2's generic code needs, from the manifest. Of the parts reused, only the change
    detector reads one (`low_cm`: objects near a gripper this low may shift); the rest fill the type
    with the arm's workspace and conservative constants. `lookahead_s` is how far ahead a hand's
    velocity is projected (one brain tick), since e12v2 reads vx/vy as "where it is one tick on"."""
    (x0, x1), (y0, y1), (z0, z1) = arm.workspace
    width = arm.gripper.max_width if arm.gripper else 0.0
    return Geometry(travel_z=arm.travel_z * CM, low_z=z0 * CM, low_cm=(z0 + 0.05) * CM, finger_cm=max(width * CM, 2.0),
                    tip_cm=1.5, hand_stop_cm=8.0, hand_slow_cm=20.0, slow_speed=1.0,
                    workspace=(x0 * CM, x1 * CM, y0 * CM, y1 * CM, z0 * CM, z1 * CM),
                    ticks_per_s=max(1, round(1 / lookahead_s)))


def _where(where: str, arm: str) -> str:
    """e12v2 has one gripper, called "gripper"; another arm's gripper stays "gripper:<id>"."""
    return "gripper" if where == f"gripper:{arm}" else where


def _number(label: str | None) -> int | None:
    return int(label) if label and label.isdigit() else None


def core_state(ws: RobotWorld, arm: str, geom: Geometry, lookahead_s: float) -> WorldState:
    """The world as e12v2 reads it, for the arm being driven. Of several hands, the one nearest that
    arm's gripper is the one e12v2 reasons about."""
    a = ws.arms[arm]
    objs = {o.id: Obj(o.id, o.name, "", _number(o.label), f"{o.size * CM:.0f} cm" if o.size else "",
                      o.x * CM, o.y * CM, o.z * CM, _where(o.where, arm))
            for o in ws.objects.values()}
    places = {p.id: Place(p.id, p.name, p.kind, p.x * CM, p.y * CM, p.capacity, tuple(p.members), p.radius * CM)
              for p in ws.places.values()}
    hand = None
    if ws.hands:
        h = min(ws.hands, key=lambda h: (h.x - a.x) ** 2 + (h.y - a.y) ** 2 + (h.z - a.z) ** 2)
        k = CM * lookahead_s
        hand = HandObs(h.x * CM, h.y * CM, h.z * CM, h.vx * k, h.vy * k, h.held_out)
    return WorldState(ws.t, objs, places, hand, ArmObs(a.x * CM, a.y * CM, a.z * CM, a.holding), geom)


def names(ws: RobotWorld) -> dict[str, str]:
    """id -> how people refer to it, for every object and place."""
    return {**{p.id: p.name for p in ws.places.values()}, **{o.id: o.name for o in ws.objects.values()}}


def xy_cm(xy: tuple[float, float] | None) -> tuple[float, float] | None:
    return None if xy is None else (xy[0] * CM, xy[1] * CM)
