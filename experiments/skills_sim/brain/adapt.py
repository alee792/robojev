"""protocol.WorldState (metres, many arms, many hands) -> e12v2's core WorldState (cm, one arm, one hand).

e12v2's pure pieces (decision text and bands, the plan validator, the change detector) were written in
cm against their own WorldState. Converting once at the protocol boundary keeps them unchanged; nothing
past this module sees a metre, and nothing before it sees a cm.
"""
from __future__ import annotations

from e12v2.core.data import ArmObs, Geometry, HandObs, Obj, Place, WorldState

from ..protocol import ArmSpec
from ..protocol import WorldState as PWorld

CM = 100.0
LOOKAHEAD_S = 0.2       # e12v2 reads a hand's vx/vy as "where it is one tick on"; one heartbeat here


def geometry(arm: ArmSpec, heartbeat_s: float = LOOKAHEAD_S) -> Geometry:
    """The numbers e12v2's generic code needs, from the manifest. Only `low_cm` is read by the parts
    reused here (the change detector: objects near a low gripper may shift); the rest fill the type."""
    (x0, x1), (y0, y1), (z0, z1) = arm.workspace
    width = arm.gripper.max_width if arm.gripper else 0.0
    return Geometry(travel_z=arm.travel_z * CM, low_z=z0 * CM, low_cm=(z0 + 0.05) * CM, finger_cm=max(width * CM, 2.0),
                    tip_cm=1.5, hand_stop_cm=8.0, hand_slow_cm=20.0, slow_speed=1.0,
                    workspace=(x0 * CM, x1 * CM, y0 * CM, y1 * CM, z0 * CM, z1 * CM), ticks_per_s=round(1 / heartbeat_s))


def _where(where: str, arm_id: str) -> str:
    return "gripper" if where == f"gripper:{arm_id}" else where


def core_state(ws: PWorld, arm_id: str, geom: Geometry) -> WorldState:
    a = ws.arms[arm_id]
    objs = {o.id: Obj(o.id, o.name, "", int(o.label) if o.label and o.label.isdigit() else None,
                      f"{o.size * CM:.0f} cm" if o.size else "", o.x * CM, o.y * CM, o.z * CM, _where(o.where, arm_id))
            for o in ws.objects.values()}
    places = {p.id: Place(p.id, p.name, p.kind, p.x * CM, p.y * CM, p.capacity, tuple(p.members), p.radius * CM)
              for p in ws.places.values()}
    hand = None
    if ws.hands:        # e12v2 reasons about one hand: the one nearest the gripper
        h = min(ws.hands, key=lambda h: (h.x - a.x) ** 2 + (h.y - a.y) ** 2 + (h.z - a.z) ** 2)
        hand = HandObs(h.x * CM, h.y * CM, h.z * CM, h.vx * CM * LOOKAHEAD_S, h.vy * CM * LOOKAHEAD_S, h.held_out)
    return WorldState(ws.t, objs, places, hand, ArmObs(a.x * CM, a.y * CM, a.z * CM, a.holding), geom)


def names(ws: PWorld) -> dict:
    return {**{p.id: p.name for p in ws.places.values()}, **{o.id: o.name for o in ws.objects.values()}}


def xy_cm(xy: tuple | None) -> tuple | None:
    return None if xy is None else (xy[0] * CM, xy[1] * CM)
