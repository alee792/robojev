"""The change detector's "object gone": a taken block must become an event after the confirmation
ticks, and a two-tick perception drop-out must not. Found by the spelling battery: the count ran
against the previous frame's objects, which no longer listed the block after the first missing tick,
so "gone" never fired and a taken letter was noticed only when its step failed to start."""
from __future__ import annotations

from robojev.brain.core.changes import ChangeDetector
from robojev.brain.core.data import ArmObs, Geometry, Obj, Place, WorldState

GEO = Geometry(travel_z=20, low_z=3, low_cm=8, finger_cm=4, tip_cm=2, hand_stop_cm=10, hand_slow_cm=25,
               slow_speed=1.0, workspace=(0, 60, -30, 30, 0, 25), ticks_per_s=10)
ARM = ArmObs(x=30, y=0, z=20, holding=None)      # high and far from every block: nothing is "the robot's doing"


def block(oid: str, x: float, y: float, where: str = "table") -> Obj:
    return Obj(id=oid, name=f"the letter {oid[-1]}", colour="wood", number=None, size="small", x=x, y=y, z=2, where=where)


def state(t: int, *objs: Obj) -> WorldState:
    return WorldState(t=t, objects={o.id: o for o in objs}, places={"slot_0": Place("slot_0", "position 1", "slot", 40, 10, 1)},
                      hand=None, arm=ARM, geometry=GEO)


def test_a_block_taken_away_is_reported_gone_after_three_ticks():
    d = ChangeDetector()
    a, b = block("b_A", 10, 0), block("b_B", 10, 12, where="slot_0")
    assert d.update(state(0, a, b)) == []
    gone = []
    for t in range(1, 6):
        gone += [c for c in d.update(state(t, a)) if c["what"] == "object_gone"]
    assert gone == [{"what": "object_gone", "object": "b_B", "from": "slot_0"}], gone


def test_a_two_tick_perception_dropout_is_not_a_gone():
    d = ChangeDetector()
    a, b = block("b_A", 10, 0), block("b_B", 10, 12)
    d.update(state(0, a, b))
    out = d.update(state(1, a)) + d.update(state(2, a)) + d.update(state(3, a, b))
    assert [c["what"] for c in out] == []


def test_a_block_that_returns_after_being_gone_appears_again():
    d = ChangeDetector()
    a, b = block("b_A", 10, 0), block("b_B", 10, 12)
    d.update(state(0, a, b))
    for t in range(1, 4):
        d.update(state(t, a))
    out = d.update(state(4, a, b))
    assert [c["what"] for c in out] == ["object_appeared"]
    # and it is anchored again, so a second disappearance is counted too
    for t in range(5, 8):
        out = d.update(state(t, a))
    assert [c["what"] for c in out] == ["object_gone"]


def test_a_block_that_appeared_and_then_vanishes_is_counted():
    d = ChangeDetector()
    a = block("b_A", 10, 0)
    d.update(state(0, a))
    new = block("b_N", 20, 0)
    assert [c["what"] for c in d.update(state(1, a, new))] == ["object_appeared"]
    out = []
    for t in range(2, 6):
        out += d.update(state(t, a))
    assert [c["what"] for c in out] == ["object_gone"]
