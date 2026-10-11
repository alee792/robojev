"""K5: two WidowX AI followers in one MuJoCo scene behind one robot server (docs/widowx-pair.md):
the pair scene, the common frame, both arms picking at once in their own halves, the shared-zone
lock (a second arm held at its side line and released, a second requester backing off, the hold
surviving pause and resume), the hand-over through the zone, the arm-arm proximity trip with the
wrist footprint, STOP parking both. No rendering, no clock: deterministic, seconds per test."""
from __future__ import annotations

import math
import random

import pytest

from robojev import conformance as trials
from robojev.protocol import HandObs, RobotServer
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.base import ZoneLock
from robojev.robots.widowx_sim.server import ARM_GAP, KEEPOUT, PAIR_HOME_XY, TRAVEL_Z, SimRobotServer

pytestmark = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(),
                                reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")

A0, A1 = scene.ARM_0, scene.ARM_1
D = scene.PAIR_SPACING                                                       # 0.76
XC = D / 2                                                                   # the zone's centre line, in either frame
SIDE = XC - KEEPOUT                                                          # each arm's side line, own frame: 0.12
OWN_0 = scene.Block("block_1", (0.19, -0.05), yaw=0.4)                      # in arm_0's half
OWN_1 = scene.Block("block_2", (D - 0.19, 0.06), yaw=-0.3, rgba=(0.2, 0.4, 0.8, 1))   # in arm_1's half
TO_STAGE = {"object": "block_1", "place": "stage_1"}
TO_TRAY = {"object": "block_2", "place": "slot_3"}
BLUE = (0.2, 0.4, 0.8, 1)


def pair_server(blocks=None, **kw) -> SimRobotServer:
    srv = SimRobotServer(scene.pair(blocks if blocks is not None else [OWN_0, OWN_1], **kw), realtime=False)
    srv.events = []
    srv.subscribe(srv.events.append)
    return srv


def phase(srv, sid) -> str:
    return srv._runs[sid].phase


def in_phase(srv, sid, name):
    return lambda: phase(srv, sid) == name or srv.status(sid).state in ("done", "failed")


def waiting(srv, sid):
    return lambda: srv.status(sid).phase_text.startswith("waiting for the shared zone")


def arm_arm_contacts(srv) -> int:
    """Contacts between geoms of different arms in the last physics step."""
    m, d, mj = srv.model, srv.data, srv.mujoco

    def owner(g) -> str:
        return (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").split("/")[0]
    return sum(1 for i in range(d.ncon)
               if owner(d.contact[i].geom1).startswith("arm_") and owner(d.contact[i].geom2).startswith("arm_")
               and owner(d.contact[i].geom1) != owner(d.contact[i].geom2))


def ee_gap(srv) -> float:
    w = srv.world()
    a, b = w.arms[A0], w.arms[A1]
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def own_x(srv, arm) -> float:
    return srv.arms[arm].ee()[0]


# ---------------------------------------------------------------- scene, manifest, frames


def test_pair_scene_builds_two_prefixed_models_and_the_zone():
    import mujoco
    m = scene.build(scene.pair([OWN_0]))
    joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)}
    assert {"arm_0/joint_0", "arm_1/joint_5", "arm_0/left_carriage_joint", "arm_1/right_carriage_joint", "block_1_free"} <= joints
    assert m.nu == 14 and m.neq == 2 and m.opt.timestep == 0.002
    spec = scene.pair()
    z = spec.zone()
    assert (z.x0, z.x1) == pytest.approx((XC - 0.04, XC + 0.04)) and z.spot_xy(1) == pytest.approx((XC, 0.0))
    assert spec.half(A0) == (0.0, z.x0) and spec.half(A1) == (z.x1, D)
    assert spec.bases()[A1] == scene.BasePose(D, 0.0, math.pi)
    places = spec.places()
    assert set(places) == {f"slot_{i}" for i in range(5)} | {"tray", "stage_0", "stage_1", "stage", "zone_0", "zone_1", "zone_2", "shared_zone"}
    assert places["shared_zone"].kind == "zone" and places["shared_zone"].members == ("zone_0", "zone_1", "zone_2")
    assert places["zone_0"].kind == "slot" and places["zone_0"].name == "shared-zone spot 0" and places["zone_0"].y == pytest.approx(-0.10)
    assert (places["slot_0"].x, places["slot_0"].y) == pytest.approx((D - 0.11, -0.16)) and places["slot_0"].name == "tray slot 0 (the leftmost slot)"
    assert (places["stage_0"].x, places["stage_0"].y) == pytest.approx((0.11, 0.10)) and places["stage_0"].name == "staging spot 0 (the leftmost slot)"
    assert scene.pair(zone_width=None).zone() is None and scene.pair(zone_width=None).half(A0) == (0.0, XC)
    with pytest.raises(ValueError):
        scene.SceneSpec(arms=("a", "b", "c")).bases()


def test_too_close_a_spacing_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="spacing 0.65 m is too close.*wrist housings.*spacing >= 0.72 m"):
        SimRobotServer(scene.pair(spacing=0.65))
    SimRobotServer(scene.pair(spacing=0.65, zone_width=None))               # no zone, no lock: allowed


def test_manifest_and_world_speak_the_common_frame():
    srv = pair_server()
    assert isinstance(srv, RobotServer)
    man = srv.manifest()
    assert [a.id for a in man.arms] == [A0, A1] and all(a.base_frame == "arm_0/base_link" for a in man.arms)
    assert man.arms[0].workspace[0] == pytest.approx((0.10, 0.42)) and man.arms[0].workspace[1] == pytest.approx((-0.25, 0.25))
    assert man.arms[1].workspace[0] == pytest.approx((D - 0.42, D - 0.10)) and man.arms[1].workspace[1] == pytest.approx((-0.25, 0.25))
    assert "base at (0.76, 0.00) heading 180 deg" in man.arms[1].description
    w = srv.world()
    a0, a1 = w.arms[A0], w.arms[A1]
    assert (a0.x, a0.y, a0.z) == pytest.approx((PAIR_HOME_XY[0], 0.0, TRAVEL_Z), abs=2e-3)
    assert (a1.x, a1.y, a1.z) == pytest.approx((D - PAIR_HOME_XY[0], 0.0, TRAVEL_Z), abs=2e-3)
    assert a0.mode == a1.mode == "idle" and a0.gripper_width == pytest.approx(0.088, abs=1e-3)
    assert (w.objects["block_2"].x, w.objects["block_2"].y) == pytest.approx((D - 0.19, 0.06), abs=1e-3)
    assert w.objects["block_1"].where == w.objects["block_2"].where == "table"
    assert w.places["shared_zone"].name == "the shared zone (free)" and all(o.claimed_by is None for o in w.objects.values())


def test_each_arm_sees_its_own_frame_and_the_frames_agree_on_a_common_point():
    srv = pair_server()
    srv.sim_world.move("block_1", XC, 0.05, yaw=0.3)                       # a known common-frame point in the zone
    srv.run_for(0.1)
    w = srv.world()
    o = w.objects["block_1"]
    assert (o.x, o.y) == pytest.approx((XC, 0.05), abs=1e-3) and o.where == "shared_zone"
    p0, p1 = srv.arms[A0].obj("block_1"), srv.arms[A1].obj("block_1")
    assert (p0.x, p0.y) == pytest.approx((XC, 0.05), abs=1e-3) and p0.yaw == pytest.approx(0.3, abs=0.01)
    assert (p1.x, p1.y) == pytest.approx((XC, -0.05), abs=1e-3) and skills.wrap(p1.yaw - (0.3 - math.pi)) == pytest.approx(0.0, abs=0.01)
    assert srv.arms[A1].base.to_common((p1.x, p1.y)) == pytest.approx((o.x, o.y), abs=1e-9)
    for arm in (A0, A1):                                                    # both reach it, each in its own box
        assert srv.arms[arm].reachable((XC, 0.05, TRAVEL_Z))
        assert srv.precondition(arm, "pick_and_place", {"object": "block_1", "place": "zone_2"}) is None
        a = w.arms[arm]
        assert srv.arms[arm].base.to_common(srv.arms[arm].ee()) == pytest.approx((a.x, a.y, a.z), abs=1e-9)
    assert srv.arms[A1].place("stage_0") == pytest.approx((D - 0.11, -0.10)) and srv.arms[A0].place("slot_0") == pytest.approx((D - 0.11, -0.16))
    # a point in one half is in that arm's box only, and the refusal names it in the asking arm's frame
    assert srv.arms[A0].reachable((0.19, -0.05, TRAVEL_Z)) and not srv.arms[A1].reachable(srv.arms[A1].base.to_local((0.19, -0.05, TRAVEL_Z)))
    assert srv.precondition(A0, "pick_and_place", TO_TRAY).startswith(f"unreachable: block_2 at ({D - 0.19:.2f}, 0.06)")
    assert srv.precondition(A1, "pick_and_place", {"object": "block_2", "place": "stage_0"}) == f"unreachable: stage_0 at ({D - 0.11:.2f}, -0.10) is outside the arm's workspace"
    srv.set_hands([HandObs(0.30, 0.10, 0.05, vx=0.1)])
    h = srv.arms[A1].hands()[0]
    assert (h.x, h.y, h.z, h.vx, h.vy) == pytest.approx((D - 0.30, -0.10, 0.05, -0.1, 0.0), abs=1e-9)
    assert srv.arms[A0].hands()[0] == srv.world().hands[0]


def test_one_arm_scene_is_unchanged():
    srv = SimRobotServer(scene.SceneSpec(blocks=[scene.Block("block_1", (0.30, -0.08), yaw=0.5)]))
    arm = srv.manifest().arms[0]
    assert len(srv.manifest().arms) == 1 and arm.base_frame == "base_link"
    assert arm.workspace == ((0.15, 0.42), (-0.25, 0.25), (scene.TABLE_Z + 0.004, 0.14))
    assert srv.arms[A0].prefix == "" and srv.arms[A0].base is scene.ORIGIN and srv.zone is None
    assert set(srv.world().places) == {f"slot_{i}" for i in range(5)} | {"tray"}
    assert (srv.world().arms[A0].x, srv.world().arms[A0].y) == pytest.approx((0.22, 0.0), abs=2e-3)


# ---------------------------------------------------------------- the lock's rules, without physics


def test_zone_lock_rules():
    lock = ZoneLock()
    no = {A0: False, A1: False}
    assert lock.step(dict(no), dict(no), dict(no)) == {} and lock.holder is None
    # both ask in the same tick from behind their side lines: arm order; the other waits
    assert lock.step(dict(no), {A0: True, A1: True}, dict(no)) == {A1: "held by arm_0"} and lock.holder == A0 and lock.queue == [A1]
    assert lock.step({A0: True, A1: False}, {A0: False, A1: True}, {A0: True, A1: False}) == {A1: "held by arm_0"}
    # arm_0 left the strip but is still beyond its side line: the waiter is told, not granted
    assert lock.step(dict(no), {A0: False, A1: True}, {A0: True, A1: False}) == {A1: "arm_0 is still within reach of it"} and lock.holder is None
    assert lock.yielding == set()
    assert lock.step(dict(no), {A0: False, A1: True}, dict(no)) == {} and lock.holder == A1       # arm_0 is back: granted
    assert lock.step({A0: False, A1: True}, {A0: True, A1: False}, {A0: False, A1: True}) == {A0: "held by arm_1"}
    assert lock.step({A0: False, A1: True}, dict(no), {A0: False, A1: True}) == {} and lock.queue == []   # arm_0 stopped asking
    assert lock.step(dict(no), dict(no), dict(no)) == {} and lock.holder is None                 # arm_1 left
    # both reached the strip's edge, both beyond their side lines: the first asked keeps its place, the second backs off
    assert lock.step(dict(no), {A0: True, A1: True}, {A0: True, A1: True}) == {A0: "arm_1 is still within reach of it", A1: "arm_0 is still within reach of it"}
    assert lock.holder is None and lock.queue == [A0, A1] and lock.yielding == {A1}
    assert lock.step(dict(no), {A0: True, A1: True}, {A0: True, A1: False}) == {A1: "held by arm_0"} and lock.holder == A0
    assert lock.yielding == set()
    lock = ZoneLock()
    lock.step(dict(no), {A0: True, A1: False}, dict(no))
    assert lock.holder == A0
    assert lock.step(dict(no), dict(no), dict(no)) == {} and lock.holder is None                 # granted but never went in: released
    lock.step({A0: False, A1: True}, dict(no), {A0: False, A1: True})
    assert lock.holder == A1                                                                      # found inside: it has the lock


# ---------------------------------------------------------------- both arms at once


def test_both_arms_pick_at_once_in_their_halves_without_contact():
    srv = pair_server()
    s0 = srv.start(A0, "pick_and_place", TO_STAGE)
    s1 = srv.start(A1, "pick_and_place", TO_TRAY)
    seen, contacts, gaps = [], [], []

    def watch():
        seen.append((srv.status(s0).state, srv.status(s1).state))
        contacts.append(arm_arm_contacts(srv))
        gaps.append(ee_gap(srv))
        return srv._runs[s0].finished and srv._runs[s1].finished
    assert srv.run_until(watch, 25)
    assert ("running", "running") in seen and ("holding", "running") not in seen and ("running", "holding") not in seen
    assert srv.status(s0).state == srv.status(s1).state == "done"
    assert max(contacts) == 0 and min(gaps) > 0.30
    srv.run_for(0.3)
    w, man = srv.world(), srv.manifest()
    assert trials.judge(w, man, "block_1", "stage_1", A0) == (True, "block_1 resting in stage_1, gripper empty, arm at travel height")
    assert trials.judge(w, man, "block_2", "slot_3", A1)[0]
    assert sorted(e.kind for e in srv.events) == ["skill_done", "skill_done"] and {e.arm for e in srv.events} == {A0, A1}
    assert srv.zone_lock.holder is None and srv.zone_lock.queue == []       # neither needed the zone
    assert w.places["shared_zone"].name == "the shared zone (free)"


def test_claims_are_reported_and_a_second_arm_cannot_take_a_claimed_block():
    srv = pair_server()
    srv.sim_world.move("block_1", XC, 0.0)                                  # in the zone, in both arms' reach
    srv.run_for(0.1)
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "stage_0"})
    srv.run_for(0.1)
    assert srv.world().objects["block_1"].claimed_by == A0 and srv.world().objects["block_2"].claimed_by is None
    assert srv.precondition(A1, "pick_and_place", {"object": "block_1", "place": "slot_0"}) == "precondition: block_1 is claimed by arm_0: its pick_and_place is running"
    assert srv.precondition(A1, "push", {"object": "block_1", "direction": "left", "distance": 0.05}) == "precondition: block_1 is claimed by arm_0: its pick_and_place is running"
    assert srv.precondition(A0, "stack_on", {"object": "block_1", "onto": "block_2"}).startswith("unreachable")   # the claimer may re-plan it
    assert srv.run_until(in_phase(srv, s0, "carry"), 20)
    assert srv.precondition(A1, "pick_and_place", {"object": "block_1", "place": "slot_0"}) == "precondition: block_1 is held by arm_0"
    assert srv.run_until(srv.finished(s0), 20) and srv.status(s0).state == "done"
    assert srv.world().objects["block_1"].claimed_by is None


# ---------------------------------------------------------------- the shared-zone lock


def test_zone_lock_holds_the_second_arm_at_its_side_line_and_releases_it():
    srv = pair_server()
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "zone_1"})
    assert srv.run_until(lambda: srv.zone_lock.holder == A0, 20)           # arm_0 has the zone (arm_1 idle behind its line)
    assert srv.world().places["shared_zone"].name == "the shared zone (held by arm_0)"
    s1 = srv.start(A1, "pick_and_place", {"object": "block_2", "place": "zone_0"})   # block_2 is beyond arm_1's side line
    assert srv.run_until(waiting(srv, s1), 5)
    st = srv.status(s1)
    assert st.state == "holding" and not srv._runs[s1].tripped and st.phase == "approach"
    assert st.phase_text.startswith("waiting for the shared zone, held by arm_0: moving above block_2 at travel height")
    w = srv.world()
    assert w.arms[A1].mode == "holding" and w.arms[A0].mode == "running"
    srv.run_for(0.3)                                                        # the EE settles onto the frozen setpoint
    x_wait = own_x(srv, A1)
    assert x_wait <= SIDE + 0.004                                           # at its side line, not beyond
    srv.resume(A1)                                                          # no model can take it in: the lock outranks resume
    srv.run_for(0.3)
    assert srv.status(s1).state == "holding" and waiting(srv, s1)() and own_x(srv, A1) == pytest.approx(x_wait, abs=1e-3)
    srv.hold(A1)                                                            # the brain's hold on top of the wait
    assert srv.run_until(srv.finished(s0), 20) and srv.status(s0).state == "done"
    assert srv.run_until(lambda: srv.zone_lock.holder is None, 5)          # arm_0, idle inside, is being parked
    assert srv.run_until(lambda: own_x(srv, A0) < SIDE, 5) and srv.world().arms[A0].mode == "idle"
    srv.run_for(0.3)
    st = srv.status(s1)
    assert st.state == "holding" and st.phase_text.startswith("held: moving above block_2") and own_x(srv, A1) == pytest.approx(x_wait, abs=1e-3)
    srv.resume(A1)
    assert srv.status(s1).state == "running"
    assert srv.run_until(lambda: srv.zone_lock.holder == A1, 10)
    assert srv.world().places["shared_zone"].name == "the shared zone (held by arm_1)"
    assert srv.run_until(srv.finished(s1), 25) and srv.status(s1).state == "done"
    srv.run_for(0.3)
    w, man = srv.world(), srv.manifest()
    assert w.objects["block_1"].where == "zone_1" and w.objects["block_2"].where == "zone_0"
    assert trials.judge(w, man, "block_2", "zone_0", A1)[0] and trials._arm_checks(w, man, A0) == []
    assert not any(e.kind == "safety_trip" for e in srv.events) and arm_arm_contacts(srv) == 0
    assert srv.run_until(lambda: srv.zone_lock.holder is None and own_x(srv, A1) < SIDE, 5)   # and arm_1 is parked in turn


def test_two_arms_carrying_to_the_zone_at_once_the_second_backs_off():
    srv = pair_server()
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "zone_1"})
    srv.run_for(0.5)
    s1 = srv.start(A1, "pick_and_place", {"object": "block_2", "place": "zone_2"})
    texts, contacts = [], []

    def watch():
        texts.append((srv.status(s0).phase_text, srv.status(s1).phase_text))
        contacts.append(arm_arm_contacts(srv))
        return srv._runs[s0].finished and srv._runs[s1].finished
    assert srv.run_until(watch, 60)
    assert srv.status(s0).state == srv.status(s1).state == "done" and max(contacts) == 0
    t0, t1 = zip(*texts)
    assert any(t.startswith("waiting for the shared zone, arm_1 is still within reach of it: carrying block_1") for t in t0)
    assert any(t.startswith("waiting for the shared zone, arm_0 is still within reach of it, backing off to its side line: carrying block_2") for t in t1)
    assert any(t.startswith("waiting for the shared zone, held by arm_0") for t in t1)
    assert not any(e.kind == "safety_trip" for e in srv.events) and sorted(e.kind for e in srv.events) == ["skill_done", "skill_done"]
    srv.run_for(0.3)
    w = srv.world()
    assert w.objects["block_1"].where == "zone_1" and w.objects["block_2"].where == "zone_2"


def test_zone_wait_survives_a_pause_and_a_replacement_releases_the_lock():
    srv = pair_server()
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "zone_1"})
    assert srv.run_until(lambda: srv.zone_lock.holder == A0, 20)
    s1 = srv.start(A1, "pick_and_place", {"object": "block_2", "place": "zone_2"})
    assert srv.run_until(lambda: srv._runs[s1].waiting == "held by arm_0", 5)
    srv.pause(A1)
    assert srv.status(s1).state == "paused" and srv.status(s1).phase_text.startswith("paused: ")
    srv.run_for(0.2)
    srv.resume(A1)
    srv.run_for(0.1)
    assert srv._runs[s1].waiting == "held by arm_0" and srv.status(s1).state == "holding"   # back to waiting, still out
    s0b = srv.start(A0, "survey", {})                                       # arm_0's pick replaced mid-zone by a skill that stays put
    assert srv.status(s0).state == "cancelled"
    assert srv.run_until(srv.finished(s0b), 5)
    assert srv.run_until(lambda: srv.zone_lock.holder == A1, 10)            # arm_0 idle inside is parked, arm_1 goes in
    assert srv.run_until(srv.finished(s1), 25) and srv.status(s1).state == "done"
    assert not any(e.kind == "safety_trip" for e in srv.events)


def test_a_block_in_the_zone_off_a_spot_and_the_zone_as_a_place():
    srv = pair_server()
    srv.sim_world.move("block_1", XC - 0.03, 0.17)
    srv.run_for(0.1)
    assert srv.world().objects["block_1"].where == "shared_zone"
    assert srv.arms[A0].place("shared_zone") == pytest.approx((XC, -0.10))               # the first free spot
    srv.sim_world.move("block_2", XC, -0.10)
    srv.run_for(0.2)
    assert srv.world().objects["block_2"].where == "zone_0" and srv.arms[A0].place("shared_zone") == pytest.approx((XC, 0.0))
    assert srv.precondition(A0, "pick_and_place", {"object": "block_1", "place": "zone_0"}) == "blocked: shared-zone spot 0 is occupied by block_2"
    assert srv.precondition(A0, "pick_and_place", {"object": "block_1", "place": "shared_zone"}) is None


# ---------------------------------------------------------------- the hand-over


def test_handover_via_the_zone():
    srv = pair_server([OWN_0])
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "zone_2"})
    assert srv.run_until(srv.finished(s0), 25) and srv.status(s0).state == "done"
    w = srv.world()
    assert w.objects["block_1"].where == "zone_2" and w.arms[A0].holding is None
    assert srv.zone_lock.holder == A0                                       # still inside at travel height
    s1 = srv.start(A1, "pick_and_place", {"object": "block_1", "place": "slot_4"})
    waited, contacts = [], []

    def watch():
        st = srv.status(s1)
        if st.state == "holding":
            waited.append(st.phase_text)
        contacts.append(arm_arm_contacts(srv))
        return srv._runs[s1].finished
    assert srv.run_until(watch, 30) and srv.status(s1).state == "done"
    assert waited and waited[0].startswith("waiting for the shared zone, held by arm_0: moving above block_1 at travel height")
    assert max(contacts) == 0
    srv.run_for(0.3)
    w, man = srv.world(), srv.manifest()
    ok, why = trials.judge(w, man, "block_1", "slot_4", A1)
    assert ok, why
    assert trials._arm_checks(w, man, A0) == [] and (w.objects["block_1"].x, w.objects["block_1"].y) == pytest.approx((D - 0.11, 0.16), abs=0.02)
    assert [e.kind for e in srv.events] == ["skill_done", "skill_done"]


def test_pair_trials_from_the_conformance_script():
    v = trials.run_pair_trial(trials.sample_pair(random.Random(7), 0, "pick_and_place"))
    assert v.ok, v
    assert v.arms == {A0: True, A1: True} and v.state == "done" and "safety_trip" not in v.why
    v = trials.run_pair_trial(trials.sample_pair(random.Random(7), 0, "handover_via_zone"))
    assert v.ok, v
    assert v.arms == {A1: True} and "arm_1 waited for the shared zone" in v.why
    with pytest.raises(SystemExit):                                         # the CLI refuses the pair skill on one arm
        trials.main(["--arms", "1", "--skill", "handover_via_zone", "--n", "1"])


# ---------------------------------------------------------------- arm-arm proximity


def test_arm_arm_proximity_trip_holds_the_closing_arm_and_names_both():
    # No zone, no lock: a one-slot tray in the middle of the table, arm_1 parked idle beyond it.
    spec = scene.SceneSpec([scene.Block("block_1", (0.20, 0.0))], scene.Tray((0.33, 0.0), n=1), (A0, A1), zone_width=None)
    srv = SimRobotServer(spec)
    srv.events = []
    srv.subscribe(srv.events.append)
    srv.arms[A1].stage(srv.arms[A1].base.to_local((0.50, 0.0, TRAVEL_Z)))
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "slot_0"})
    contacts = []
    assert srv.run_until(lambda: contacts.append(arm_arm_contacts(srv)) or any(e.kind == "safety_trip" for e in srv.events), 15)
    ev = next(e for e in srv.events if e.kind == "safety_trip")
    assert ev.arm == A0 and ev.skill_id == s0 and ev.data["arms"] == [A0, A1]
    assert ev.text.startswith("the safety filter tripped on arm_0: arm_0 and arm_1 would come within 4 cm of each other")
    st = srv.status(s0)
    assert st.state == "holding" and st.phase == "carry" and st.phase_text.startswith("safety trip (arm_0 and arm_1 would come within 4 cm")
    w = srv.world()
    assert w.arms[A0].mode == "tripped" and w.arms[A1].mode == "idle" and (w.arms[A1].x, w.arms[A1].y) == pytest.approx((0.50, 0.0), abs=2e-3)
    assert 0.20 < w.arms[A0].x < 0.28                                       # stopped by the wrists, 22+ cm short of the other EE
    srv.run_for(0.5)
    assert max(contacts) == 0 and arm_arm_contacts(srv) == 0
    assert ARM_GAP - 0.003 <= srv.arm_gap(srv.arms[A0], srv.arms[A1]) < ARM_GAP + 0.01   # the wrist housings keep the gap
    srv.resume(A0)                                                          # arm_1 is still there: trips again at once
    srv.run_for(0.2)
    assert [e.kind for e in srv.events].count("safety_trip") == 2 and srv.status(s0).state == "holding"
    srv.arms[A1].drive((PAIR_HOME_XY[0], 0.0, TRAVEL_Z), 0.15)             # arm_1 leaves (as a brain's skill or STOP's park would)
    srv.run_for(2.5)
    srv.resume(A0)
    assert srv.run_until(srv.finished(s0), 20) and srv.status(s0).state == "done"
    srv.run_for(0.3)
    assert trials.judge(srv.world(), srv.manifest(), "block_1", "slot_0", A0)[0]
    assert [e.kind for e in srv.events].count("safety_trip") == 2


def test_two_closing_arms_both_hold_the_later_started_first():
    tray = scene.Tray((0.35, 0.0), (0.06, 0.0), 2)                          # slot_0 (0.35, 0), slot_1 (0.41, 0)
    spec = scene.SceneSpec([scene.Block("block_1", (0.21, 0.0)), scene.Block("block_2", (D - 0.21, 0.0), rgba=BLUE)],
                           tray, (A0, A1), zone_width=None)
    srv = SimRobotServer(spec)
    srv.events = []
    srv.subscribe(srv.events.append)
    s0 = srv.start(A0, "pick_and_place", {"object": "block_1", "place": "slot_1"})     # across each other's path
    srv.run_for(0.3)
    s1 = srv.start(A1, "pick_and_place", {"object": "block_2", "place": "slot_0"})
    contacts = []
    assert srv.run_until(lambda: contacts.append(arm_arm_contacts(srv)) or len([e for e in srv.events if e.kind == "safety_trip"]) >= 2, 20)
    trips = [e for e in srv.events if e.kind == "safety_trip"]
    assert [e.arm for e in trips[:2]] == [A1, A0] and trips[0].data["arms"] == [A1, A0] and trips[0].t == trips[1].t
    assert srv.status(s0).state == srv.status(s1).state == "holding" and srv._runs[s0].tripped and srv._runs[s1].tripped
    srv.run_for(0.5)
    assert max(contacts) == 0 and arm_arm_contacts(srv) == 0 and ee_gap(srv) > 0.20


# ---------------------------------------------------------------- STOP


def test_stop_parks_both_arms():
    srv = pair_server()
    s0 = srv.start(A0, "pick_and_place", TO_STAGE)
    s1 = srv.start(A1, "pick_and_place", TO_TRAY)
    assert srv.run_until(in_phase(srv, s0, "carry"), 20)
    srv.stop()
    assert sorted(e.kind for e in srv.events) == ["skill_failed", "skill_failed"]
    assert srv.status(s0).reason.startswith("stopped: STOP") and srv.status(s1).reason.startswith("stopped: STOP")
    assert srv.world().arms[A0].mode == srv.world().arms[A1].mode == "stopped"
    with pytest.raises(ValueError, match="STOP"):
        srv.start(A1, "survey", {})
    srv.run_for(8.0)
    w = srv.world()
    assert (w.arms[A0].x, w.arms[A0].y) == pytest.approx((PAIR_HOME_XY[0], 0.0), abs=0.01)
    assert (w.arms[A1].x, w.arms[A1].y) == pytest.approx((D - PAIR_HOME_XY[0], 0.0), abs=0.01)
    assert w.arms[A0].holding == "block_1"                                  # parked keeping its grip
    assert len(srv.events) == 2

