"""The surgical-tool scene and the two orientation skills (catalog 0.2): rods grasped by their handle,
`insert` into a rack's tight-fitting holders with the regrasp when the wrist's roll range runs out,
`reorient` of cubes and rods to the four directions, their failure codes, hold/pause through the
press, and the conformance judges. No rendering, no clock: every test steps the server itself."""
from __future__ import annotations

import math
import random

import pytest

from robojev import catalog
from robojev import conformance as trials
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.server import ARM_ID, TRAVEL_Z, SimRobotServer

pytestmark = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(),
                                reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")

CUBE = scene.Block("block_1", (0.30, -0.08), yaw=0.5)
DEG = math.degrees


def server(spec: scene.SceneSpec | None = None, **kw) -> SimRobotServer:
    srv = SimRobotServer(spec or scene.tools_scene(), realtime=False, **kw)
    srv.events = []
    srv.subscribe(srv.events.append)
    return srv


def rod_scene(xy=(0.24, 0.05), yaw=0.4, rack: scene.Rack | None = None, blocks=()) -> scene.SceneSpec:
    return scene.SceneSpec(blocks=list(blocks), tools=[scene.Tool("scalpel", xy, yaw)], rack=rack)


def phase(srv, sid) -> str:
    return srv._runs[sid].phase


def in_phase(srv, sid, name):
    return lambda: phase(srv, sid) == name or srv.status(sid).state in ("done", "failed")


def run(srv, skill, args, timeout=40) -> str:
    sid = srv.start(ARM_ID, skill, args)
    assert srv.run_until(srv.finished(sid), timeout), srv.status(sid)
    srv.run_for(0.3)
    return sid


def failed(srv, sid, code: str):
    st = srv.status(sid)
    assert st.state == "failed" and st.reason.startswith(code + ": "), st
    ev = [e for e in srv.events if e.kind == "skill_failed"]
    assert len(ev) == 1 and ev[0].skill_id == sid and ev[0].text == st.reason
    return st


def front_off(o, yaw: float) -> float:
    return abs(DEG(skills.wrap(o.yaw - yaw)))


# ---------------------------------------------------------------- scene, world, manifest


def test_tools_scene_builds_and_world_reports_yaw_shape_and_holder_yaw():
    spec = scene.tools_scene(rack=scene.Rack(yaw=0.3), tray=scene.Tray())
    spec.blocks.append(CUBE)
    srv = server(spec)
    assert srv.model.nbody > 10
    w = srv.world()
    assert {o.id for o in w.objects.values()} == {"scalpel", "forceps", "scissors", "block_1"}
    for t in spec.tools:
        o = w.objects[t.name]
        assert o.shape == "rod" and o.label == t.label and o.name == f"the {t.label}" and o.size == pytest.approx(0.12)
        assert o.yaw == pytest.approx(t.yaw, abs=1e-3) and o.where == "table" and o.z == pytest.approx(scene.TABLE_Z + 0.006, abs=1e-3)
    c = w.objects["block_1"]
    assert c.shape == "cube" and c.yaw == pytest.approx(0.5, abs=1e-3) and c.size == 0.04
    holders = [p for p in w.places.values() if p.kind == "holder"]
    assert [h.id for h in holders] == ["holder_0", "holder_1", "holder_2"]
    assert all(h.yaw == pytest.approx(0.3) and h.capacity == 1 for h in holders)
    assert holders[0].name == "rack slot 0 (the leftmost)" and holders[0].y > holders[2].y
    assert holders[2].name == "rack slot 2 (the rightmost)" and holders[1].name == "rack slot 1"
    assert w.places["rack"].kind == "group" and w.places["rack"].members == ("holder_0", "holder_1", "holder_2")
    assert w.places["slot_0"].yaw is None
    # the ArmIO's own reads: a rod's pose carries the grasp strategy, a holder its yaw
    a = srv.arms[ARM_ID]
    o = a.obj("forceps")
    assert o.shape == "rod" and o.size == pytest.approx(0.014) and o.length == 0.12 and o.handle == pytest.approx(0.045)
    assert a.place("holder_1")[2] == pytest.approx(0.3) and len(a.place("slot_1")) == 2
    assert scene.Rack(yaw=math.pi / 2).places()["holder_0"].name == "rack slot 0 (the farthest)"


def test_manifest_advertises_insert_and_reorient_from_the_catalog():
    srv = server()
    man = srv.manifest()
    assert catalog.check_manifest_skills(man.skills) == []
    assert man.catalog == catalog.CATALOG_VERSION
    assert man.skill("insert") is catalog.INSERT and man.skill("reorient") is catalog.REORIENT
    assert {s.name for s in skills.SPECS} == set(skills.SKILLS)


def test_grasp_by_shape_helpers():
    cube = skills.ObjPose(0.30, 0.0, 0.0, 0.5, 0.04)
    rod = skills.ObjPose(0.30, 0.0, 0.0, 0.5, 0.014, "rod", 0.12, 0.045)
    assert skills.grasp_point(cube) == (0.30, 0.0)
    gx, gy = skills.grasp_point(rod)
    assert (gx, gy) == pytest.approx((0.30 + 0.045 * math.cos(0.5), 0.045 * math.sin(0.5)))
    assert len(skills.grasp_candidates(cube)) == 4 and len(skills.grasp_candidates(rod)) == 2
    for c in skills.grasp_candidates(rod):
        assert abs(math.cos(c - rod.yaw)) < 1e-9                   # across the rod's axis
    assert skills.grasp_candidates(cube)[0] == pytest.approx(skills.grasp_yaw(0.30, 0.0, 0.5) % (math.pi / 2) + 0.5 - 0.5 % (math.pi / 2))
    p = skills.Params()
    near = skills.ObjPose(0.30, 0.11, 0, 0, 0.014, "rod", 0.12, 0.045)  # a rod 11 cm away reaches 6 cm toward us: shrink
    assert skills.open_width(0.04, [near], (0.30, 0.0), 0.088, p) == pytest.approx(2 * (0.11 - 0.06 - 0.014 - 0.005))
    cube_near = skills.ObjPose(0.30, 0.11, 0, 0, 0.04)                   # a cube that far reaches 2 cm: no shrink
    assert skills.open_width(0.04, [cube_near], (0.30, 0.0), 0.088, p) == pytest.approx(0.07)
    assert skills.front_of("left") == pytest.approx(math.pi / 2) and skills.front_of("toward_robot") == pytest.approx(math.pi)


def test_turn_plan_is_direct_within_the_range_and_splits_beyond_it():
    R = math.radians(120)
    at = (0.30, 0.0)
    n = skills.neutral_yaw(*at)
    # a rod held with its front 90 deg from the closing axis, wanted pointing the way that needs 60 deg of roll: direct
    plan = skills.turn_plan("rod", math.pi / 2, n + math.radians(60) + math.pi / 2, at, at, 0.0, R)
    assert plan.direct and DEG(plan.roll) == pytest.approx(60)
    # wanted 170 deg of roll: beyond the range, so turn now to +120 (regrasping the other way round then needs -60 + ...)
    plan = skills.turn_plan("rod", math.pi / 2, n + math.radians(170) + math.pi / 2, at, at, math.radians(10), R)
    assert not plan.direct and DEG(plan.roll) == pytest.approx(120)
    # after that turn the regrasp's own roll (the other candidate) is within the range, and its destination roll too
    r_next = skills.wrap(plan.roll + math.pi)
    assert abs(DEG(r_next)) <= 120 and abs(DEG(skills.roll_of(n + math.radians(170) + math.pi / 2 - (-math.pi / 2), *at))) <= 120
    # a cube's front on the closing axis, wanted 150 deg away: regrasp half a turn on (its destination roll is then
    # 30 deg, the least of the three relations), so turn as far as the range goes now
    plan = skills.turn_plan("cube", 0.0, n + math.radians(150), at, at, 0.0, R)
    assert not plan.direct and abs(DEG(plan.roll)) == pytest.approx(120)
    assert skills.turn_plan("cube", 0.0, n + math.radians(100), at, at, 0.0, R).direct


# ---------------------------------------------------------------- a rod picked and placed


def test_rod_pick_and_place_lands_its_centre_in_the_slot():
    srv = server(scene.tools_scene(tray=scene.Tray()))
    sid = run(srv, "pick_and_place", {"object": "forceps", "place": "slot_3"})
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == "forceps placed at slot_3"
    sk = srv._runs[sid]
    assert abs(DEG(sk.delta)) == pytest.approx(90, abs=3)         # pinched across the handle
    assert sk.size == pytest.approx(0.014) and sk.open_w == pytest.approx(0.044)
    w = srv.world()
    o = w.objects["forceps"]
    slot = w.places["slot_3"]
    assert o.where == "slot_3" and math.hypot(o.x - slot.x, o.y - slot.y) < 0.01
    ok, why = trials.judge(w, srv.manifest(), "forceps", "slot_3")
    assert ok, why
    assert [e.kind for e in srv.events] == ["skill_done"]


# ---------------------------------------------------------------- insert


INSERT_CASES = [  # (rod xy, rod yaw, rack yaw, holder, regrasp expected)
    ((0.23, 0.0), 0.4, math.pi, "holder_0", True),    # the front nearly opposite the opening: the other way round is out of the roll range
    ((0.23, 0.0), -1.3, math.pi, "holder_1", False),
    ((0.23, 0.0), 2.6, math.pi, "holder_2", False),
    ((0.23, 0.0), 1.0, 0.3, "holder_1", False),
    ((0.21, 0.08), -2.5, 2.0, "holder_0", False),
]


@pytest.mark.parametrize("rod_xy,rod_yaw,rack_yaw,holder,regrasp", INSERT_CASES)
def test_insert_seats_the_rod_aligned_with_the_holder(rod_xy, rod_yaw, rack_yaw, holder, regrasp):
    srv = server(rod_scene(rod_xy, rod_yaw, scene.Rack(yaw=rack_yaw)))
    assert srv.precondition(ARM_ID, "insert", {"object": "scalpel", "place": holder}) is None
    sid = run(srv, "insert", {"object": "scalpel", "place": holder})
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == f"scalpel seated in {holder}", st
    assert (srv._runs[sid].rounds == 1) == regrasp
    w = srv.world()
    o, h = w.objects["scalpel"], w.places[holder]
    assert o.where == holder
    assert front_off(o, h.yaw) < 3 and math.hypot(o.x - h.x, o.y - h.y) < 0.004
    ok, why = trials.judge_insert(w, srv.manifest(), "scalpel", holder)
    assert ok, why
    assert w.arms[ARM_ID].holding is None and w.arms[ARM_ID].z == pytest.approx(TRAVEL_Z, abs=0.01)
    assert not any(e.kind == "safety_trip" for e in srv.events) and [e.kind for e in srv.events] == ["skill_done"]


def test_insert_phases_and_the_regrasp_round():
    srv = server(rod_scene((0.23, 0.0), 0.4, scene.Rack(yaw=math.pi)))
    sid = srv.start(ARM_ID, "insert", {"object": "scalpel", "place": "holder_0"})
    seen = []
    def watch():
        p = phase(srv, sid)
        if not seen or seen[-1] != p:
            seen.append(p)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(watch, 40)
    one = list(skills.Grasping.PHASES)
    assert seen == one + list(skills.Insert.PHASES) + ["done"]      # a put-down turned as far as the wrist goes, then the seating
    sk = srv._runs[sid]
    assert sk.rounds == 1 and abs(DEG(sk.dest_roll)) <= 120 + 1e-6
    assert srv.status(sid).state == "done"


def test_insert_into_the_rack_group_takes_a_free_holder():
    srv = server()
    run(srv, "insert", {"object": "forceps", "place": "holder_0"})
    assert srv.world().objects["forceps"].where == "holder_0"
    assert srv.precondition(ARM_ID, "insert", {"object": "scissors", "place": "holder_0"}) == "precondition: rack slot 0 (the leftmost) is occupied by forceps"
    sid = run(srv, "insert", {"object": "scissors", "place": "rack"})
    assert srv.status(sid).state == "done" and srv.world().objects["scissors"].where == "holder_1"


def test_insert_refusals_are_preconditions():
    spec = scene.tools_scene(tray=scene.Tray())
    spec.blocks.append(scene.Block("block_1", (0.25, -0.12)))
    srv = server(spec)
    pre = srv.precondition
    assert pre(ARM_ID, "insert", {"object": "block_1", "place": "holder_1"}) == "precondition: block_1 is a cube; only a rod seats in a holder"
    assert pre(ARM_ID, "insert", {"object": "forceps", "place": "slot_1"}).startswith("precondition: tray slot 1 is not a holder")
    assert pre(ARM_ID, "insert", {"object": "forceps", "place": "bin_7"}) == "precondition: there is no place bin_7 in the world"
    assert pre(ARM_ID, "insert", {"object": "forceps"}) == "precondition: insert needs place"
    assert pre(ARM_ID, "pick_and_place", {"object": "forceps", "place": "holder_1"}) == "precondition: rack slot 1 is a holder: insert seats things there"
    assert pre(ARM_ID, "pick_and_place", {"object": "block_1", "place": "rack"}) == "precondition: the rack (any free slot) is a holder: insert seats things there"
    assert pre(ARM_ID, "reorient", {"object": "forceps", "direction": "up"}).startswith("precondition: direction must be one of")
    assert pre(ARM_ID, "insert", {"object": "forceps", "place": "holder_1"}) is None
    far = server(scene.SceneSpec(tools=[scene.Tool("scalpel", (0.30, 0.0), 0.0)], rack=scene.Rack((0.40, 0.0), yaw=0.0)))
    assert far.precondition(ARM_ID, "insert", {"object": "scalpel", "place": "holder_1"}).startswith("unreachable: the handle at holder_1 would be at (0.45, 0.00)")
    with pytest.raises(ValueError, match="only a rod"):
        srv.start(ARM_ID, "insert", {"object": "block_1", "place": "holder_1"})
    assert srv.events == []


def test_insert_into_a_holder_too_narrow_is_blocked_and_keeps_the_rod():
    srv = server(scene.tools_scene(rack=scene.Rack(clearance=-0.002)))      # the rod lies across the walls
    sid = run(srv, "insert", {"object": "forceps", "place": "holder_1"}, timeout=30)
    st = failed(srv, sid, "blocked")
    assert st.phase == "insert" and "resting on a wall" in st.reason
    w = srv.world()
    assert w.arms[ARM_ID].holding == "forceps" and w.objects["forceps"].where == f"gripper:{ARM_ID}"
    assert w.objects["forceps"].z > scene.TABLE_Z + 0.015                  # on top of the walls, not in the opening
    assert not any(e.kind == "safety_trip" for e in srv.events)


def test_insert_is_blocked_when_the_holder_is_taken_meanwhile():
    srv = server()
    sid = srv.start(ARM_ID, "insert", {"object": "forceps", "place": "holder_1"})
    assert srv.run_until(in_phase(srv, sid, "carry"), 20)
    hx, hy = srv.world().places["holder_1"].x, srv.world().places["holder_1"].y
    srv.sim_world.move("scissors", hx, hy, yaw=math.pi)                   # someone seats another tool there
    srv.run_for(0.3)
    assert srv.world().objects["scissors"].where == "holder_1"
    srv.retarget(sid)
    srv.run_for(0.1)
    st = failed(srv, sid, "blocked")
    assert st.reason == "blocked: holder_1 is occupied by scissors" and srv.world().arms[ARM_ID].holding == "forceps"


def test_insert_begins_from_a_held_rod():
    srv = server(scene.tools_scene(tray=scene.Tray()))
    sid = srv.start(ARM_ID, "pick_and_place", {"object": "forceps", "place": "slot_1"})
    assert srv.run_until(in_phase(srv, sid, "carry"), 20)
    srv.run_for(0.3)
    sid2 = srv.start(ARM_ID, "insert", {"object": "forceps", "place": "holder_1"})    # a correction mid-carry
    assert srv.status(sid).state == "cancelled"
    seen = []
    def watch():
        p = phase(srv, sid2)
        if not seen or seen[-1] != p:
            seen.append(p)
        return srv.status(sid2).state in ("done", "failed")
    assert srv.run_until(watch, 40)
    # straight from the fingers into the lift; the roll range may still ask for a regrasp round before the seating
    assert seen[:3] == ["approach", "lift", "carry"] and "insert" in seen and seen[-1] == "done", seen
    srv.run_for(0.3)
    ok, why = trials.judge_insert(srv.world(), srv.manifest(), "forceps", "holder_1")
    assert ok, why


@pytest.mark.parametrize("how", ["hold", "pause"])
def test_hold_and_pause_mid_insert_keep_the_grip_and_resume_seats(how):
    srv = server()
    sid = srv.start(ARM_ID, "insert", {"object": "forceps", "place": "holder_1"})
    assert srv.run_until(in_phase(srv, sid, "insert"), 20) and phase(srv, sid) == "insert"
    srv.run_for(0.3)
    getattr(srv, how)(ARM_ID)
    st = srv.status(sid)
    assert st.phase == "insert" and st.state == ("holding" if how == "hold" else "paused")
    assert st.phase_text.startswith(("held: pressing", "paused: pressing"))
    sp0 = srv.arms[ARM_ID].mover.setpoint
    srv.run_for(0.2)
    a0, o0 = srv.world().arms[ARM_ID], srv.world().objects["forceps"]
    srv.heartbeat()
    srv.run_for(1.5)
    w = srv.world()
    a1, o1 = w.arms[ARM_ID], w.objects["forceps"]
    assert srv.arms[ARM_ID].mover.setpoint == sp0 and srv.status(sid).state == st.state
    assert a1.holding == "forceps" and o1.where == f"gripper:{ARM_ID}"
    assert math.dist((a0.x, a0.y, a0.z), (a1.x, a1.y, a1.z)) < 0.003
    assert math.dist((o0.x, o0.y, o0.z), (o1.x, o1.y, o1.z)) < 0.003
    srv.resume(ARM_ID)
    assert srv.status(sid).state == "running" and phase(srv, sid) == "insert"
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"
    srv.run_for(0.3)
    ok, why = trials.judge_insert(srv.world(), srv.manifest(), "forceps", "holder_1")
    assert ok, why


# ---------------------------------------------------------------- reorient


@pytest.mark.parametrize("direction", skills.DIRECTIONS)
@pytest.mark.parametrize("shape", ["cube", "rod"])
def test_reorient_turns_the_object_in_place(shape, direction):
    if shape == "cube":
        srv, oid, start = server(scene.SceneSpec(blocks=[CUBE])), "block_1", CUBE.xy
    else:
        srv, oid, start = server(rod_scene((0.24, 0.05), 0.4)), "scalpel", (0.24, 0.05)
    assert srv.precondition(ARM_ID, "reorient", {"object": oid, "direction": direction}) is None
    sid = run(srv, "reorient", {"object": oid, "direction": direction})
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == f"{oid} turned to point {direction}", st
    w = srv.world()
    o = w.objects[oid]
    assert front_off(o, skills.front_of(direction)) < 5
    assert math.dist((o.x, o.y), start) < 0.01 and o.where == "table"
    ok, why = trials.judge_reorient(w, srv.manifest(), oid, start, direction)
    assert ok, why
    assert w.arms[ARM_ID].holding is None and w.arms[ARM_ID].z == pytest.approx(TRAVEL_Z, abs=0.01)
    # the rod from 23 deg to toward_robot needs more than the roll range: two grasps; everything else one
    assert srv._runs[sid].rounds == (1 if (shape, direction) == ("rod", "toward_robot") else 0)
    assert [e.kind for e in srv.events] == ["skill_done"]


def test_reorient_cube_phases_and_hold_in_carry():
    srv = server(scene.SceneSpec(blocks=[CUBE]))
    sid = srv.start(ARM_ID, "reorient", {"object": "block_1", "direction": "toward_robot"})
    assert srv.run_until(in_phase(srv, sid, "carry"), 20) and phase(srv, sid) == "carry"
    srv.run_for(0.2)
    srv.hold(ARM_ID)
    assert srv.status(sid).state == "holding" and srv.world().arms[ARM_ID].holding == "block_1"
    srv.run_for(0.5)
    srv.resume(ARM_ID)
    seen = []
    def watch():
        p = phase(srv, sid)
        if not seen or seen[-1] != p:
            seen.append(p)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(watch, 30)
    assert seen == ["carry", "lower", "release", "retreat", "done"]
    srv.run_for(0.3)
    assert trials.judge_reorient(srv.world(), srv.manifest(), "block_1", CUBE.xy, "toward_robot")[0]


def test_reorient_refuses_an_unreachable_turn():
    srv = server(scene.SceneSpec(tools=[scene.Tool("scalpel", (0.40, 0.0), math.pi)]))
    why = srv.precondition(ARM_ID, "reorient", {"object": "scalpel", "direction": "away_from_robot"})
    assert why.startswith("unreachable: turned away_from_robot, scalpel would be held at (0.4") and why.endswith("outside the arm's workspace")
    assert srv.precondition(ARM_ID, "reorient", {"object": "scalpel", "direction": "toward_robot"}) is None


# ---------------------------------------------------------------- conformance


def test_conformance_samples_tools_trials_and_judges_them():
    rng = random.Random(7)
    ins = trials.sample(rng, 0, 5, "insert", "tools")
    assert ins.scene == "tools" and ins.tray is None and ins.rack is not None and ins.args["object"] == "tool_1"
    assert ins.args["place"].startswith("holder_") and trials._clear_of(ins.tools[0], ins.rack)
    v = trials.run_trial(ins, timeout_s=trials.TURN_TIMEOUT_S)
    assert v.ok and v.state == "done", v
    ro = trials.sample(random.Random(3), 1, 5, "reorient", "tools")
    assert ro.rack is None and ro.args["direction"] in skills.DIRECTIONS
    v = trials.run_trial(ro, timeout_s=trials.TURN_TIMEOUT_S)
    assert v.ok and v.state == "done", v
    pp = trials.sample(random.Random(5), 2, 5, "pick_and_place", "tools")
    assert pp.rack is None and pp.tray is not None and pp.args["place"] in ("slot_1", "slot_2", "slot_3")
    with pytest.raises(ValueError, match="tools scene"):
        trials.sample(rng, 3, 5, "insert", "letters")
    with pytest.raises(ValueError, match="not push"):
        trials.sample(rng, 4, 5, "push", "tools")
    letters = trials.sample(random.Random(0), 0, 5, "reorient")
    assert letters.scene == "letters" and letters.blocks and not letters.tools
    assert "turn" in trials.describe(letters) and "rack yaw" in trials.describe(ins)


def test_conformance_judges_read_the_world_not_the_skill():
    srv = server()
    w, man = srv.world(), srv.manifest()
    ok, why = trials.judge_insert(w, man, "forceps", "holder_1")
    assert not ok and "not inside its walls" in why and "deg off" in why
    ok, why = trials.judge_reorient(w, man, "forceps", (0.23, 0.0), "left")
    assert not ok and "deg off left" in why
    ok, why = trials.judge_reorient(w, man, "forceps", (0.20, 0.0), "right")           # yaw -1.3 rad: 15.5 deg off right
    assert not ok and "ended 3.0 cm from where it started" in why and "deg off" in why
    ok, why = trials.judge_reorient(w, man, "scissors", (0.23, -0.15), "toward_robot")  # yaw 2.6 rad: 31 deg off
    assert not ok
    ok, why = trials.judge_reorient(w, man, "scalpel", (0.23, 0.15), "away_from_robot")  # 0.4 rad = 23 deg off: too far
    assert not ok
    w.objects["scalpel"].yaw = 0.2                                                      # 11 deg: within 15
    ok, why = trials.judge_reorient(w, man, "scalpel", (0.23, 0.15), "away_from_robot")
    assert ok, why
