"""K2-K4: the MuJoCo WidowX behind the robot protocol, with every standard skill as physics. No
rendering, no clock: every test steps the server itself (realtime=False), so the run is deterministic
and takes seconds."""
from __future__ import annotations

import json
import math

import pytest

from robojev import catalog
from robojev import conformance as trials
from robojev.protocol import HandObs, RobotServer
from robojev.recording.recorder import JsonlRecorder
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.server import ARM_ID, TRAVEL_Z, SimRobotServer

pytestmark = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(),
                                reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")

ARGS = {"object": "block_1", "place": "slot_2"}
STACK = {"object": "block_1", "onto": "block_2"}
PUSH = {"object": "block_1", "direction": "left", "distance": 0.08}
HAND = {"object": "block_1"}
BLOCK_2 = scene.Block("block_2", (0.32, 0.10), yaw=-0.2, rgba=(0.2, 0.4, 0.8, 1))


def server(blocks=None, tray=scene.Tray(), **kw) -> SimRobotServer:
    blocks = blocks if blocks is not None else [scene.Block("block_1", (0.30, -0.08), yaw=0.5)]
    srv = SimRobotServer(scene.SceneSpec(blocks=blocks, tray=tray), realtime=False, **kw)
    srv.events = []
    srv.subscribe(srv.events.append)
    return srv


def two_blocks(**kw) -> SimRobotServer:
    return server([scene.Block("block_1", (0.30, -0.08), yaw=0.5), BLOCK_2], **kw)


def hand_under(srv, dz=0.05, held_out=True) -> HandObs:
    """A hand `dz` below the fingertips, where they are now."""
    a = srv.world().arms[ARM_ID]
    h = HandObs(a.x, a.y, a.z - dz, held_out=held_out)
    srv.set_hands([h])
    return h


def phases_seen(srv, sid, timeout=25) -> list[str]:
    """Run the skill to its end, returning the distinct phases in order (as a recorder would see them)."""
    seen = []
    def watch():
        p = phase(srv, sid)
        if not seen or seen[-1] != p:
            seen.append(p)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(watch, timeout)
    return seen


def phase(srv, sid) -> str:
    return srv._runs[sid].phase


def in_phase(srv, sid, name):
    return lambda: phase(srv, sid) == name or srv.status(sid).state in ("done", "failed")


# ---------------------------------------------------------------- scene and manifest


def test_scene_builds_with_blocks_and_tray():
    m = scene.build(scene.SceneSpec(blocks=[scene.Block("block_1", (0.3, 0.0)), scene.Block("block_2", (0.3, 0.1))]))
    assert m.nbody > 10 and m.opt.timestep == 0.002


def test_manifest_conforms_to_the_catalog():
    srv = server()
    assert isinstance(srv, RobotServer)
    man = srv.manifest()
    assert man.catalog == catalog.CATALOG_VERSION
    assert catalog.check_manifest_skills(man.skills) == []
    assert {s.name for s in man.skills} == {"pick_and_place", "stack_on", "push", "hand_over", "hold", "survey"}
    arm = man.arms[0]
    assert arm.id == ARM_ID and arm.gripper.max_width == pytest.approx(0.088) and arm.travel_z == TRAVEL_Z
    w = srv.world()
    assert w.objects["block_1"].where == "table" and w.arms[ARM_ID].mode == "idle"
    assert w.places["tray"].members == tuple(f"slot_{i}" for i in range(5))
    assert w.arms[ARM_ID].gripper_width == pytest.approx(0.088, abs=1e-3)


# ---------------------------------------------------------------- pick_and_place


@pytest.mark.parametrize("index", [0, 1, 2])
def test_three_trials_succeed(index):
    import random
    trial = trials.sample(random.Random(100 + index), index, 5)
    v = trials.run_trial(trial)
    assert v.ok, v
    assert v.state == "done" and v.sim_s < 15


def test_phases_and_status_text_are_literal():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    seen = []
    def watch():
        p = phase(srv, sid)
        if not seen or seen[-1] != p:
            seen.append(p)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(watch, 20)
    assert seen == list(skills.PickAndPlace.PHASES) + ["done"]
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == "block_1 placed at slot_2" and st.reason is None
    done = [e for e in srv.events if e.kind == "skill_done"]
    assert len(done) == 1 and done[0].skill_id == sid and done[0].arm == ARM_ID
    assert srv.world().objects["block_1"].where == "slot_2"


def test_hold_from_carry_keeps_the_block_and_resume_finishes():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "carry"), 20) and phase(srv, sid) == "carry"
    srv.run_for(0.2)
    srv.hold(ARM_ID)
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("held: carrying block_1")
    sp0 = srv.arms[ARM_ID].mover.setpoint      # the setpoint freezes at once; the EE settles onto it
    srv.run_for(0.2)                           # from ~1.3 cm behind (tracking lag at 0.15 m/s)
    ee0 = srv.world().arms[ARM_ID]
    srv.run_for(1.0)
    w = srv.world()
    assert w.arms[ARM_ID].mode == "holding" and w.arms[ARM_ID].holding == "block_1"
    assert w.objects["block_1"].where == f"gripper:{ARM_ID}"
    assert srv.arms[ARM_ID].mover.setpoint == sp0
    assert math.dist((ee0.x, ee0.y, ee0.z), (w.arms[ARM_ID].x, w.arms[ARM_ID].y, w.arms[ARM_ID].z)) < 0.003
    srv.resume(ARM_ID)
    assert srv.status(sid).state == "running"
    assert srv.run_until(srv.finished(sid), 20)
    srv.run_for(0.3)
    ok, why = trials.judge(srv.world(), srv.manifest(), "block_1", "slot_2")
    assert ok, why


def test_pause_does_not_resume_by_itself():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.run_for(0.3)
    srv.pause(ARM_ID)
    assert srv.status(sid).state == "paused" and srv.world().arms[ARM_ID].mode == "paused"
    sp0 = srv.arms[ARM_ID].mover.setpoint
    srv.run_for(0.2)                           # the EE settles onto the frozen setpoint
    a0 = srv.world().arms[ARM_ID]
    srv.heartbeat()
    srv.run_for(2.0)
    a1 = srv.world().arms[ARM_ID]
    assert srv.status(sid).state == "paused"
    assert srv.arms[ARM_ID].mover.setpoint == sp0
    assert math.dist((a0.x, a0.y, a0.z), (a1.x, a1.y, a1.z)) < 0.003
    assert srv.status(sid).phase_text.startswith("paused: ")
    srv.resume(ARM_ID)
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"


def test_control_calls_are_idempotent():
    srv = server()
    srv.hold(ARM_ID)                         # idle arm: nothing to hold, no error
    srv.resume(ARM_ID)
    srv.resume(ARM_ID)
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.run_for(0.3)
    srv.hold(ARM_ID)
    srv.hold(ARM_ID)
    assert srv.status(sid).state == "holding"
    srv.resume(ARM_ID)
    srv.resume(ARM_ID)
    assert srv.status(sid).state == "running"
    srv.pause(ARM_ID)
    srv.pause(ARM_ID)
    srv.hold(ARM_ID)                         # hold on a paused arm leaves it paused
    assert srv.status(sid).state == "paused"
    srv.resume(ARM_ID)
    srv.retarget(sid)
    srv.retarget(sid)
    assert srv.status(sid).state == "running"
    srv.stop()
    srv.stop()
    st = srv.status(sid)
    assert st.state == "failed" and st.reason.startswith("stopped: STOP")
    assert [e.kind for e in srv.events].count("skill_failed") == 1
    assert srv.world().arms[ARM_ID].mode == "stopped"
    with pytest.raises(ValueError, match="STOP"):
        srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.run_for(0.5)                          # parking moves nothing fast and raises nothing
    assert len(srv.events) == 1


def test_retarget_follows_a_moved_block():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "descend"), 10)
    srv.sim_world.move("block_1", 0.34, 0.02, yaw=-0.3)
    srv.retarget(sid)
    assert phase(srv, sid) == "approach" and srv.status(sid).heading == pytest.approx((0.34, 0.02), abs=1e-6)
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"
    srv.run_for(0.3)
    assert trials.judge(srv.world(), srv.manifest(), "block_1", "slot_2")[0]


# ---------------------------------------------------------------- safety


def test_lag_trip_on_an_immovable_obstacle():
    wall = scene.Block("wall", (0.32, -0.06), size=0.16, mass=20.0, rgba=(0.3, 0.3, 0.3, 1))
    srv = server([scene.Block("block_1", (0.38, 0.05)), wall])
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(lambda: any(e.kind == "safety_trip" for e in srv.events), 5)
    trip = next(e for e in srv.events if e.kind == "safety_trip")
    assert trip.skill_id == sid and trip.arm == ARM_ID and "in the way" in trip.text
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("safety trip (")
    assert srv.world().arms[ARM_ID].mode == "tripped"
    sp0 = srv.arms[ARM_ID].mover.setpoint
    srv.run_for(1.0)
    assert srv.arms[ARM_ID].mover.setpoint == sp0                    # held, not pushing on
    assert [e.kind for e in srv.events].count("safety_trip") == 1   # one trip per obstacle
    assert not any(e.kind == "skill_failed" for e in srv.events)


def test_missed_heartbeat_holds_the_arm():
    srv = server(heartbeat_timeout_s=0.5)
    srv.heartbeat()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.run_for(0.3)
    srv.heartbeat()
    srv.run_for(0.4)
    assert srv.status(sid).state == "running" and not srv.events
    srv.run_for(0.2)
    lost = [e for e in srv.events if e.kind == "heartbeat_lost"]
    assert len(lost) == 1 and lost[0].skill_id is None and srv.status(sid).state == "holding"
    assert srv.world().arms[ARM_ID].mode == "holding"
    sp0 = srv.arms[ARM_ID].mover.setpoint
    srv.run_for(0.2)                          # the EE settles onto the frozen setpoint
    a0 = srv.world().arms[ARM_ID]
    srv.heartbeat()                           # a late heartbeat re-arms the watchdog, resumes nothing
    srv.run_for(0.4)
    a1 = srv.world().arms[ARM_ID]
    assert srv.status(sid).state == "holding" and srv.arms[ARM_ID].mover.setpoint == sp0
    assert math.dist((a0.x, a0.y, a0.z), (a1.x, a1.y, a1.z)) < 0.003
    assert [e.kind for e in srv.events].count("heartbeat_lost") == 1
    srv.run_for(0.4)                          # ... and silence after it is a second loss
    assert [e.kind for e in srv.events].count("heartbeat_lost") == 2
    srv.resume(ARM_ID)
    for _ in range(40):                       # a brain that keeps beating keeps the arm moving
        srv.heartbeat()
        srv.run_for(0.4)
        if srv.status(sid).state in ("done", "failed"):
            break
    assert srv.status(sid).state == "done"


# ---------------------------------------------------------------- recorder and refusals


def test_recorder_writes_frames_with_the_skill_id(tmp_path):
    rec = JsonlRecorder(tmp_path / "run.jsonl")
    srv = server(recorder=rec)
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(srv.finished(sid), 20)
    srv.run_for(0.2)                          # idle ticks are not frames
    summary = rec.close()
    lines = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]
    frames = [line for line in lines if line["type"] == "frame"]
    events = [line for line in lines if line["type"] == "event"]
    assert summary["frames"] == len(frames) and 200 < len(frames) < 800
    assert all(f["skill_id"] == sid and f["arm"] == ARM_ID for f in frames)
    assert {f["phase"] for f in frames} == set(skills.PickAndPlace.PHASES) | {"done"}
    assert frames[0]["t"] == 0.0 and frames[1]["t"] == pytest.approx(0.02)
    assert set(frames[10]["observation"]) >= {"ee", "yaw", "width", "holding", "setpoint", "objects"}
    assert set(frames[10]["action"]) >= {"goal", "speed", "yaw", "grip"}
    assert any(f["observation"]["holding"] == "block_1" for f in frames)
    assert events[-1]["kind"] == "skill_done" and events[-1]["skill_id"] == sid
    notes = [line for line in lines if line["type"] == "note"]
    assert notes == [{"type": "note", "kind": "skill_start", "skill_id": sid, "arm": ARM_ID, "skill": "pick_and_place", "args": ARGS}]
    assert lines.index(notes[0]) < lines.index(frames[0])


def test_bad_start_is_refused_without_a_failed_event():
    srv = server()
    bad = {"object": "block_9", "place": "slot_2"}
    why = srv.precondition(ARM_ID, "pick_and_place", bad)
    assert why == "not_in_view: there is no object block_9 in the world"
    with pytest.raises(ValueError, match="block_9"):
        srv.start(ARM_ID, "pick_and_place", bad)
    srv.run_for(1.0)
    assert srv.events == [] and srv.world().arms[ARM_ID].mode == "idle"
    assert srv.precondition(ARM_ID, "pick_and_place", {"object": "block_1"}) == "precondition: pick_and_place needs place"
    assert srv.precondition(ARM_ID, "pick_and_place", {**ARGS, "speed": 1}).startswith("precondition: pick_and_place takes no argument")
    assert srv.precondition(ARM_ID, "pick_and_place", {"object": "block_1", "place": "bin_7"}) == "precondition: there is no place bin_7 in the world"
    assert srv.precondition(ARM_ID, "push", {}) == "precondition: push needs object, direction, distance"
    assert srv.precondition(ARM_ID, "push", {**PUSH, "direction": "up"}).startswith("precondition: direction must be one of left, right")
    assert srv.precondition(ARM_ID, "push", {**PUSH, "distance": 0.5}) == "precondition: distance must be between 0.02 and 0.3"
    assert srv.precondition(ARM_ID, "wiggle", {}) == f"precondition: {ARM_ID} has no skill wiggle"
    assert srv.precondition("arm_1", "survey", {}) == "precondition: no arm arm_1"
    assert srv.precondition(ARM_ID, "hold", {"seconds": 30}) == "precondition: seconds must be between 0.5 and 10"
    assert srv.precondition(ARM_ID, "pick_and_place", ARGS) is None


def test_occupied_slot_is_refused_and_tray_group_resolves_to_a_free_slot():
    srv = server([scene.Block("block_1", (0.30, -0.08)), scene.Block("block_2", scene.Tray().slot_xy(2))])
    assert srv.world().objects["block_2"].where == "slot_2"
    assert srv.precondition(ARM_ID, "pick_and_place", ARGS) == "blocked: tray slot 2 is occupied by block_2"
    sid = srv.start(ARM_ID, "pick_and_place", {"object": "block_1", "place": "tray"})
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"
    srv.run_for(0.3)
    assert srv.world().objects["block_1"].where == "slot_0"


def test_hold_and_survey_skills():
    srv = server()
    sid = srv.start(ARM_ID, "hold", {"seconds": 0.6})
    assert srv.run_until(srv.finished(sid), 5) and srv.status(sid).state == "done"
    assert 0.5 < srv.t < 0.8
    sid2 = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid2, "close"), 10)
    sid3 = srv.start(ARM_ID, "survey", {})          # replaces the running skill
    assert srv.status(sid2).state == "cancelled" and srv.status(sid2).reason == f"cancelled: replaced by {sid3}"
    assert srv.run_until(srv.finished(sid3), 10) and srv.status(sid3).state == "done"
    assert srv.world().arms[ARM_ID].z == pytest.approx(TRAVEL_Z, abs=0.01)
    assert not any(e.kind == "skill_failed" for e in srv.events)


# ---------------------------------------------------------------- failure codes


def failed(srv, sid, code: str):
    st = srv.status(sid)
    assert st.state == "failed" and st.reason.startswith(code + ": "), st
    ev = [e for e in srv.events if e.kind == "skill_failed"]
    assert len(ev) == 1 and ev[0].skill_id == sid and ev[0].text == st.reason
    assert st.phase_text.startswith("failed in phase '")
    return st


GRASPING = [("pick_and_place", ARGS), ("stack_on", STACK), ("hand_over", HAND)]


@pytest.mark.parametrize("skill,args", GRASPING)
def test_grasp_failed_when_the_block_is_gone_at_close(skill, args):
    srv = two_blocks()
    sid = srv.start(ARM_ID, skill, args)
    assert srv.run_until(in_phase(srv, sid, "descend"), 10)
    srv.run_for(0.6)
    srv.sim_world.move("block_1", 0.20, 0.20)        # taken away under the descending fingers, no retarget
    assert srv.run_until(srv.finished(sid), 10)
    st = failed(srv, sid, "grasp_failed")
    assert "not between them" in st.reason and srv.world().arms[ARM_ID].mode == "idle"


@pytest.mark.parametrize("skill,args", GRASPING)
def test_dropped_when_the_block_leaves_the_fingers_in_carry(skill, args):
    srv = two_blocks()
    sid = srv.start(ARM_ID, skill, args)
    assert srv.run_until(in_phase(srv, sid, "carry"), 20)
    srv.sim_world.move("block_1", 0.20, 0.20)
    assert srv.run_until(srv.finished(sid), 2)
    st = failed(srv, sid, "dropped")
    assert "during carry" in st.reason and srv.t < 15


def test_unreachable_after_a_retarget_outside_the_workspace():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.run_for(0.2)
    srv.sim_world.move("block_1", 0.50, 0.20)
    srv.retarget(sid)
    srv.run_for(0.1)
    failed(srv, sid, "unreachable")
    srv2 = server([scene.Block("block_1", (0.50, 0.20))])
    assert srv2.precondition(ARM_ID, "pick_and_place", ARGS).startswith("unreachable: block_1 at (0.50, 0.20)")


class FakeIO:
    """A policy backend with no physics: the EE goes where `step` says. Shows the policies need only
    skills.ArmIO (D1 points them at the real arm the same way)."""
    t = 0.0
    table_z = -0.02
    travel_z = 0.10
    max_width = 0.088

    def __init__(self, ee=(0.22, 0.0, 0.10)):
        self._ee, self._yaw, self._w, self.goal, self.frozen = tuple(ee), math.pi / 2, 0.088, None, False

    def ee(self): return self._ee
    def ee_yaw(self): return self._yaw
    def arrived(self): return False
    def width(self): return self._w
    def reachable(self, xyz): return True
    def obj(self, oid): return skills.ObjPose(0.30, -0.08, 0.0, 0.5, 0.04)
    def objects(self): return {"block_1": self.obj("block_1")}
    def where(self, oid): return "table"
    def place(self, pid): return (0.36, 0.0)
    def hands(self): return []
    def held(self): return None
    def goto(self, xyz, speed, yaw=None): self.goal = xyz
    def grip(self, w): self._w = w
    def freeze(self): self.frozen = True
    def unfreeze(self): self.frozen = False

    def run(self, skill, seconds, move=0.0):
        for _ in range(int(seconds / 0.02)):
            self.t += 0.02
            self._ee = (self._ee[0], self._ee[1], self._ee[2] + move)
            skill.tick(self)
            if skill.finished:
                return


def test_stalled_when_the_ee_makes_no_progress():
    io = FakeIO()
    s = skills.PickAndPlace("sk1", ARM_ID, ARGS, skills.Params(stall_s=0.5))
    io.run(s, 2.0)
    assert s.state == "failed" and s.reason.startswith("stalled: no progress approaching block_1") and io.frozen
    assert 0.5 < io.t < 0.7


def test_timeout_when_the_skill_moves_but_never_arrives():
    io = FakeIO()
    s = skills.PickAndPlace("sk1", ARM_ID, ARGS, skills.Params(timeout_s=1.0, stall_s=0.5))
    io.run(s, 3.0, move=0.004)                       # 4 mm a tick: never a stall, never there
    assert s.state == "failed" and s.reason.startswith("timeout: 1 s of motion, still in phase 'approach'")
    assert io.t == pytest.approx(1.02, abs=0.03)


def test_held_time_does_not_count_toward_the_timeout():
    io = FakeIO()
    s = skills.PickAndPlace("sk1", ARM_ID, ARGS, skills.Params(timeout_s=1.0, stall_s=5.0))
    io.run(s, 0.5, move=0.004)
    s.hold(io)
    assert s.state == "holding" and io.frozen
    io.run(s, 5.0)
    assert s.state == "holding" and s.status().phase_text == "held: moving above block_1 at travel height"
    s.resume(io)
    io.run(s, 0.4, move=0.004)
    assert s.state == "running"
    io.run(s, 0.3, move=0.004)
    assert s.state == "failed" and s.reason.startswith("timeout")


# ---------------------------------------------------------------- K4: stack_on


def test_stack_on_phases_where_and_judge():
    srv = two_blocks()
    sid = srv.start(ARM_ID, "stack_on", STACK)
    assert phases_seen(srv, sid) == list(skills.Grasping.PHASES) + ["done"]
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == "block_1 stacked on block_2" and st.reason is None
    srv.run_for(0.3)
    w = srv.world()
    assert w.objects["block_1"].where == "on:block_2" and w.top_of("block_2") == "block_1"
    assert w.objects["block_2"].where == "table" and w.arms[ARM_ID].holding is None
    assert math.hypot(w.objects["block_1"].x - w.objects["block_2"].x, w.objects["block_1"].y - w.objects["block_2"].y) < 0.01
    assert w.objects["block_1"].z - w.objects["block_2"].z == pytest.approx(0.04, abs=0.003)
    ok, why = trials.judge_stack(w, srv.manifest(), "block_1", "block_2")
    assert ok, why
    assert [e.kind for e in srv.events] == ["skill_done"]


def test_where_reports_on_only_when_resting_centred_on_top():
    srv = two_blocks()
    top = scene.TABLE_Z + 0.06                                     # a 4 cm cube's centre when it sits on another
    srv.sim_world.move("block_1", 0.32 + 0.012, 0.10, z=top)
    srv.run_for(0.3)
    assert srv.world().objects["block_1"].where == "on:block_2"
    srv.sim_world.move("block_1", 0.32 + 0.02, 0.10, z=top)       # overhanging past 1.5 cm: not "on" it
    assert srv.world().objects["block_1"].where == "table"
    srv.sim_world.move("block_1", 0.32, 0.10, z=top + 0.03)        # hanging in the air above it: not "on" it
    assert srv.world().objects["block_1"].where == "table"
    srv.sim_world.move("block_1", 0.30, -0.08)
    srv.run_for(0.2)
    assert srv.world().objects["block_1"].where == "table" and srv.world().top_of("block_2") is None


def test_stack_on_is_blocked_by_a_block_already_on_top():
    third = scene.Block("block_3", (0.22, 0.15))
    srv = server([scene.Block("block_1", (0.30, -0.08), yaw=0.5), BLOCK_2, third])
    assert srv.precondition(ARM_ID, "stack_on", STACK) is None
    sid = srv.start(ARM_ID, "stack_on", STACK)
    assert srv.run_until(in_phase(srv, sid, "carry"), 20)
    srv.sim_world.move("block_3", 0.32, 0.10, z=scene.TABLE_Z + 0.06)      # someone beat us to it
    assert srv.run_until(srv.finished(sid), 5)
    st = failed(srv, sid, "blocked")
    assert st.reason == "blocked: block_2 already has block_3 on it" and st.phase in ("carry", "lower")
    assert srv.world().arms[ARM_ID].holding == "block_1"                    # still held: the brain decides
    srv2 = server([scene.Block("block_1", (0.30, -0.08)), BLOCK_2, scene.Block("block_3", (0.32, 0.10))])
    srv2.sim_world.move("block_3", 0.32, 0.10, z=scene.TABLE_Z + 0.06)
    srv2.run_for(0.3)
    assert srv2.precondition(ARM_ID, "stack_on", STACK) == "blocked: block_2 already has block_3 on it"
    assert srv2.precondition(ARM_ID, "stack_on", {"object": "block_1", "onto": "block_1"}) == "precondition: block_1 cannot be stacked on itself"
    assert srv2.precondition(ARM_ID, "stack_on", {"object": "block_1", "onto": "block_9"}) == "not_in_view: there is no object block_9 in the world"


def test_stack_on_refuses_a_base_too_tall_or_out_of_reach():
    slab = scene.Block("slab", (0.32, 0.10), size=0.08, mass=0.2)
    srv = server([scene.Block("block_1", (0.30, -0.08)), slab])
    why = srv.precondition(ARM_ID, "stack_on", {"object": "block_1", "onto": "slab"})
    assert why == "unreachable: the top of slab is 0.080 m above the table, too high to stack on"
    srv2 = server([scene.Block("block_1", (0.30, -0.08)), scene.Block("block_2", (0.50, 0.10))])
    assert srv2.precondition(ARM_ID, "stack_on", STACK).startswith("unreachable: block_2 at (0.50, 0.10)")


def test_stack_on_follows_a_moved_base_on_retarget():
    srv = two_blocks()
    sid = srv.start(ARM_ID, "stack_on", STACK)
    assert srv.run_until(in_phase(srv, sid, "lower"), 20)
    srv.sim_world.move("block_2", 0.36, 0.04)
    srv.retarget(sid)
    assert phase(srv, sid) == "carry" and srv.status(sid).heading == pytest.approx((0.36, 0.04), abs=1e-6)
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"
    srv.run_for(0.3)
    assert srv.world().objects["block_1"].where == "on:block_2"


# ---------------------------------------------------------------- K4: push


def test_push_phases_distance_and_judge():
    srv = server([scene.Block("block_1", (0.30, 0.0), yaw=0.4)])
    sid = srv.start(ARM_ID, "push", PUSH)
    assert phases_seen(srv, sid) == list(skills.Push.PHASES) + ["done"]
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text.startswith("block_1 pushed") and st.phase_text.endswith("cm left")
    srv.run_for(0.3)
    w = srv.world()
    o = w.objects["block_1"]
    assert o.y - 0.0 == pytest.approx(0.08, abs=0.01) and abs(o.x - 0.30) < 0.015 and o.where == "table"
    assert w.arms[ARM_ID].gripper_width < 0.003 and w.arms[ARM_ID].z == pytest.approx(TRAVEL_Z, abs=0.01)
    ok, why = trials.judge_push(w, srv.manifest(), "block_1", (0.30, 0.0), "left", 0.08)
    assert ok, why
    assert not any(e.kind == "safety_trip" for e in srv.events)


@pytest.mark.parametrize("direction,dx,dy", [("right", 0, -1), ("toward_robot", -1, 0), ("away_from_robot", 1, 0)])
def test_push_directions_are_the_base_frame(direction, dx, dy):
    srv = server([scene.Block("block_1", (0.30, 0.0), yaw=-0.3)])
    args = {"object": "block_1", "direction": direction, "distance": 0.05}
    sid = srv.start(ARM_ID, "push", args)
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"
    srv.run_for(0.3)
    o = srv.world().objects["block_1"]
    assert (o.x - 0.30) * dx + o.y * dy == pytest.approx(0.05, abs=0.01)
    assert trials.judge_push(srv.world(), srv.manifest(), "block_1", (0.30, 0.0), direction, 0.05)[0]


def test_push_blocked_by_an_obstacle_fails_before_the_lag_trip():
    wall = scene.Block("wall", (0.30, 0.10), size=0.08, mass=20.0, rgba=(0.3, 0.3, 0.3, 1))
    srv = server([scene.Block("block_1", (0.30, 0.0)), wall])
    sid = srv.start(ARM_ID, "push", {"object": "block_1", "direction": "left", "distance": 0.10})
    assert srv.run_until(srv.finished(sid), 20)
    st = failed(srv, sid, "blocked")
    assert st.reason.startswith("blocked: block_1 stopped after ") and "something is in the way" in st.reason
    assert st.phase == "slide" and 0.03 < srv.world().objects["block_1"].y < 0.05
    assert not any(e.kind == "safety_trip" for e in srv.events)


def test_push_unreachable_path_is_refused_and_fails_on_retarget():
    srv = server([scene.Block("block_1", (0.30, 0.10))])
    why = srv.precondition(ARM_ID, "push", {"object": "block_1", "direction": "left", "distance": 0.20})
    assert why == "unreachable: pushing block_1 20 cm left ends at (0.30, 0.29), outside the arm's workspace"
    why = srv.precondition(ARM_ID, "push", {"object": "block_1", "direction": "away_from_robot", "distance": 0.15})
    assert why.startswith("unreachable: pushing block_1 15 cm away_from_robot ends at")
    sid = srv.start(ARM_ID, "push", {"object": "block_1", "direction": "left", "distance": 0.08})
    srv.run_for(0.3)
    srv.sim_world.move("block_1", 0.30, 0.20)
    srv.retarget(sid)
    srv.run_for(0.1)
    failed(srv, sid, "unreachable")


def test_push_refuses_a_held_object_and_stalls_when_the_block_is_taken_mid_slide():
    srv = server([scene.Block("block_1", (0.30, 0.0))])
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "carry"), 20)
    assert srv.precondition(ARM_ID, "push", PUSH) == "precondition: block_1 is held by arm_0"
    srv2 = server([scene.Block("block_1", (0.30, 0.0))])
    sid2 = srv2.start(ARM_ID, "push", PUSH)
    assert srv2.run_until(in_phase(srv2, sid2, "slide"), 20)
    srv2.run_for(0.6)
    srv2.sim_world.move("block_1", 0.20, -0.20)
    assert srv2.run_until(srv2.finished(sid2), 10)
    st = failed(srv2, sid2, "stalled")
    assert "slipped off them" in st.reason


# ---------------------------------------------------------------- K4: hand_over


def test_hand_over_waits_for_a_held_out_hand_then_releases():
    srv = server()
    sid = srv.start(ARM_ID, "hand_over", HAND)
    assert srv.run_until(in_phase(srv, sid, "release"), 20) and phase(srv, sid) == "release"
    a = srv.world().arms[ARM_ID]
    assert (a.x, a.y) == pytest.approx(skills.Params().offer_xy, abs=0.01) and a.z == pytest.approx(skills.Params().offer_z, abs=0.01)
    srv.run_for(1.0)
    st = srv.status(sid)
    assert st.state == "running" and st.phase_text == "holding block_1 out at (0.40, 0.00), waiting for a hand beneath it"
    hand_under(srv, dz=0.10)                                 # too far below the fingertips to count
    srv.run_for(0.5)
    assert "waiting for a hand beneath it" in srv.status(sid).phase_text
    hand_under(srv, dz=0.05, held_out=False)                 # near, but not offered
    srv.run_for(0.5)
    assert srv.status(sid).phase_text == "holding block_1 out; a hand is beneath it but not held out to take it"
    assert srv.world().arms[ARM_ID].holding == "block_1" and srv.world().hands[0].held_out is False
    hand_under(srv, dz=0.05, held_out=True)
    assert srv.run_until(srv.finished(sid), 5)
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == "block_1 released into the hand"
    srv.run_for(0.3)
    w = srv.world()
    assert w.arms[ARM_ID].holding is None and w.objects["block_1"].where != f"gripper:{ARM_ID}"
    assert w.arms[ARM_ID].z == pytest.approx(TRAVEL_Z, abs=0.01)
    assert [e.kind for e in srv.events] == ["skill_done"]


def test_hand_over_times_out_waiting_for_a_hand():
    srv = server()
    sid = srv.start(ARM_ID, "hand_over", HAND)
    assert srv.run_until(in_phase(srv, sid, "release"), 20)
    t_wait = srv.t
    srv.run_for(14.0)
    assert srv.status(sid).state == "running" and "waiting for a hand" in srv.status(sid).phase_text
    assert srv.run_until(srv.finished(sid), 3)
    st = failed(srv, sid, "timeout")
    assert st.reason == "timeout: held block_1 out for 15 s and no hand came to take it"
    assert srv.t - t_wait == pytest.approx(15.0, abs=0.1) and srv.world().arms[ARM_ID].holding == "block_1"


def test_hand_over_block_taken_from_the_fingers_while_waiting_is_dropped():
    srv = server()
    sid = srv.start(ARM_ID, "hand_over", HAND)
    assert srv.run_until(in_phase(srv, sid, "release"), 20)
    srv.run_for(0.5)
    srv.sim_world.move("block_1", 0.20, 0.20)
    assert srv.run_until(srv.finished(sid), 2)
    st = failed(srv, sid, "dropped")
    assert st.reason.startswith("dropped: block_1 left the fingers while held out for a hand")


@pytest.mark.parametrize("skill,args", [("stack_on", STACK), ("hand_over", HAND)])
def test_grasping_skills_begin_from_a_held_block(skill, args):
    srv = two_blocks()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "carry"), 20)
    sid2 = srv.start(ARM_ID, skill, args)                      # a correction mid-carry
    assert srv.status(sid).state == "cancelled"
    seen = []
    def watch():
        p = phase(srv, sid2)
        if not seen or seen[-1] != p:
            seen.append(p)
        if skill == "hand_over" and p == "release" and not srv.world().hands:
            hand_under(srv)
        return srv.status(sid2).state in ("done", "failed")
    assert srv.run_until(watch, 25)
    assert seen[:2] == ["approach", "lift"] and "descend" not in seen and seen[-1] == "done"
    assert srv.status(sid2).state == "done"
    srv.run_for(0.3)
    if skill == "stack_on":
        assert srv.world().objects["block_1"].where == "on:block_2"
    assert srv.world().arms[ARM_ID].holding is None


# ---------------------------------------------------------------- K4: hold / resume mid-skill, every skill


@pytest.mark.parametrize("skill,args,hold_in", [("stack_on", STACK, "lower"), ("push", PUSH, "slide"), ("hand_over", HAND, "carry")])
def test_hold_mid_skill_freezes_and_resume_finishes(skill, args, hold_in):
    srv = two_blocks()
    sid = srv.start(ARM_ID, skill, args)
    assert srv.run_until(in_phase(srv, sid, hold_in), 20) and phase(srv, sid) == hold_in
    srv.run_for(0.3)
    srv.hold(ARM_ID)
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("held: ") and st.phase == hold_in
    sp0 = srv.arms[ARM_ID].mover.setpoint
    srv.run_for(0.2)
    a0 = srv.world().arms[ARM_ID]
    o0 = srv.world().objects["block_1"]
    srv.run_for(1.0)
    a1, o1 = srv.world().arms[ARM_ID], srv.world().objects["block_1"]
    assert srv.arms[ARM_ID].mover.setpoint == sp0 and a1.mode == "holding"
    assert math.dist((a0.x, a0.y, a0.z), (a1.x, a1.y, a1.z)) < 0.003
    assert math.dist((o0.x, o0.y, o0.z), (o1.x, o1.y, o1.z)) < 0.003
    if skill != "push":
        assert a1.holding == "block_1"
    srv.resume(ARM_ID)
    assert srv.status(sid).state == "running" and phase(srv, sid) == hold_in
    def go():
        if skill == "hand_over" and phase(srv, sid) == "release" and not srv.world().hands:
            hand_under(srv)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(go, 20) and srv.status(sid).state == "done", srv.status(sid)
    srv.run_for(0.3)
    w, m = srv.world(), srv.manifest()
    if skill == "stack_on":
        assert trials.judge_stack(w, m, "block_1", "block_2")[0]
    elif skill == "push":
        assert trials.judge_push(w, m, "block_1", (0.30, -0.08), "left", 0.08)[0]
    else:
        assert w.arms[ARM_ID].holding is None and w.arms[ARM_ID].z == pytest.approx(TRAVEL_Z, abs=0.01)


# ---------------------------------------------------------------- K4: the neighbour-adaptive opening


def test_open_width_shrinks_for_neighbours_within_limits():
    p = skills.Params()
    at = (0.36, 0.0)
    far = skills.ObjPose(0.36, 0.20, 0, 0, 0.04)
    assert skills.open_width(0.04, [], at, 0.088, p) == pytest.approx(0.07)
    assert skills.open_width(0.04, [far], at, 0.088, p) == pytest.approx(0.07)
    at8 = skills.ObjPose(0.36, 0.08, 0, 0, 0.04)                      # the tray's pitch: unchanged
    assert skills.open_width(0.04, [at8], at, 0.088, p) == pytest.approx(0.07)
    at7 = skills.ObjPose(0.36, 0.07, 0, 0, 0.04)                      # 2 * (7 - 2 - 1.4 - 0.5) cm
    assert skills.open_width(0.04, [at7], at, 0.088, p) == pytest.approx(0.062)
    at6 = skills.ObjPose(0.36, -0.06, 0, 0, 0.04)                     # the floor: block + 8 mm
    assert skills.open_width(0.04, [far, at6], at, 0.088, p) == pytest.approx(0.048)
    assert skills.open_width(0.08, [], at, 0.088, p) == pytest.approx(0.088)   # capped at the jaws


def test_pick_from_a_slot_with_neighbours_6_cm_away():
    tray = scene.Tray(step=(0.0, -0.06))
    srv = server([scene.Block(f"block_{i}", tray.slot_xy(i), yaw=0.1 * i) for i in (0, 1, 2)], tray=tray)
    w0 = srv.world()
    assert [w0.objects[f"block_{i}"].where for i in (0, 1, 2)] == ["slot_0", "slot_1", "slot_2"]
    sid = srv.start(ARM_ID, "pick_and_place", {"object": "block_1", "place": "slot_4"})
    assert srv.run_until(in_phase(srv, sid, "close"), 10)
    assert srv._runs[sid].open_w == pytest.approx(0.048)
    assert srv.run_until(srv.finished(sid), 20) and srv.status(sid).state == "done"
    assert srv._runs[sid].release_w == pytest.approx(0.07)              # no neighbours at slot_4
    srv.run_for(0.3)
    w = srv.world()
    ok, why = trials.judge(w, srv.manifest(), "block_1", "slot_4")
    assert ok, why
    for i in (0, 2):                                                    # the neighbours were not touched
        assert math.dist((w.objects[f"block_{i}"].x, w.objects[f"block_{i}"].y),
                         (w0.objects[f"block_{i}"].x, w0.objects[f"block_{i}"].y)) < 0.002
        assert w.objects[f"block_{i}"].where == f"slot_{i}"
    assert not any(e.kind == "safety_trip" for e in srv.events)


@pytest.mark.parametrize("skill", ["stack_on", "push"])
def test_three_random_trials_per_skill_succeed(skill):
    import random
    for index in range(3):
        trial = trials.sample(random.Random(200 + index), index, 5, skill)
        v = trials.run_trial(trial)
        assert v.ok, (trial, v)
        assert v.state == "done" and v.sim_s < 15
