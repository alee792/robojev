"""K2: the MuJoCo WidowX behind the robot protocol. Physics, no rendering, no clock: every test steps
the server itself (realtime=False), so the run is deterministic and takes seconds."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from skills_sim import catalog, scene, skills, trials  # noqa: E402
from skills_sim.protocol import RobotServer  # noqa: E402
from skills_sim.recorder import JsonlRecorder  # noqa: E402
from skills_sim.sim_server import ARM_ID, TRAVEL_Z, SimRobotServer  # noqa: E402

pytestmark = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(),
                                reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")

ARGS = {"object": "block_1", "place": "slot_2"}


def server(blocks=None, **kw) -> SimRobotServer:
    blocks = blocks if blocks is not None else [scene.Block("block_1", (0.30, -0.08), yaw=0.5)]
    srv = SimRobotServer(scene.SceneSpec(blocks=blocks), realtime=False, **kw)
    srv.events = []
    srv.subscribe(srv.events.append)
    return srv


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
    assert {s.name for s in man.skills} == {"pick_and_place", "hold", "survey"}
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
    assert st.state == "failed" and st.reason.startswith("stalled: STOP")
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
    lines = [json.loads(l) for l in (tmp_path / "run.jsonl").read_text().splitlines()]
    frames = [l for l in lines if l["type"] == "frame"]
    events = [l for l in lines if l["type"] == "event"]
    assert summary["frames"] == len(frames) and 200 < len(frames) < 800
    assert all(f["skill_id"] == sid and f["arm"] == ARM_ID for f in frames)
    assert {f["phase"] for f in frames} == set(skills.PickAndPlace.PHASES) | {"done"}
    assert frames[0]["t"] == 0.0 and frames[1]["t"] == pytest.approx(0.02)
    assert set(frames[10]["observation"]) >= {"ee", "yaw", "width", "holding", "setpoint", "objects"}
    assert set(frames[10]["action"]) >= {"goal", "speed", "yaw", "grip"}
    assert any(f["observation"]["holding"] == "block_1" for f in frames)
    assert events[-1]["kind"] == "skill_done" and events[-1]["skill_id"] == sid


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
    assert srv.precondition(ARM_ID, "push", {}) == f"precondition: {ARM_ID} has no skill push"
    assert srv.precondition("arm_1", "survey", {}) == "precondition: no arm arm_1"
    assert srv.precondition(ARM_ID, "hold", {"seconds": 30}) == "precondition: hold takes 0.5 to 10 seconds"
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
    assert srv.status(sid2).state == "failed" and srv.status(sid2).reason == f"precondition: replaced by {sid3}"
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


def test_grasp_failed_when_the_block_is_gone_at_close():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "descend"), 10)
    srv.run_for(0.6)
    srv.sim_world.move("block_1", 0.20, 0.20)        # taken away under the descending fingers, no retarget
    assert srv.run_until(srv.finished(sid), 10)
    st = failed(srv, sid, "grasp_failed")
    assert "not between them" in st.reason and srv.world().arms[ARM_ID].mode == "idle"


def test_dropped_when_the_block_leaves_the_fingers_in_carry():
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
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
    def place(self, pid): return (0.36, 0.0)
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
