"""D1: the real WidowX server (robots/widowx_real) built and tested against the fake driver. The
policies are the sim's; the arm is v1's RealArm with FakeTrossenDriver under it. The `server`
fixture runs the server, RealArm's own I/O thread and the fake driver on one virtual clock that only
the test advances (robots/widowx_real/clock.py: `run_until`/`run_for` tick the policies and step
time), so every outcome, down to the number of frames recorded, is the same however loaded the
machine is; on the wall clock a busy machine starved the 50 Hz policy thread and the 20 Hz arm thread.
The threaded wall-clock server is exercised by the pre-flight and conformance tests. Tests of the real
driver path skip when trossen_arm is not installed."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

import robojev.arm.real
from robojev import catalog
from robojev import conformance as trials
from robojev.arm.real import STAGED, SLEEP
from robojev.protocol import RobotServer
from robojev.recording.recorder import JsonlRecorder
from robojev.robots.widowx_real import orientation as ori
from robojev.robots.widowx_real.arm import (ARM_ID, MAX_WIDTH, OPEN_Q_MIN, REAL_PARAMS, TRAVEL_ABOVE_TABLE, OrientingDriver,
                                            day_one_config)
from robojev.robots.widowx_real.clock import LockstepClock
from robojev.robots.widowx_real.fake_driver import (GRIP_MAX, FakeDriverFactory, FakeTrossenDriver, install_fake_trossen_arm,
                                                    real_driver_available)
from robojev.robots.widowx_real.server import PreflightError, RealRobotServer
from robojev.robots.widowx_real.world import PerceptionWorld, StaticWorld, load_blocks
from robojev.robots.widowx_sim import scene, skills

TABLE_Z = -0.02
TRAVEL_Z = TABLE_Z + TRAVEL_ABOVE_TABLE
BLOCK = scene.Block("block_1", (0.30, -0.08), yaw=0.5)
BLOCK_2 = scene.Block("block_2", (0.32, 0.10), yaw=-0.2)
ARGS = {"object": "block_1", "place": "slot_2"}


def config():
    cfg = day_one_config(TABLE_Z)
    return replace(cfg, safety=replace(cfg.safety, effort_baseline_s=0.2))     # 1 s on hardware; the fake has no noise


@pytest.fixture
def server(monkeypatch):
    made = []

    def make(blocks=(BLOCK,), recorder=None, **kw) -> RealRobotServer:
        clock = LockstepClock()
        monkeypatch.setattr(robojev.arm.real, "time", clock.time)    # RealArm reads its module's `time`; undone after close
        world = StaticWorld(list(blocks), table_z=TABLE_Z)
        fac = FakeDriverFactory(lambda: srv.world_src, clock=clock.now)
        srv = RealRobotServer("127.0.0.1", world, config(), recorder=recorder, driver_factory=fac, clock=clock, **kw)
        srv.factory = fac
        srv.events = []
        srv.subscribe(srv.events.append)
        srv.preflight("pytest", teleop_off=True)
        made.append(srv)
        return srv.open()

    yield make
    for s in made:
        s.close()


def phase(srv, sid) -> str:
    return srv.status(sid).phase


def in_phase(srv, sid, name):
    return lambda: phase(srv, sid) == name or srv.status(sid).state in ("done", "failed")


def phases_seen(srv, sid, timeout=25) -> list[str]:
    seen = []

    def watch():
        p = phase(srv, sid)
        if not seen or seen[-1] != p:
            seen.append(p)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(watch, timeout)
    return seen


# ---------------------------------------------------------------- orientation maths


def test_down_orientation_known_cases():
    assert ori.down_orientation(math.pi / 2) == pytest.approx((0.0, math.pi / 2, 0.0))        # v1's straight-down triple
    assert ori.down_orientation(math.pi / 2, 1.309) == pytest.approx((0.0, 1.309, 0.0))       # v1's verified 75 deg
    aa = np.array(ori.down_orientation(math.pi / 2 + math.pi))                               # Rz(pi) Ry(pi/2): 180 deg about (1,0,-1)
    assert np.linalg.norm(aa) == pytest.approx(math.pi)
    assert abs(float(aa @ np.array([1, 0, -1]) / math.sqrt(2))) == pytest.approx(math.pi)
    R = ori.down_matrix(math.pi / 2 + math.pi / 2)                                           # Rz(pi/2) Ry(pi/2), by hand
    assert np.allclose(R, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]])


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, math.pi / 2, 2.5, -3.0])
def test_down_orientation_points_fingers_down_with_the_closing_axis_along_yaw(yaw):
    R = ori.from_angle_axis(ori.down_orientation(yaw))
    assert np.allclose(R @ [1, 0, 0], [0, 0, -1])                                             # EE x (fingers) down
    assert np.allclose(R @ [0, 1, 0], [math.cos(yaw), math.sin(yaw), 0])                      # EE y (closing axis) along yaw
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert skills.wrap(ori.closing_yaw_of(ori.down_orientation(yaw)) - yaw) == pytest.approx(0.0, abs=1e-9)
    assert ori.pitch_of(ori.down_orientation(yaw)) == pytest.approx(math.pi / 2)
    assert np.allclose(R, ori.rot_z(yaw - math.pi / 2) @ ori.rot_y(math.pi / 2))


def test_angle_axis_roundtrip_including_the_pi_singularity():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(size=3)
        v *= rng.uniform(0, math.pi) / np.linalg.norm(v)
        R = ori.from_angle_axis(v)
        assert np.allclose(ori.from_angle_axis(ori.to_angle_axis(R)), R, atol=1e-8)
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [-1, 0, 1]):
        v = np.array(axis, float) / np.linalg.norm(axis) * math.pi
        R = ori.from_angle_axis(v)
        assert np.allclose(ori.from_angle_axis(ori.to_angle_axis(R)), R, atol=1e-8)
    assert ori.to_angle_axis(np.eye(3)) == (0.0, 0.0, 0.0)
    assert ori.pitch_of((0.0, 0.0, 0.0)) == pytest.approx(0.0) and ori.pitch_of((0.0, 1.0, 0.0)) == pytest.approx(1.0)


def test_orienting_driver_rewrites_the_pitch_only_stream_and_rate_limits_yaw():
    clock = [0.0]
    d = FakeTrossenDriver(clock=lambda: clock[0])
    o = OrientingDriver(d, yaw_rate=1.0, clock=lambda: clock[0])
    o.set_yaw(math.pi / 2 + 0.5)
    sent = []
    d.set_cartesian_positions = lambda pose, space, gt, blocking: sent.append(list(pose))
    for _ in range(11):                               # 10 x 50 ms at 1 rad/s = 0.5 rad, the whole way
        o.set_cartesian_positions([0.3, 0.0, 0.1, 0.0, 1.2, 0.0], None, 0.1, False)
        clock[0] += 0.05
    yaws = [ori.closing_yaw_of(p[3:6]) for p in sent]
    assert yaws[0] == pytest.approx(math.pi / 2)                                  # the first call has no dt
    assert yaws[1] == pytest.approx(math.pi / 2 + 0.05) and yaws[-1] == pytest.approx(math.pi / 2 + 0.5)
    assert all(ori.pitch_of(p[3:6]) == pytest.approx(1.2) for p in sent)          # RealArm's pitch is kept
    assert o.arrived() and o.rewrites == 11 and o.passed == 0
    o.set_cartesian_positions([0.3, 0.0, 0.1, 0.1, 1.2, 0.2], None, 0.5, False)   # the relax-to-measured-pose command: untouched
    o.set_cartesian_positions([0.3, 0.0, 0.1, 0.0, 0.5, 0.0], None, 2.5, True)    # a blocking staging move: untouched
    assert sent[-2][3:6] == [0.1, 1.2, 0.2] and sent[-1][3:6] == [0.0, 0.5, 0.0] and o.passed == 2
    assert o.get_all_positions() == [0.0] * 7                                      # everything else is forwarded


# ---------------------------------------------------------------- the world and the fake driver


def test_static_world_bookkeeping_tracks_what_the_arm_did():
    w = StaticWorld([BLOCK, BLOCK_2], table_z=TABLE_Z)
    p = skills.Params()
    assert w.where("block_1", {}) == "table" and w.under((0.30, -0.08, TABLE_Z + 0.005)) == "block_1"
    assert w.under((0.30, -0.08, TRAVEL_Z)) is None and w.under((0.36, -0.04, TABLE_Z + 0.005)) is None
    w.track((0.30, -0.08, TABLE_Z + 0.005), "block_1", 0.04, p)
    assert w.where("block_1", {"block_1": "arm_0"}) == "gripper:arm_0"
    w.track((0.36, -0.04, TRAVEL_Z), "block_1", 0.04, p)                 # carried: it follows the EE
    assert w.pose("block_1").x == pytest.approx(0.36) and w.pose("block_1").z == pytest.approx(TRAVEL_Z - 0.005 + 0.02)
    w.track((0.36, -0.04, TABLE_Z + 0.009), None, 0.07, p)              # released over slot 2: it settles there
    assert w.where("block_1", {}) == "slot_2" and w.pose("block_1").z == pytest.approx(TABLE_Z + 0.02)
    w.track((0.32, 0.10, TABLE_Z + 0.005), "block_2", 0.04, p)
    w.track((0.36, -0.04, TABLE_Z + 0.049), "block_2", 0.04, p)         # lowered onto block_1
    w.track((0.36, -0.04, TABLE_Z + 0.049), None, 0.07, p)
    assert w.where("block_2", {}) == "on:block_1" and w.pose("block_2").z == pytest.approx(TABLE_Z + 0.06)
    assert w.place_xy("tray", {}) == (w.places["slot_0"].x, w.places["slot_0"].y)
    assert w.occupant("slot_2", {}) == "block_1"
    w.track((0.36, -0.04, TRAVEL_Z), "block_1", 0.04, p)                 # lifting the base drops what sat on it
    assert w.where("block_2", {"block_1": "arm_0"}) == "slot_2" and w.pose("block_2").z == pytest.approx(TABLE_Z + 0.02)
    w.move("block_1", 0.30, 0.0)
    w.track((0.30 - 0.03, 0.0, TABLE_Z + 0.005), None, 0.0, p)          # closed pads 3 cm behind the centre: 4 mm overlap
    assert w.pose("block_1").x == pytest.approx(0.30 + 0.004) and w.pose("block_1").y == pytest.approx(0.0)
    w.track((0.27, 0.0, TRAVEL_Z), None, 0.0, p)                         # too high to touch it
    assert w.pose("block_1").x == pytest.approx(0.304)


def test_load_blocks_and_the_perception_stub(tmp_path):
    f = tmp_path / "blocks.json"
    f.write_text(json.dumps({"table_z": -0.015, "blocks": [{"name": "block_1", "xy": [0.3, -0.05], "yaw_deg": 30}],
                             "tray": {"origin": [0.36, 0.12], "step": [0, -0.08], "n": 3}}))
    blocks, tray, table_z = load_blocks(f)
    assert blocks[0].yaw == pytest.approx(math.radians(30)) and blocks[0].size == 0.04 and table_z == -0.015
    assert tray.n == 3 and set(StaticWorld.from_json(f).places) == {"slot_0", "slot_1", "slot_2", "tray"}
    f.write_text(json.dumps({"blocks": [{"name": "b", "xy": [0.3, 0]}], "tray": None}))
    assert load_blocks(f)[1] is None and StaticWorld.from_json(f).places == {}
    with pytest.raises(NotImplementedError, match="P1"):
        PerceptionWorld()


def test_fake_gripper_closes_at_grip_rate_and_stops_on_a_block():
    clock = [0.0]
    w = StaticWorld([BLOCK], table_z=TABLE_Z)
    d = FakeTrossenDriver(w, clock=lambda: clock[0])
    d.set_gripper_position(GRIP_MAX, 1.0, True)
    d.set_cartesian_positions([0.30, -0.08, TABLE_Z + 0.005, 0, math.pi / 2, 0], None, 0.1, True)
    d.set_gripper_mode("external_effort")
    d.set_gripper_external_effort(-6.0, 0.3, False)
    clock[0] += 0.2
    assert d.get_all_positions()[6] == pytest.approx(0.03)                 # 0.05 m/s per side
    clock[0] += 1.0
    assert d.get_all_positions()[6] == pytest.approx(0.02)                 # half the block's width per side
    d.set_cartesian_positions([0.30, 0.10, TABLE_Z + 0.005, 0, math.pi / 2, 0], None, 0.1, True)   # nothing there
    clock[0] += 1.0
    assert d.get_all_positions()[6] == 0.0
    d.set_cartesian_positions([0.30, 0.10, TRAVEL_Z, 0, math.pi / 2, 0], None, 0.1, False)       # non-blocking: linear over goal_time
    clock[0] += 0.05
    assert d.get_cartesian_positions()[2] == pytest.approx((TABLE_Z + 0.005 + TRAVEL_Z) / 2)


def test_the_trossen_arm_stand_in_is_enums_only():
    m = install_fake_trossen_arm()
    assert m.Mode.position and m.InterpolationSpace.cartesian and m.Model.wxai_v0 and m.StandardEndEffector.wxai_v0_follower
    if not real_driver_available():
        with pytest.raises(RuntimeError, match="not installed"):
            m.TrossenArmDriver()


@pytest.mark.skipif(not real_driver_available(), reason="trossen_arm is not installed: the real driver path is verified on hardware")
def test_real_driver_module_has_what_real_arm_uses():
    import trossen_arm
    for name in ("Mode", "InterpolationSpace", "Model", "StandardEndEffector", "TrossenArmDriver"):
        assert hasattr(trossen_arm, name)
    assert trossen_arm.Mode.position and trossen_arm.Mode.external_effort and trossen_arm.InterpolationSpace.cartesian


# ---------------------------------------------------------------- the adapter and the server


def test_manifest_matches_the_sim_shape(server):
    srv = server()
    assert isinstance(srv, RobotServer)
    man = srv.manifest()
    assert man.robot == "widowx-bay-1" and man.catalog == catalog.CATALOG_VERSION
    assert catalog.check_manifest_skills(man.skills) == []
    assert {s.name for s in man.skills} == {"pick_and_place", "stack_on", "push", "hand_over", "insert", "reorient", "hold", "survey"}
    arm = man.arms[0]
    assert arm.id == ARM_ID and arm.gripper.max_width == pytest.approx(0.08) and arm.travel_z == pytest.approx(TRAVEL_Z)
    assert arm.workspace[2][0] == pytest.approx(TABLE_Z + 0.004)
    w = srv.world()
    assert w.objects["block_1"].where == "table" and w.arms[ARM_ID].mode == "idle"
    assert w.arms[ARM_ID].gripper_width == pytest.approx(0.08) and (w.arms[ARM_ID].x, w.arms[ARM_ID].y) == (0.28, 0.0)
    assert w.arms[ARM_ID].z == pytest.approx(TRAVEL_Z)                 # RealArm's hover start is at travel height
    assert srv.arm.pitch_ok and srv.arm.snapshot().status == "live"


def test_adapter_goto_arrives_grip_closes_and_held_names_the_block(server):
    srv = server()
    io = srv.arms[ARM_ID]
    yaw = skills.grasp_yaw(0.30, -0.08, 0.5)
    io.goto((0.30, -0.08, TRAVEL_Z), 0.15, yaw=yaw)
    assert not io.arrived()
    assert srv.run_until(io.arrived, 3.0)                                            # setpoint there, EE within arrive_tol
    assert math.dist(io.ee(), (0.30, -0.08, TRAVEL_Z)) <= io.arrive_tol
    assert srv.run_until(lambda: math.dist(io.ee(), (0.30, -0.08, TRAVEL_Z)) < 0.003, 1.0) and io.arrived()
    assert abs(skills.wrap(io.ee_yaw() - yaw)) < 0.02
    assert srv.arm.mover.pitch == pytest.approx(math.pi / 2)                          # arrived() waited for the pitch ramp
    assert srv.run_until(lambda: abs(ori.pitch_of(io.snap().rot) - math.pi / 2) < 1e-3, 1.0)   # ... and the stream reached 90 deg
    io.goto((0.30, -0.08, TABLE_Z + 0.005), 0.10)
    assert srv.run_until(io.arrived, 3.0) and io.held() is None
    io.grip(0.032)                                                                   # 4 cm block minus the squeeze
    assert srv.arm._gripper_goal == pytest.approx(0.016)                             # per side: RealArm closes by force
    assert srv.run_until(lambda: io.held() == "block_1", 3.0)
    assert io.width() == pytest.approx(0.04, abs=0.003) and io.snap().holding
    assert srv.world().objects["block_1"].where == f"gripper:{ARM_ID}"
    io.grip(0.048)                                                                   # the smallest opening the policies use
    assert srv.arm._gripper_goal == pytest.approx(OPEN_Q_MIN)                        # widened past RealArm's force-close rule
    assert srv.run_until(lambda: io.width() >= 2 * OPEN_Q_MIN - 1e-6, 2.0) and io.held() is None
    io.grip(1.0)
    assert srv.arm._gripper_goal == pytest.approx(MAX_WIDTH / 2)
    assert io.reachable((0.30, 0.0, TRAVEL_Z)) and not io.reachable((0.50, 0.0, TRAVEL_Z))


def test_pick_and_place_end_to_end_and_recorded(server, tmp_path):
    rec = JsonlRecorder(tmp_path / "d1.jsonl")
    srv = server(recorder=rec)
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert phases_seen(srv, sid) == list(skills.PickAndPlace.PHASES) + ["done"]
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == "block_1 placed at slot_2"
    assert [e.kind for e in srv.events] == ["skill_done"] and srv.events[0].skill_id == sid
    w = srv.world()
    assert w.objects["block_1"].where == "slot_2" and w.arms[ARM_ID].holding is None and w.arms[ARM_ID].mode == "idle"
    assert abs(w.arms[ARM_ID].z - TRAVEL_Z) < 0.01
    assert srv.orient.rewrites > 100 and srv.orient.passed <= 2        # the yaw rode on every streamed setpoint
    ok, why = trials.judge(w, srv.manifest(), "block_1", "slot_2")
    assert ok, why
    rec.close()
    lines = [json.loads(line) for line in (tmp_path / "d1.jsonl").read_text().splitlines()]
    kinds = [ln.get("kind") for ln in lines if ln["type"] == "note"]
    assert kinds[0] == "preflight" and "skill_start" in kinds and "arm_event" in kinds
    frames = [ln for ln in lines if ln["type"] == "frame"]
    assert len(frames) > 200 and frames[-1]["phase"] == "done" and frames[-2]["phase"] == "retreat"
    assert {"ext_force", "rot", "arm_status"} <= set(frames[0]["observation"]) and "orient" in frames[0]["action"]


def test_hold_and_resume_mid_carry_keep_the_grip(server):
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "carry"), 15) and phase(srv, sid) == "carry"
    srv.hold(ARM_ID)
    srv.run_for(0.15)
    sp0 = srv.arm.mover.setpoint
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("held: carrying block_1")
    assert srv.world().arms[ARM_ID].mode == "holding" and srv.world().objects["block_1"].where == f"gripper:{ARM_ID}"
    srv.run_for(0.4)
    assert srv.arm.mover.setpoint == sp0 and srv.arm.mover.frozen         # not moving
    assert srv.status(sid).phase == "carry"
    srv.resume(ARM_ID)
    assert srv.run_until(srv.finished(sid), 15)
    assert srv.status(sid).state == "done" and srv.world().objects["block_1"].where == "slot_2"
    assert [e.kind for e in srv.events] == ["skill_done"]


def test_effort_trip_holds_and_reports_a_safety_trip(server):
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(lambda: math.dist(srv.arms[ARM_ID].ee(), (0.28, 0.0, TRAVEL_Z)) > 0.01, 3.0)
    srv.factory.driver.efforts = [60.0, 0.0, 0.0, 0.0, 0.0, 0.0]              # 60 N over the baseline, trip is 40 N
    assert srv.run_until(lambda: any(e.kind == "safety_trip" for e in srv.events), 2.0)
    trip = next(e for e in srv.events if e.kind == "safety_trip")
    assert trip.skill_id == sid and "external force deviation 60.0 N" in trip.text and trip.data["ext_force"][0] == 60.0
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("safety trip (external force deviation")
    assert srv.world().arms[ARM_ID].mode == "tripped"
    sp0 = srv.arm.mover.setpoint
    srv.run_for(0.3)
    assert srv.arm.mover.setpoint == sp0 and srv.arm.mover.frozen
    assert [e.kind for e in srv.events].count("safety_trip") == 1               # one trip, not one per tick
    srv.factory.driver.efforts = [0.0] * 6
    srv.run_for(0.3)
    assert srv.status(sid).state == "holding"                                   # a fresh reading resumes nothing
    srv.resume(ARM_ID)
    assert srv.run_until(lambda: srv.arm.mover.setpoint != sp0, 2.0)
    assert srv.status(sid).state == "running" and srv.world().arms[ARM_ID].mode == "running"


def test_heartbeat_loss_holds_every_arm(server):
    srv = server(heartbeat_timeout_s=0.3)
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.heartbeat()
    srv.run_for(0.15)
    assert srv.status(sid).state == "running" and not srv.events
    assert srv.run_until(lambda: any(e.kind == "heartbeat_lost" for e in srv.events), 1.0)
    assert srv.status(sid).state == "holding" and srv.world().arms[ARM_ID].mode == "holding"
    srv.heartbeat()
    srv.run_for(0.1)
    assert srv.status(sid).state == "holding"                                   # a heartbeat does not resume
    srv.resume(ARM_ID)

    def alive_and_finished():                                                   # the watchdog is armed: keep beating
        srv.heartbeat()
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(alive_and_finished, 20) and srv.status(sid).state == "done"
    assert [e.kind for e in srv.events].count("heartbeat_lost") == 1


def test_stop_fails_the_skill_parks_and_refuses(server):
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "descend"), 10)
    srv.stop()
    st = srv.status(sid)
    assert st.state == "failed" and st.reason.startswith("stopped:")
    assert [e.kind for e in srv.events] == ["skill_failed"] and srv.world().arms[ARM_ID].mode == "stopped"
    with pytest.raises(ValueError, match="stopped"):
        srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.hold(ARM_ID)
    srv.resume(ARM_ID)                                                          # no-ops, no errors
    srv._parking.join(5.0)
    calls = srv.factory.driver.calls
    assert calls[-3:] == [("set_all_positions", tuple(STAGED), 3.0, True), ("set_all_positions", tuple(SLEEP), 3.0, True), ("cleanup",)]
    assert srv.arm.snapshot().status == "stopped" and srv.factory.driver.cleaned_up


def test_preflight_gates_open():
    world = StaticWorld([BLOCK], table_z=TABLE_Z)
    srv = RealRobotServer("127.0.0.1", world, config(), driver_factory=FakeDriverFactory(world))
    with pytest.raises(PreflightError, match="preflight"):
        srv.open()
    srv.preflight("someone", teleop_off=False)
    with pytest.raises(PreflightError, match="teleop"):
        srv.open()
    srv.preflight("", teleop_off=True)
    with pytest.raises(PreflightError, match="operator"):
        srv.open()
    rec = srv.preflight("someone", teleop_off=True, estop_tested=True)
    assert rec.problems() == [] and srv.preflight_record is rec
    with srv:
        assert srv.world().arms[ARM_ID].mode == "idle"


def test_conformance_runs_the_same_trials_on_the_real_server_with_the_fake_driver(tmp_path, capsys):
    f = tmp_path / "blocks.json"
    f.write_text(json.dumps({"table_z": TABLE_Z, "blocks": [{"name": "block_1", "xy": [0.30, -0.08], "yaw_deg": 20}]}))
    out = tmp_path / "d1.jsonl"
    rc = trials.main(["--robot", "real", "--driver", "fake", "--blocks", str(f), "--skill", "pick_and_place", "--n", "1",
                      "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert trials.REAL_JUDGE_NOTE in text and "(0.300, -0.080) yaw  +20.0 deg -> slot_" in text, text
    assert "s arm" in text and "1/1 succeeded" in text
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    notes = {ln["kind"]: ln for ln in lines if ln["type"] == "note"}
    assert notes["verdict"]["verdict"]["ok"] and notes["preflight"]["teleop_off"] and "fake" in notes["preflight"]["operator"]
    assert notes["verdict"]["trial"]["blocks"][0]["xy"] == [0.30, -0.08]
    assert REAL_PARAMS.settle_s > 1.1                                           # outlasts RealArm's force-close window


def test_conformance_real_without_blocks_or_ack_is_refused(capsys):
    with pytest.raises(SystemExit):
        trials.main(["--robot", "real", "--n", "1"])
    assert "--blocks" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        trials.main(["--robot", "real", "--blocks", "x.json", "--n", "1"])
    assert "--teleop-off-ack" in capsys.readouterr().err
