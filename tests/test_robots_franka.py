"""The Franka server (robots/franka) built and tested against the fake driver, before any arm. The
policies are the sim's; the driver is `FakeFrankaDriver` under `FrankaArmIO`. The `server` fixture
runs the server and the fake on one lockstep clock that only the test advances (robots/widowx_real/
clock.py): `run_until`/`run_for` tick the policies and step virtual time, so every outcome is the
same however loaded the machine is. The conformance and brain tests run the threaded wall-clock
server, as the arm will."""
from __future__ import annotations

import asyncio
import json
import math

import pytest

from robojev import catalog
from robojev import conformance as trials
from robojev.brain import Brain, BrainConfig
from robojev.brain.eval.oracle import Goal
from robojev.brain.mocks import oracle_models
from robojev.protocol import RobotServer
from robojev.recording.recorder import JsonlRecorder
from robojev.robots.franka import FRANKA_LIMITS, FakeFrankaDriver, FrankaDriver, FrankaRobotServer, PreflightError
from robojev.robots.franka.arm import ARM_ID, FRANKA_PARAMS
from robojev.robots.franka.fake_driver import GRIP_RATE, READY_XYZ, REFLEX_TEXT
from robojev.robots.widowx_real.clock import LockstepClock
from robojev.robots.widowx_real.world import StaticWorld
from robojev.robots.widowx_sim import scene, skills

TABLE_Z = -0.02
LIMITS = FRANKA_LIMITS.for_table(TABLE_Z)
TRAVEL_Z = LIMITS.travel_z
BLOCK = scene.Block("block_1", (0.30, -0.08), yaw=0.5)
BLOCK_2 = scene.Block("block_2", (0.32, 0.10), yaw=-0.2)
ARGS = {"object": "block_1", "place": "slot_2"}
ACK = dict(fci_active=True, stop_in_hand=True, workspace_clear=True, rt_kernel=True)


@pytest.fixture
def server():
    made = []

    def make(blocks=(BLOCK,), recorder=None, tray=scene.Tray(), fault_after_s=None, **kw) -> FrankaRobotServer:
        clock = LockstepClock()
        world = StaticWorld(list(blocks), tray, table_z=TABLE_Z)
        drivers = []

        def factory():
            drivers.append(FakeFrankaDriver(lambda: srv.world_src, clock=clock.now, fault_after_s=fault_after_s))
            return drivers[-1]

        srv = FrankaRobotServer(factory, world, recorder=recorder, limits=LIMITS, clock=clock, **kw)
        srv.events = []
        srv.subscribe(srv.events.append)
        srv.preflight("pytest", **ACK)
        made.append(srv)
        srv.open()
        srv.fake = drivers[0]
        return srv

    yield make
    for s in made:
        s.close()


def phase(srv, sid) -> str:
    return srv.status(sid).phase


def in_phase(srv, sid, name):
    return lambda: phase(srv, sid) == name or srv.status(sid).state in ("done", "failed")


def watch_run(srv, sid, timeout=25) -> tuple[list[str], list[tuple[str, str | None]]]:
    """Run the skill to its end; return the phases seen and the (block where, holding) pairs seen, deduplicated."""
    phases, seen = [], []

    def watch():
        p = phase(srv, sid)
        if not phases or phases[-1] != p:
            phases.append(p)
        w = srv.world()
        pair = (w.objects["block_1"].where, w.arms[ARM_ID].holding)
        if not seen or seen[-1] != pair:
            seen.append(pair)
        return srv.status(sid).state in ("done", "failed")
    assert srv.run_until(watch, timeout)
    return phases, seen


# ---------------------------------------------------------------- the fake driver


def test_fake_driver_tracks_a_target_grips_a_block_and_faults_on_demand():
    clock = [0.0]
    w = StaticWorld([BLOCK], table_z=TABLE_Z)
    d = FakeFrankaDriver(w, clock=lambda: clock[0], fault_after_s=10.0)
    assert isinstance(d, FrankaDriver)
    s = d.read()
    assert s.xyz == READY_XYZ and s.width == 0.08 and not s.moving and s.error is None
    d.set_target((0.30, -0.08, TRAVEL_Z), math.pi / 2, 0.2)
    clock[0] += 1.0
    s = d.read()
    assert s.moving and math.dist(READY_XYZ, s.xyz) == pytest.approx(0.2)             # 0.2 m/s, in a straight line
    clock[0] += 5.0
    s = d.read()
    assert not s.moving and s.xyz == (0.30, -0.08, TRAVEL_Z)
    d.set_target((0.30, -0.08, TABLE_Z + 0.005), math.pi / 2, 0.1)
    clock[0] += 2.0
    d.set_gripper(0.032, force=20.0)                                                  # the 4 cm block minus the squeeze
    clock[0] += 0.2
    s = d.read()
    assert s.width == pytest.approx(0.08 - 0.2 * GRIP_RATE) and s.gripper_moving and not s.grasped
    clock[0] += 1.0
    s = d.read()
    assert s.width == pytest.approx(0.04) and s.grasped and not s.gripper_moving      # stopped on the block
    d.set_gripper(0.032, force=20.0)                                                  # the policy repeats it every tick
    assert d.read().grasped
    d.stop()
    assert d.read().width == pytest.approx(0.04) and d.read().grasped                 # the grip survives a stop
    clock[0] += 1.0
    s = d.read()
    assert s.error == REFLEX_TEXT and not s.moving                                    # 10 s after it was built
    d.set_target((0.40, 0.0, TRAVEL_Z), math.pi / 2, 0.2)
    clock[0] += 1.0
    assert d.read().xyz == (0.30, -0.08, TABLE_Z + 0.005)                             # refused: in error
    d.recover()
    assert d.read().error is None
    d.set_target((0.40, 0.0, TRAVEL_Z), math.pi / 2, 0.2)
    clock[0] += 0.5
    assert d.read().moving
    d.park()
    assert d.read().xyz == READY_XYZ and d.read().grasped and d.calls == [("stop",), ("recover",), ("park",)]
    d2 = FakeFrankaDriver(None, clock=lambda: clock[0])
    d2.set_gripper(0.0)
    clock[0] += 2.0
    assert d2.read().width == 0.0 and not d2.read().grasped                           # nothing between the fingers


# ---------------------------------------------------------------- the server


def test_manifest_conforms_and_the_arm_starts_idle_at_the_hover_start(server):
    srv = server()
    assert isinstance(srv, RobotServer)
    man = srv.manifest()
    assert man.robot == "franka-1" and man.catalog == catalog.CATALOG_VERSION
    assert catalog.check_manifest_skills(man.skills) == []
    assert {"pick_and_place", "stack_on", "hand_over", "push"} <= {s.name for s in man.skills}
    arm = man.arms[0]
    assert arm.id == ARM_ID and arm.base_frame == "fr3_link0" and arm.gripper.max_width == pytest.approx(0.08)
    assert arm.workspace == ((0.25, 0.75), (-0.40, 0.40), (TABLE_Z + 0.004, TABLE_Z + 0.30))
    assert arm.travel_z == pytest.approx(TRAVEL_Z)
    w = srv.world()
    a = w.arms[ARM_ID]
    assert w.objects["block_1"].where == "table" and a.mode == "idle" and a.holding is None
    assert (a.x, a.y) == LIMITS.hover_xy and a.z == pytest.approx(TRAVEL_Z) and a.gripper_width == pytest.approx(0.08)
    assert set(w.places) == {f"slot_{i}" for i in range(5)} | {"tray"}
    io = srv.arms[ARM_ID]
    assert io.reachable((0.70, 0.35, TRAVEL_Z)) and not io.reachable((0.20, 0.0, TRAVEL_Z)) and not io.reachable((0.40, 0.0, TABLE_Z))
    assert srv.precondition(ARM_ID, "pick_and_place", ARGS) is None
    assert "unreachable" in srv.precondition(ARM_ID, "push", {"object": "block_1", "direction": "toward_robot", "distance": 0.3})


def test_pick_and_place_end_to_end_with_holding_and_where_changing(server, tmp_path):
    rec = JsonlRecorder(tmp_path / "franka.jsonl")
    srv = server(recorder=rec)
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    phases, seen = watch_run(srv, sid)
    assert phases == list(skills.PickAndPlace.PHASES) + ["done"]
    assert seen == [("table", None), (f"gripper:{ARM_ID}", "block_1"), ("slot_2", None)]
    st = srv.status(sid)
    assert st.state == "done" and st.phase_text == "block_1 placed at slot_2"
    assert [e.kind for e in srv.events] == ["skill_done"] and srv.events[0].skill_id == sid
    w = srv.world()
    assert w.objects["block_1"].where == "slot_2" and w.arms[ARM_ID].holding is None and w.arms[ARM_ID].mode == "idle"
    assert abs(w.arms[ARM_ID].z - TRAVEL_Z) < 0.01
    slot = w.places["slot_2"]
    assert math.hypot(w.objects["block_1"].x - slot.x, w.objects["block_1"].y - slot.y) < 0.005
    ok, why = trials.judge(w, srv.manifest(), "block_1", "slot_2")
    assert ok, why
    assert srv.fake.targets_sent > 100 and srv.fake.grip_force is None             # the last gripper command was the release
    rec.close()
    lines = [json.loads(line) for line in (tmp_path / "franka.jsonl").read_text().splitlines()]
    kinds = [ln.get("kind") for ln in lines if ln["type"] == "note"]
    assert kinds[0] == "preflight" and "skill_start" in kinds
    frames = [ln for ln in lines if ln["type"] == "frame"]
    assert len(frames) > 200 and frames[-1]["phase"] == "done" and frames[-2]["phase"] == "retreat"
    assert {"grasped", "error", "moving"} <= set(frames[0]["observation"]) and "yaw" in frames[0]["action"]
    assert any(f["observation"]["holding"] == "block_1" for f in frames)


def test_stack_on_and_push_run_on_the_fake(server):
    srv = server(blocks=(BLOCK, BLOCK_2))
    sid = srv.start(ARM_ID, "stack_on", {"object": "block_1", "onto": "block_2"})
    assert srv.run_until(srv.finished(sid), 25) and srv.status(sid).state == "done", srv.status(sid)
    w = srv.world()
    assert w.objects["block_1"].where == "on:block_2" and w.arms[ARM_ID].holding is None
    ok, why = trials.judge_stack(w, srv.manifest(), "block_1", "block_2")
    assert ok, why
    srv2 = server(blocks=(BLOCK_2,))
    sid = srv2.start(ARM_ID, "push", {"object": "block_2", "direction": "left", "distance": 0.06})
    assert srv2.run_until(srv2.finished(sid), 25) and srv2.status(sid).state == "done", srv2.status(sid)
    ok, why = trials.judge_push(srv2.world(), srv2.manifest(), "block_2", BLOCK_2.xy, "left", 0.06, table_z=TABLE_Z)
    assert ok, why


def test_hold_pause_and_resume_mid_carry_keep_the_grip(server):
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "carry"), 15) and phase(srv, sid) == "carry"
    srv.hold(ARM_ID)
    srv.run_for(0.1)
    ee0, w0 = srv.arms[ARM_ID].ee(), srv.arms[ARM_ID].width()
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("held: carrying block_1")
    assert srv.fake.target is None and srv.fake.calls[-1] == ("stop",)              # the motion was stopped, not re-aimed
    srv.run_for(0.5)
    assert srv.arms[ARM_ID].ee() == ee0 and srv.arms[ARM_ID].width() == pytest.approx(w0)
    w = srv.world()
    assert w.arms[ARM_ID].mode == "holding" and w.arms[ARM_ID].holding == "block_1" and w.objects["block_1"].where == f"gripper:{ARM_ID}"
    srv.pause(ARM_ID)
    srv.run_for(0.3)
    assert srv.status(sid).state == "paused" and srv.world().arms[ARM_ID].mode == "paused" and srv.arms[ARM_ID].ee() == ee0
    assert srv.world().arms[ARM_ID].holding == "block_1"
    srv.resume(ARM_ID)
    assert srv.run_until(lambda: srv.arms[ARM_ID].ee() != ee0, 1.0)                 # moving again
    assert srv.status(sid).state == "running" and srv.status(sid).phase == "carry"
    assert srv.run_until(srv.finished(sid), 15)
    assert srv.status(sid).state == "done" and srv.world().objects["block_1"].where == "slot_2"
    assert [e.kind for e in srv.events] == ["skill_done"]
    assert srv.fake.calls.count(("recover",)) == 0                                    # nothing to recover from


def test_stop_fails_the_skill_parks_and_keeps_the_block(server):
    srv = server()
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(in_phase(srv, sid, "carry"), 15) and srv.world().arms[ARM_ID].holding == "block_1"
    srv.stop()
    st = srv.status(sid)
    assert st.state == "failed" and st.reason.startswith("stopped:")
    assert [e.kind for e in srv.events] == ["skill_failed"] and srv.events[0].skill_id == sid
    with pytest.raises(ValueError, match="stopped"):
        srv.start(ARM_ID, "pick_and_place", ARGS)
    srv.hold(ARM_ID)
    srv.resume(ARM_ID)                                                                # no-ops, no errors
    srv.stop()
    srv._parking.join(5.0)
    assert srv.fake.calls[-2:] == [("stop",), ("park",)] and srv.fake.parked
    w = srv.world()
    assert w.arms[ARM_ID].mode == "stopped" and (w.arms[ARM_ID].x, w.arms[ARM_ID].y, w.arms[ARM_ID].z) == READY_XYZ
    assert w.arms[ARM_ID].holding == "block_1" and w.arms[ARM_ID].gripper_width == pytest.approx(0.04)
    assert w.objects["block_1"].where == f"gripper:{ARM_ID}"
    assert [e.kind for e in srv.events] == ["skill_failed"]                           # stop() twice: once


def test_collision_reflex_becomes_a_safety_trip_and_resume_recovers(server):
    srv = server(fault_after_s=4.0)                                                   # the arm is about 2 s into the skill
    sid = srv.start(ARM_ID, "pick_and_place", ARGS)
    assert srv.run_until(lambda: any(e.kind == "safety_trip" for e in srv.events), 5.0)
    trip = next(e for e in srv.events if e.kind == "safety_trip")
    assert trip.skill_id == sid and trip.arm == ARM_ID and "cartesian_reflex" in trip.text and trip.data["error"] == REFLEX_TEXT
    st = srv.status(sid)
    assert st.state == "holding" and st.phase_text.startswith("safety trip (the arm reported cartesian_reflex")
    assert srv.world().arms[ARM_ID].mode == "tripped"
    ee0 = srv.arms[ARM_ID].ee()
    srv.run_for(0.5)
    assert srv.arms[ARM_ID].ee() == ee0 and srv.status(sid).state == "holding"
    assert [e.kind for e in srv.events].count("safety_trip") == 1                     # one trip, not one per tick
    assert ("recover",) not in srv.fake.calls and srv.fake.read().error == REFLEX_TEXT  # nothing cleared it on its own
    srv.resume(ARM_ID)
    assert srv.fake.calls.count(("recover",)) == 1 and srv.fake.error is None
    assert srv.status(sid).state == "running" and srv.world().arms[ARM_ID].mode == "running"
    assert srv.run_until(lambda: srv.arms[ARM_ID].ee() != ee0, 1.0)
    assert srv.run_until(srv.finished(sid), 25) and srv.status(sid).state == "done"
    assert srv.world().objects["block_1"].where == "slot_2"
    assert [e.kind for e in srv.events] == ["safety_trip", "skill_done"]


def test_reflex_on_an_idle_arm_freezes_it_until_resume(server):
    srv = server()
    srv.fake.inject_error()
    assert srv.run_until(lambda: any(e.kind == "safety_trip" for e in srv.events), 1.0)
    assert srv.world().arms[ARM_ID].mode == "tripped" and srv.events[0].skill_id is None
    srv.resume(ARM_ID)
    assert srv.fake.error is None and srv.world().arms[ARM_ID].mode == "idle"
    sid = srv.start(ARM_ID, "survey", {})
    assert srv.run_until(srv.finished(sid), 5) and srv.status(sid).state == "done"


def test_preflight_gates_open():
    world = StaticWorld([BLOCK], table_z=TABLE_Z)
    srv = FrankaRobotServer(lambda: FakeFrankaDriver(world), world, limits=LIMITS)
    with pytest.raises(PreflightError, match="preflight"):
        srv.open()
    for missing, word in (("fci_active", "FCI"), ("stop_in_hand", "stop button"), ("workspace_clear", "workspace"),
                          ("rt_kernel", "RT kernel")):
        srv.preflight("someone", **{**ACK, missing: False})
        with pytest.raises(PreflightError, match=word):
            srv.open()
    srv.preflight("", **ACK)
    with pytest.raises(PreflightError, match="operator"):
        srv.open()
    rec = srv.preflight("someone", **ACK, notes="dry run")
    assert rec.problems() == [] and srv.preflight_record is rec and rec.notes == "dry run"
    with srv:
        assert srv.world().arms[ARM_ID].mode == "idle"


# ---------------------------------------------------------------- the wall-clock server: conformance and the brain


def test_conformance_runs_on_the_fake_franka(tmp_path, capsys):
    f = tmp_path / "blocks.json"
    f.write_text(json.dumps({"table_z": TABLE_Z, "blocks": [{"name": "block_1", "xy": [0.30, -0.08], "yaw_deg": 20}]}))
    out = tmp_path / "franka.jsonl"
    rc = trials.main(["--robot", "franka", "--driver", "fake", "--blocks", str(f), "--skill", "pick_and_place", "--n", "1",
                      "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert trials.REAL_JUDGE_NOTE in text and "ok    0  block (0.300, -0.080) yaw  +20.0 deg -> slot_" in text, text
    assert "1/1 succeeded" in text
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    notes = {ln["kind"]: ln for ln in lines if ln["type"] == "note"}
    assert notes["verdict"]["verdict"]["ok"] and notes["preflight"]["fci_active"] and "fake" in notes["preflight"]["operator"]
    assert "--robot franka" in notes["preflight"]["notes"]
    assert FRANKA_PARAMS.settle_s >= 0.3


def test_conformance_franka_refusals_name_what_is_missing(tmp_path, capsys, monkeypatch):
    f = tmp_path / "blocks.json"
    f.write_text(json.dumps({"table_z": TABLE_Z, "blocks": [{"name": "block_1", "xy": [0.30, -0.08]}]}))
    with pytest.raises(SystemExit):
        trials.main(["--robot", "franka", "--n", "1"])
    assert "--blocks" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        trials.main(["--robot", "franka", "--blocks", str(f), "--n", "1"])          # the real driver is the default
    assert "--preflight-ack" in capsys.readouterr().err
    # the real driver exists; with no arm its connect fails, and that must read as one line naming the address
    from robojev.robots.franka import franky_driver

    def no_arm(self, ip, **kw):
        raise ConnectionError(f"no route to {ip}")
    monkeypatch.setattr(franky_driver.FrankyDriver, "__init__", no_arm)
    with pytest.raises(SystemExit):
        trials.main(["--robot", "franka", "--driver", "franky", "--blocks", str(f), "--preflight-ack", "me", "--n", "1", "--robot-ip", "10.0.0.9"])
    err = capsys.readouterr().err
    assert "could not connect to the Franka at 10.0.0.9" in err and "no route to 10.0.0.9" in err and "FCI" in err
    with pytest.raises(SystemExit):
        trials.main(["--robot", "real", "--driver", "franky", "--blocks", str(f), "--n", "1"])
    assert "--robot franka" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        trials.main(["--robot", "franka", "--driver", "trossen", "--blocks", str(f), "--n", "1"])
    assert "--robot real" in capsys.readouterr().err


def test_brain_sorts_two_blocks_on_the_fake_franka():
    world = StaticWorld([scene.Block("block_1", (0.30, 0.08)), scene.Block("block_2", (0.30, -0.06))], scene.Tray(n=2), table_z=TABLE_Z)
    srv = FrankaRobotServer(lambda: FakeFrankaDriver(lambda: srv.world_src), world, limits=LIMITS)
    srv.preflight("pytest", **ACK)
    with srv:
        decider, llm = oracle_models(srv, [Goal({"block_1": "slot_0", "block_2": "slot_1"})])
        r = asyncio.run(Brain(srv, decider, llm, None, BrainConfig(max_s=60)).run(
            "Put block 1 in the leftmost tray slot and block 2 in the next one."))
        assert r.outcome == "done", r.log
        where = {o.id: o.where for o in srv.world().objects.values()}
        assert where == {"block_1": "slot_0", "block_2": "slot_1"}
        assert r.steps_done == 2 and r.steps_failed == 0 and r.events["safety_trip"] == 0
        assert srv.world().arms[ARM_ID].mode == "idle" and srv.world().arms[ARM_ID].holding is None
