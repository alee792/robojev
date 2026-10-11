"""L1 (docs/harness-spikes.md): the thin end-to-end slice. The brain (robojev.brain), a client of
the robot protocol on a real asyncio clock, drives the MuJoCo WidowX server
(robojev.robots.widowx_sim.server, realtime=True) through physics `pick_and_place`, with e12v2's
mock decider and mock planner (no API keys). The e12v2 core scenarios, judged from the server's
world() and never from what the brain or the skill say about themselves:

  (a) a 3-block numbered sort, lowest on the left;
  (b) the same with "actually, highest on the left" typed while a block is carried;
  (c) a hand near the gripper mid-carry: pause within 0.5 s, resume when it leaves, no contact;
  (d) STOP mid-carry: the arm parks, the block is held or set down, never dropped from height.

The sim has no person: its world() reports no hands. The hand in (c) is a perception override in
front of the server (mocks.ScriptedPerception, labelled as such there), not a body in the physics,
so "no contact" is the closest the gripper came to the reported hand position.

Measured numbers are printed per test (pytest -s) and recorded in experiments/results/l1_slice.txt.
pytest-asyncio is not installed: each episode runs under asyncio.run in a plain test.
"""
from __future__ import annotations

import asyncio
import math
import time

import pytest

from robojev import conformance as trials
from robojev.brain import Brain, BrainConfig
from robojev.brain.eval.oracle import Goal
from robojev.brain.eval.scenarios import Utt
from robojev.brain.mocks import PRESS_STOP, Line, ScriptedPerception, ScriptedUser, oracle_models
from robojev.robots.widowx_sim import scene

pytestmark = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(),
                                reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")
pytest.importorskip("mujoco")
from robojev.robots.widowx_sim.server import ARM_ID, HOME_XY, TRAVEL_Z, SimRobotServer  # noqa: E402

TASK = "Line the blocks up in the tray, lowest number on the left."
CORRECTION = "actually, highest on the left"
REACT_S = 0.5                      # the bar: text or hand -> the server reports holding/paused
REST_Z = scene.TABLE_Z + 0.02      # a 4 cm block's centre when it sits on the table
FALL_M = 0.02                      # higher than this, unheld, for longer than the grace = dropped from height
GRACE_S = 0.2                      # a flicker of the physics grasp detector is not a drop (skills.Params.drop_grace)

# blocks 5, 1, 3 left to right as the robot sees them (+y is left); three slots at x 0.36
BLOCKS = [scene.Block("block_5", (0.26, 0.12)), scene.Block("block_1", (0.26, 0.0)), scene.Block("block_3", (0.26, -0.12))]
ASCENDING = Goal({"block_1": "slot_0", "block_3": "slot_1", "block_5": "slot_2"})
DESCENDING = Goal({"block_5": "slot_0", "block_3": "slot_1", "block_1": "slot_2"})


# ---------------------------------------------------------------- helpers


@pytest.fixture
def srv():
    """The physics server on the wall clock, behind the logging perception layer."""
    with SimRobotServer(scene.SceneSpec(blocks=list(BLOCKS), tray=scene.Tray(n=3)), realtime=True) as sim:
        with ScriptedPerception(sim) as p:
            p.events = []
            p.subscribe(p.events.append)
            yield p


def brain(srv, versions, utts=None, user=None, **cfg) -> Brain:
    decider, llm = oracle_models(srv, versions, utts)
    return Brain(srv, decider, llm, user, BrainConfig(max_s=90, **cfg))


def episode(b: Brain):
    t = time.monotonic()
    r = asyncio.run(b.run(TASK))
    return r, time.monotonic() - t


def carrying(srv, oid=None, after_s: float = 0.3):
    """A condition: the skill on the arm has been in its carry phase, `oid` (or anything) in the
    gripper, for `after_s` (mid-carry: a carry here is 10-16 cm, about a second)."""
    seen: list[float] = []

    def cond(*_) -> bool:
        a = srv.world().arms[ARM_ID]
        if a.holding is None or (oid is not None and a.holding != oid) or a.skill is None \
                or srv.status(a.skill).phase != "carry":
            seen.clear()
            return False
        if not seen:
            seen.append(time.monotonic())
        return time.monotonic() - seen[0] >= after_s
    return cond


def judge_sort(srv, goal: Goal) -> tuple[bool, str]:
    """Every block resting in its slot, gripper empty, arm back at travel height: from world() only."""
    w, m = srv.world(), srv.manifest()
    verdicts = [trials.judge(w, m, oid, place) for oid, place in goal.assign.items()]
    return all(ok for ok, _ in verdicts), "; ".join(why for _, why in verdicts)


def tools(srv, *names):
    return [c for c in srv.calls if c[1] in names]


def first_mode(srv, modes, after: float) -> float | None:
    """Wall time the server first reported one of `modes` at or after `after`."""
    return next((t for t, mode, _, _ in srv.history if t >= after and mode in modes), None)


def drops(srv, oid: str, since: float = 0.0) -> list[tuple[float, float]]:
    """Spells in the history where `oid` was neither in the gripper nor near the table for longer than
    the grasp detector's grace: (when, how long). Empty = never dropped from height."""
    out, start = [], None
    for i, (t, _, holding, objs) in enumerate(srv.history):
        if t < since:
            continue
        high = holding != oid and objs[oid][1] > REST_Z + FALL_M
        if high and start is None:
            start = t
        elif not high and start is not None:
            if t - start > GRACE_S:
                out.append((start, t - start))
            start = None
    if start is not None and srv.history[-1][0] - start > GRACE_S:
        out.append((start, srv.history[-1][0] - start))
    return out


def summary(name: str, r, wall: float, **extra) -> None:
    bits = [f"wall {wall:.1f} s", f"events {dict(r.events)}", f"steps started/done/failed {r.steps_started}/{r.steps_done}/{r.steps_failed}",
            f"plans {len(r.llm)} (stale {r.stale_plans}, failed {r.failed_plans}, loops {r.loops})"]
    bits += [f"{k} {v}" for k, v in extra.items()]
    print(f"\nL1 {name}: " + "; ".join(bits))


# ---------------------------------------------------------------- (a) the sort


def test_a_three_block_sort_on_physics(srv):
    r, wall = episode(brain(srv, [ASCENDING]))
    ok, why = judge_sort(srv, ASCENDING)
    summary("sort", r, wall, verdict=why)
    assert r.outcome == "done", r.log
    assert ok, why
    assert r.steps_done == 3 and r.steps_failed == 0 and r.failed_plans == 0
    assert [c[2][1] for c in tools(srv, "start")] == ["pick_and_place"] * 3
    assert r.events["scene_change"] == 0                      # placing a block never read as someone else's doing
    assert not tools(srv, "stop", "pause")
    assert wall < 60.0


# ---------------------------------------------------------------- (b) a correction mid-carry


def test_b_correction_mid_carry_holds_then_reverses_the_order(srv):
    user = ScriptedUser([Line(carrying(srv, "block_1"), CORRECTION)])
    r, wall = episode(brain(srv, [ASCENDING, DESCENDING], {CORRECTION: Utt("correction", version=1)}, user))
    t_text = user.t0 + user.said[0][0]
    t_hold = first_mode(srv, ("holding", "paused"), t_text)
    ok, why = judge_sort(srv, DESCENDING)
    dropped = drops(srv, "block_1")
    summary("correction", r, wall, text_to_hold_s=None if t_hold is None else round(t_hold - t_text, 3), verdict=why,
            starts=[c[2][2] for c in tools(srv, "start")], drops=dropped)
    assert r.outcome == "done", r.log
    assert t_hold is not None and t_hold - t_text < REACT_S, (t_hold, t_text)
    said = next(d for d in r.decisions if d["event"] == "user_text")
    assert said["combined"].right_now == "hold"
    assert ok, why
    # the held block was carried on to its new slot: the new skill began with it in the gripper
    starts = [c[2][2] for c in tools(srv, "start")]
    assert starts[:2] == [{"object": "block_1", "place": "slot_0"}, {"object": "block_1", "place": "slot_2"}]
    assert [e.kind for e in srv.events if e.kind == "skill_failed"] == []
    assert dropped == [] and r.steps_failed == 0
    assert not tools(srv, "stop")


# ---------------------------------------------------------------- (c) a hand near the gripper


def test_c_hand_near_the_gripper_pauses_within_half_a_second_and_resumes(srv):
    srv.hand_when(carrying(srv), for_s=1.5, dist=0.15)
    r, wall = episode(brain(srv, [ASCENDING]))
    appeared = next(t for t, what in srv.hand_log if what == "appears")
    left = next(t for t, what in srv.hand_log if what == "leaves")
    t_pause = first_mode(srv, ("paused",), appeared)
    resumed = [t for t, _, _ in tools(srv, "resume")]
    ok, why = judge_sort(srv, ASCENDING)
    summary("hand", r, wall, hand_to_pause_s=None if t_pause is None else round(t_pause - appeared, 3),
            leave_to_resume_s=None if not resumed else round(resumed[0] - left, 3), min_hand_dist_m=round(srv.min_hand_dist, 3),
            verdict=why)
    assert r.outcome == "done", r.log
    assert t_pause is not None and t_pause - appeared < REACT_S, (t_pause, appeared)
    assert t_pause < left
    assert resumed and resumed[0] >= left                      # resumed only once the hand had gone
    # paused means paused: the server reported nothing but "paused" between the pause and the resume
    between = {mode for t, mode, _, _ in srv.history if t_pause < t < resumed[0]}
    assert between <= {"paused"}, between
    assert srv.min_hand_dist > 0.05                            # the gripper never reached the hand
    assert ok, why
    assert drops(srv, "block_1") == [] and r.steps_failed == 0
    assert not tools(srv, "stop")


# ---------------------------------------------------------------- (d) STOP mid-carry


def test_d_stop_mid_carry_parks_the_arm_and_never_drops_the_block(srv):
    user = ScriptedUser([Line(carrying(srv, "block_1"), PRESS_STOP)])
    r, wall = episode(brain(srv, [ASCENDING], user=user))
    t_stop = tools(srv, "stop")[0][0]
    assert r.outcome == "stopped"
    assert not [c for c in srv.calls if c[0] > t_stop and c[1] != "stop"]       # nothing sent after STOP
    # the server parks at crawl speed: wait for it (6 s is 18 cm at 0.03 m/s)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        a = srv.world().arms[ARM_ID]
        if math.hypot(a.x - HOME_XY[0], a.y - HOME_XY[1]) < 0.01 and abs(a.z - TRAVEL_Z) < 0.01:
            break
        time.sleep(0.05)
    w = srv.world()
    a, b = w.arms[ARM_ID], w.objects["block_1"]
    parked_after = time.monotonic() - t_stop
    dropped = drops(srv, "block_1", since=t_stop)
    summary("stop", r, wall, stop_to_parked_s=round(parked_after, 2), arm_mode=a.mode, holding=a.holding,
            block_where=b.where, block_z=round(b.z, 3), drops=dropped)
    assert a.mode == "stopped"
    assert math.hypot(a.x - HOME_XY[0], a.y - HOME_XY[1]) < 0.01, (a.x, a.y)
    assert a.holding == "block_1" or (b.where in ("table",) and abs(b.z - REST_Z) < 0.005)   # held, or set down
    assert dropped == [], dropped
    failed = [e for e in srv.events if e.kind == "skill_failed"]
    assert len(failed) == 1 and failed[0].text.startswith("stopped: ")
    assert r.events["user_text"] == 0                          # STOP is code: no decision is asked about it
