"""Catalog 0.2 in the brain: the two tool skills (insert, reorient) planned, validated and run on the
stub, and the fast in-plan fix `swap_target` ("actually, the scalpel" mid-reach: the arm goes for
the other object within the decision's own latency, no planner call).

Episodes run against the physics-free stub with e12v2's oracle mocks, under asyncio.run.
"""
from __future__ import annotations

import asyncio
import math
import time

import pytest

from robojev import catalog
from robojev.brain import Brain, BrainConfig
from robojev.brain.adapt import core_state, front_of, geometry
from robojev.brain.core import plan as P
from robojev.brain.core.data import Event
from robojev.brain.core.decision import View, build_questions, build_state, object_fact, swap_candidates
from robojev.brain.eval.oracle import Goal, oracle_plan
from robojev.brain.eval.scenarios import Utt
from robojev.brain.mocks import Line, ScriptedUser, oracle_models
from robojev.brain.steps import call_args, to_catalog, to_core
from robojev.robots import stub as S

TOOLS_TASK = "Put the scalpel and the forceps in the rack and turn the scissors to face me."
ONE_BLOCK = "Put block 1 in tray slot 1."


# ---------------------------------------------------------------- helpers


@pytest.fixture
def servers():
    made = []

    def make(scene="sort", numbers=(5, 1, 3), pick_s=0.4, **kw) -> S.StubRobotServer:
        objs, places = S.tools_scene() if scene == "tools" else S.sort_scene(numbers)
        srv = S.StubRobotServer(S.widowx_like(), objs, places,
                                durations={"pick_and_place": pick_s, "insert": pick_s, "reorient": pick_s}, **kw)
        made.append(srv)
        return srv.open()
    yield make
    for s in made:
        s.close()


def where(srv) -> dict:
    return {o.id: o.where for o in srv.world().objects.values()}


def starts(srv) -> list[tuple]:
    return [(c[0], c[2][1], c[2][2]) for c in srv.calls if c[1] == "start"]


def episode(brain, task):
    return asyncio.run(brain.run(task))


def core(srv):
    return core_state(srv.world(), "arm_0", geometry(srv.manifest().arms[0], 0.2), 0.2)


class Recording:
    """A decider in front of another: keeps every (event, request state, questions, combined) it saw."""

    def __init__(self, inner):
        self.inner, self.name, self.seen = inner, "recording", []

    def decide(self, ev, view):
        out = self.inner.decide(ev, view)
        self.seen.append((ev, out["request"]["state"], out["request"]["questions"], out["combined"]))
        return out

    def text_event(self, text):
        return next(x for x in self.seen if x[0].kind == "user_text" and x[0].data.get("text") == text)


def reaching_for(srv, oid):
    """A condition: the arm's running skill is reaching for `oid`, nothing grasped yet."""
    def holds(*_):
        w = srv.world()
        a = w.arms["arm_0"]
        if a.holding is not None or a.skill is None:
            return False
        st = srv.status(a.skill)
        return st.args.get("object") == oid and st.phase == "reach"
    return holds


# ---------------------------------------------------------------- the vocabulary and the validator


def test_insert_and_reorient_round_trip_through_the_step_vocabulary():
    ins = {"id": "s1", "skill": "insert", "object": "scalpel", "target": "holder_1", "direction": "none", "distance": None, "seconds": None}
    assert to_core(ins)["skill"] == "insert" and to_catalog(to_core(ins)) == ins
    assert call_args(ins, catalog.INSERT) == {"object": "scalpel", "place": "holder_1"}
    turn = {"id": "s2", "skill": "reorient", "object": "scissors", "target": "none", "direction": "toward_robot", "distance": None, "seconds": None}
    assert call_args(turn, catalog.REORIENT) == {"object": "scissors", "direction": "toward_robot"}
    assert P.step_goal(ins) == ("scalpel", "holder_1") and P.step_goal(turn) is None        # a reorient leaves placement alone
    assert P.step_text(ins) == "insert(scalpel → holder_1)" and P.step_text(turn) == "reorient(scissors to face toward_robot)"


def _tools_plan(steps, done=None):
    return {"reading": TOOLS_TASK, "steps": steps, "constraints": [],
            "done_when": done or [{"id": "d1", "text": "the scalpel is in holder_1", "object": "scalpel", "relation": "in", "target": "holder_1"}]}


def _step(i, skill, obj, target="none", direction="none"):
    return {"id": f"s{i}", "skill": skill, "object": obj, "target": target, "direction": direction}


def test_validator_knows_what_insert_and_reorient_do(servers):
    srv = servers("tools")
    st = core(srv)
    assert {o.shape for o in st.objects.values()} == {"rod"}
    assert {o.id: o.front for o in st.objects.values()} == {"scalpel": "away_from_robot", "forceps": "left", "scissors": "right"}
    assert st.places["holder_1"].kind == "holder" and st.places["holder_1"].yaw == pytest.approx(math.pi)
    ok = _tools_plan([_step(1, "insert", "scalpel", "holder_1"), _step(2, "reorient", "scissors", direction="toward_robot")],
                     done=[{"id": "d1", "text": "the scalpel is in holder_1", "object": "scalpel", "relation": "in", "target": "holder_1"},
                           {"id": "d2", "text": "the scissors' front points toward the person", "object": "scissors", "relation": "other", "target": "none"}])
    assert P.validate(ok, st) == []
    # insert needs a holder, an empty one, and a rod
    assert "needs a holder" in P.validate(_tools_plan([_step(1, "insert", "scalpel", "table")]), st)[0]
    both = _tools_plan([_step(1, "insert", "scalpel", "holder_1"), _step(2, "insert", "forceps", "holder_1")])
    assert any("holder_1 is still occupied by scalpel" in e for e in P.validate(both, st))
    st.objects["scalpel"].shape = "cube"
    errs = P.validate(_tools_plan([_step(1, "insert", "scalpel", "holder_1")]), st)
    assert errs[0] == "step s1: insert takes a rod, and scalpel is a cube; a holder is a tight fit for a rod only"
    st.objects["scalpel"].shape = None
    assert "of unknown shape" in P.validate(_tools_plan([_step(1, "insert", "scalpel", "holder_1")]), st)[0]
    # reorient needs a direction and an object on the table
    assert "reorient needs a direction" in P.validate(_tools_plan([_step(1, "reorient", "scissors")]), st)[0]
    bad = _tools_plan([_step(1, "hand_over", "scissors", "person"), _step(2, "reorient", "scissors", direction="left")])
    assert any("not on the table to be turned" in e for e in P.validate(bad, st))
    # a held rod may be inserted first (the holding rule names insert)
    st.objects["scalpel"].shape = "rod"
    held = _tools_plan([_step(1, "insert", "scalpel", "holder_1")])
    assert P.validate(held, st, holding="scalpel") == []
    assert "or insert of scalpel" in P.validate(_tools_plan([_step(1, "reorient", "scalpel", direction="left")]), st, holding="scalpel")[0]


def test_front_words_come_from_the_yaw_by_quadrant():
    assert [front_of(y) for y in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.7, -0.7, 2.5, None)] == \
        ["away_from_robot", "left", "toward_robot", "right", "away_from_robot", "away_from_robot", "toward_robot", None]


def test_decision_text_says_what_an_object_is_and_which_way_it_faces(servers):
    srv = servers("tools")
    v = View(t=0, task=TOOLS_TASK, user_messages=[], state=core(srv), plan=None, queue=[], status={}, current=None,
             phase="not started", mode="running", pause_reason=None, replan_pending=False,
             names={o.id: o.name for o in srv.world().objects.values()}, path_to=None, hand=None, runnable_other=None)
    assert object_fact(v, "forceps")["shape"] == "a rod, its front pointing left"
    assert object_fact(v, "scissors")["shape"] == "a rod, its front pointing right"
    sort = S.StubRobotServer(S.widowx_like(), *S.sort_scene())
    v2 = View(t=0, task="", user_messages=[], state=core(sort), plan=None, queue=[], status={}, current=None, phase="", mode="running",
              pause_reason=None, replan_pending=False, names={}, path_to=None, hand=None, runnable_other=None)
    assert "shape" not in object_fact(v2, "block_1")                             # nothing known: nothing said


# ---------------------------------------------------------------- the stub runs them


def test_stub_insert_seats_a_rod_and_refuses_an_occupied_holder(servers):
    srv = servers("tools")
    sid = srv.start("arm_0", "insert", {"object": "scalpel", "place": "holder_1"})
    assert srv.world().arms["arm_0"].skill == sid
    while srv.status(sid).state == "running":
        time.sleep(0.01)
    w = srv.world()
    assert srv.status(sid).state == "done" and "seated in rack holder 1" in srv.status(sid).phase_text
    assert w.objects["scalpel"].where == "holder_1" and w.objects["scalpel"].yaw == pytest.approx(math.pi)
    assert w.arms["arm_0"].skill is None and w.arms["arm_0"].holding is None
    why = srv.precondition("arm_0", "insert", {"object": "forceps", "place": "holder_1"})
    assert why == "precondition: rack holder 1 (the leftmost) is occupied by the scalpel"
    with pytest.raises(ValueError, match="occupied"):
        srv.start("arm_0", "insert", {"object": "forceps", "place": "holder_1"})
    assert srv.precondition("arm_0", "insert", {"object": "forceps", "place": "table"}) == "precondition: table is not a holder"
    srv.move("forceps", 0.21, 0.0)
    srv._objs["forceps"].shape = "cube"                                          # a toy's perception: a cube is refused
    assert "a holder takes a rod" in srv.precondition("arm_0", "insert", {"object": "forceps", "place": "holder_2"})


def test_stub_reorient_turns_the_object_where_it_lies(servers):
    srv = servers("tools")
    before = srv.world().objects["scissors"]
    sid = srv.start("arm_0", "reorient", {"object": "scissors", "direction": "toward_robot"})
    phases = set()
    while srv.status(sid).state == "running":
        phases.add(srv.status(sid).phase)
        time.sleep(0.01)
    after = srv.world().objects["scissors"]
    assert phases >= {"reach"} and "turn" in phases
    assert after.yaw == pytest.approx(math.pi) and (after.x, after.y, after.where) == (before.x, before.y, "table")
    assert srv.world().arms["arm_0"].holding is None                              # never carried off
    assert srv.status(sid).phase_text == "the scissors turned to face toward robot"


# ---------------------------------------------------------------- a plan with both runs end to end


def test_a_plan_with_insert_and_reorient_runs_on_the_stub_and_its_done_conditions_are_checked(servers):
    srv = servers("tools", pick_s=0.4)
    goal = Goal({"scalpel": "holder_1", "forceps": "holder_2"}, fronts={"scissors": "toward_robot"})
    plan = oracle_plan(goal, core(srv))
    assert [(s["skill"], s["object"], s["target"], s["direction"]) for s in plan["steps"]] == [
        ("insert", "scalpel", "holder_1", "none"), ("insert", "forceps", "holder_2", "none"), ("reorient", "scissors", "none", "toward_robot")]
    assert [c["relation"] for c in plan["done_when"]] == ["in", "in", "other"]
    assert P.validate(plan, core(srv)) == []
    decider, llm = oracle_models(srv, [goal])
    rec = Recording(decider)
    r = episode(Brain(srv, rec, llm, None, BrainConfig(max_s=20)), TOOLS_TASK)
    assert r.outcome == "done", r.log
    assert [s[1] for s in starts(srv)] == ["insert", "insert", "reorient"]           # catalog names on the wire
    assert starts(srv)[0][2] == {"object": "scalpel", "place": "holder_1"} and starts(srv)[2][2] == {"object": "scissors", "direction": "toward_robot"}
    w = srv.world()
    assert where(srv) == {"scalpel": "holder_1", "forceps": "holder_2", "scissors": "table"}
    assert w.objects["scissors"].yaw == pytest.approx(math.pi) and w.objects["scalpel"].yaw == pytest.approx(math.pi)
    assert r.steps_done == 3 and r.steps_failed == 0 and r.failed_plans == 0
    # the "other" done condition was put to the decider on the last step, and the measurable ones were measured
    last = [qs for ev, _, qs, _ in rec.seen if ev.kind == "step_done"][-1]
    assert any(k.startswith("done.") and "front points toward robot" in q["instructions"] for k, q in last.items())
    state = [st for ev, st, _, _ in rec.seen if ev.kind == "step_done"][-1]
    assert state["event_object"]["shape"] == "a rod, its front pointing toward robot"
    arrived = next(st for ev, st, _, _ in rec.seen if ev.kind == "plan_arrived")
    assert arrived["plan"]["steps_to_run"] == ["insert(the scalpel → rack holder 1 (the leftmost))",
                                               "insert(the forceps → rack holder 2)", "reorient(the scissors to face toward_robot)"]


# ---------------------------------------------------------------- swap_target


def _one_block_models(srv, swap_line, kind="swap", swap_to="block_3", v1=None):
    versions = [Goal({"block_1": "tray_slot_1"}), v1 or Goal({swap_to: "tray_slot_1"})]
    utt = Utt(kind, version=1 if kind == "swap" else None, swap_to=swap_to if kind == "swap" else None)
    decider, llm = oracle_models(srv, versions, {swap_line: utt})
    return Recording(decider), llm


def test_swap_candidates_name_other_objects_the_line_mentions(servers):
    srv = servers()
    st = core(srv)
    cur = {"id": "s1", "skill": "move_object", "object": "block_1", "target": "tray_slot_1", "direction": "none"}

    def view(**kw):
        base = dict(t=0, task=ONE_BLOCK, user_messages=[], state=st, plan=None, queue=["s1"], status={"s1": "running"}, current=cur,
                    phase="reaching", mode="running", pause_reason=None, replan_pending=False, names={o.id: o.name for o in st.objects.values()},
                    path_to=None, hand=None, runnable_other=None)
        return View(**{**base, **kw})
    assert swap_candidates("actually, block 3", view()) == ["block_3"]
    assert swap_candidates("Block 3, not block 1", view()) == ["block_3"]                 # never the current object
    assert swap_candidates("the one with the 5 on it", view()) == ["block_5"]            # by label
    assert swap_candidates("block 3 or block 5, either", view()) == ["block_3", "block_5"]
    assert swap_candidates("actually, highest on the left", view()) == []
    assert swap_candidates("block 30 please", view()) == []                              # whole words only
    assert swap_candidates("actually, block 3", view(grasped=True)) == []                # already in the gripper
    assert swap_candidates("actually, block 3", view(status={"s1": "pending"})) == []
    assert swap_candidates("actually, block 3", view(current={**cur, "skill": "push"})) == []
    # the fix is offered on the line, with criteria naming both objects
    ev = Event(1, 0, "user_text", 'the user typed: "actually, block 3"', data={"text": "actually, block 3"})
    qs = build_questions(ev, view())
    crit = qs["in_plan_fix"]["criteria"]
    assert list(crit) == ["none", "skip", "swap_target:block_3"]
    assert "block 3 handled instead of block 1" in crit["swap_target:block_3"]["what"] and "the rest of the plan stands" in crit["swap_target:block_3"]["what"]
    assert "in_plan_fix" not in build_questions(Event(2, 0, "user_text", "x", data={"text": "nice day"}), view(status={"s1": "pending"}))
    assert build_state(ev, view())["event"]["user_just_said"] == "actually, block 3"


def test_letter_labels_match_as_written_and_the_other_one_means_a_duplicate():
    from robojev.brain.core.data import ArmObs, Geometry, Obj, Place, WorldState
    geom = Geometry(12, 2, 7, 4.5, 1.5, 8, 20, 1, (0, 60, -40, 40, 0, 30), 5)
    objs = {"letter_A1": Obj("letter_A1", "the letter A", "", None, "", 20, 5, 0, "table", label="A"),
            "letter_A2": Obj("letter_A2", "the second letter A", "", None, "", 20, -5, 0, "table", label="A"),
            "letter_P1": Obj("letter_P1", "the letter P", "", None, "", 28, 0, 0, "table", label="P"),
            "letter_I1": Obj("letter_I1", "the letter I", "", None, "", 28, 8, 0, "slot_2", label="I")}
    st = WorldState(0, objs, {"slot_0": Place("slot_0", "position 0", "slot", 36, 0, 1)}, None, ArmObs(0, 0, 12, None), geom)
    cur = {"id": "s1", "skill": "move_object", "object": "letter_A1", "target": "slot_0", "direction": "none"}
    v = View(t=0, task="spell PA", user_messages=[], state=st, plan=None, queue=["s1"], status={"s1": "running"}, current=cur, phase="",
             mode="running", pause_reason=None, replan_pending=False, names={}, path_to=None, hand=None, runnable_other=None)
    assert swap_candidates("use the other A", v) == ["letter_A2"]
    assert swap_candidates("take the other one instead", v) == ["letter_A2"]
    assert swap_candidates("I need a moment, mind the P", v) == ["letter_P1"]            # "a" is an article, "P" is the cube
    assert swap_candidates("the letter P instead", v) == ["letter_P1"]


def test_a_correction_naming_another_block_mid_reach_swaps_at_once_with_no_planner_call(servers):
    srv = servers(pick_s=1.2)
    line = "actually, block 3"
    rec, llm = _one_block_models(srv, line)
    user = ScriptedUser([Line(reaching_for(srv, "block_1"), line)])
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=20)), ONE_BLOCK)
    assert r.outcome == "done", r.log
    assert where(srv) == {"block_3": "tray_slot_1", "block_1": "table", "block_5": "table"}   # the original destination
    assert [(s[1], s[2]) for s in starts(srv)] == [("pick_and_place", {"object": "block_1", "place": "tray_slot_1"}),
                                                   ("pick_and_place", {"object": "block_3", "place": "tray_slot_1"})]
    assert [x["kind"] for x in r.llm] == ["plan"]                                          # no planner call for the swap
    fix = next(e for e in r.log if e["note"] == "fix")
    assert fix["fix"] == "swap_target:block_3" and fix["ms"] < 500
    said = next(e for e in r.log if e["note"] == "event" and e["event"] == "user_text")
    assert starts(srv)[1][0] - said["t"] < 0.5                                              # the brain's and the stub's clocks start together
    assert srv.status("sk1").state == "cancelled"                                          # the new start cancelled the reach, no event
    _, _, qs, c = rec.text_event(line)
    assert "swap_target:block_3" in qs["in_plan_fix"]["criteria"] and c.fix == "swap_target:block_3" and c.route == "stay_local"
    assert [e["words"] for e in r.log if e["note"] == "user_words"] == [[ONE_BLOCK, line]]  # the line is about the end result
    after = [st for ev, st, _, _ in rec.seen if ev.t > said["t"] and ev.kind == "step_done"]
    assert after and after[-1]["user_words"] == [ONE_BLOCK, line] and after[-1]["user_words_in_task"].startswith("all of them")
    assert not [c for c in srv.calls if c[1] in ("hold", "pause", "stop")]
    print(f"\nSWAP on the stub: {fix['ms']} ms from the line to the new start")


def test_a_swap_that_does_not_fit_goes_to_the_planner(servers):
    """Block 3 is already in its slot when the user names it: the rewritten plan would undo a done
    condition, so code refuses the swap and the planner gets the line as a correction."""
    srv = servers(pick_s=1.2)
    line = "actually, block 3"
    versions = [Goal({"block_3": "tray_slot_1", "block_1": "tray_slot_2"}), Goal({"block_3": "tray_slot_2"})]
    decider, llm = oracle_models(srv, versions, {line: Utt("swap", version=1, swap_to="block_3")})
    rec = Recording(decider)
    user = ScriptedUser([Line(reaching_for(srv, "block_1"), line)])
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=25)), "Block 3 in tray slot 1, then block 1 in tray slot 2.")
    assert r.outcome == "done", r.log
    _, _, qs, c = rec.text_event(line)
    assert c.fix == "swap_target:block_3" and c.route == "stay_local"                      # the decider wanted the swap
    refused = next(e for e in r.log if e["note"] == "fix_refused")
    assert refused["why"].startswith("the swap does not fit: ") and "done condition" in refused["why"]
    assert not [e for e in r.log if e["note"] == "fix"]
    assert [x["kind"] for x in r.llm] == ["plan", "replan"]                                # the planner instead
    replan = next(e for e in r.log if e["note"] == "llm_request" and e["request"] == "replan")
    assert "the swap does not fit" in replan["why"]
    assert where(srv)["block_3"] == "tray_slot_2"
    assert [e["words"] for e in r.log if e["note"] == "user_words"] == [[ "Block 3 in tray slot 1, then block 1 in tray slot 2.", line]]


def test_a_remark_naming_a_block_changes_nothing(servers):
    srv = servers(pick_s=1.2)
    line = "block 3 looks a bit scratched"
    rec, llm = _one_block_models(srv, line, kind="chatter")
    user = ScriptedUser([Line(reaching_for(srv, "block_1"), line)])
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=20)), ONE_BLOCK)
    assert r.outcome == "done", r.log
    _, _, qs, c = rec.text_event(line)
    assert "swap_target:block_3" in qs["in_plan_fix"]["criteria"]                           # offered
    assert (c.route, c.fix) == ("stay_local", "none")                                      # declined
    assert [(s[1], s[2]["object"]) for s in starts(srv)] == [("pick_and_place", "block_1")]
    assert [x["kind"] for x in r.llm] == ["plan"] and not [e for e in r.log if e["note"] in ("fix", "fix_refused")]
    assert where(srv)["block_1"] == "tray_slot_1"


def test_the_swap_is_not_offered_once_the_object_is_grasped(servers):
    srv = servers(pick_s=1.2)
    line = "actually, block 3"
    rec, llm = _one_block_models(srv, line)
    user = ScriptedUser([Line(lambda *_: srv.world().arms["arm_0"].holding == "block_1", line)])
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=20)), ONE_BLOCK)
    assert r.outcome == "done", r.log
    _, _, qs, c = rec.text_event(line)
    assert not any(k.startswith("swap_target") for k in qs.get("in_plan_fix", {}).get("criteria", {}))
    assert c.fix == "none"
    assert where(srv)["block_1"] == "tray_slot_1"                                           # the step in hand finished as written
