"""The three design fixes from the first live run (docs/live-results.md "After the fixes"), offline:

  P1  the decision request carries `user_words` (the user's task-level words, verbatim, newest last,
      no transient commands) next to `task` (the plan's reading), and the questions judge a correction
      against the user's words;
  P2  a new plan is routed by the plan check alone: the route and in-plan-fix questions are not asked
      on plan_arrived, their answers are ignored, and a plan-check yes still sends the plan back; the
      replan-loop limit stays as the backstop;
  P3  a restated task that repeats a transient command is not adopted: the brain keeps the previous
      reading, logs why, and the planner instructions say the restatement holds only the end result
      in the user's words;
  P4  the live decider defaults to Jev; the Decisions API stays selectable.

Episodes run against the physics-free stub server with e12v2's oracle mocks, under asyncio.run.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from robojev.brain import Brain, BrainConfig
from robojev.brain.core import plan as P
from robojev.brain.core.combine import Gates, combine
from robojev.brain.core.data import ArmObs, Event, Geometry, Obj, Place, WorldState
from robojev.brain.core.decision import View, build_questions, build_state
from robojev.brain.core.planner import INSTRUCTIONS
from robojev.brain.eval.scenarios import SORT_TASK, Utt, sort_goal
from robojev.brain.mocks import LLMReply, Line, ScriptedUser, oracle_models
from robojev.brain.rules import ReplanGuard, enforce_gate, is_transient, transient_mentions
from robojev.brain.schema import instructions
from robojev.robots import stub as S

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_models as LM  # noqa: E402

CORRECTION = "actually, highest on the left"
SECOND = "no wait, I did mean lowest on the left"
WAIT, GO_ON = "hang on a second", "ok, go on"


# ---------------------------------------------------------------- helpers


@pytest.fixture
def servers():
    made = []

    def make(numbers=(5, 1, 3), pick_s=0.4, **kw) -> S.StubRobotServer:
        objs, places = S.sort_scene(numbers)
        srv = S.StubRobotServer(S.widowx_like(), objs, places, durations={"pick_and_place": pick_s}, **kw)
        made.append(srv)
        return srv.open()
    yield make
    for s in made:
        s.close()


def numbers(srv) -> dict:
    return {o.id: int(o.label) for o in srv.world().objects.values()}


def where(srv) -> dict:
    return {o.id: o.where for o in srv.world().objects.values()}


def carrying(srv, oid):
    return lambda *_: srv.world().arms["arm_0"].holding == oid


def episode(brain, task=SORT_TASK):
    return asyncio.run(brain.run(task))


class Recording:
    """A decider in front of another: keeps every (event, view, request state, questions) it saw."""

    def __init__(self, inner):
        self.inner, self.name, self.seen = inner, "recording", []

    def decide(self, ev, view):
        out = self.inner.decide(ev, view)
        self.seen.append((ev, view, out["request"]["state"], out["request"]["questions"]))
        return out


def world() -> WorldState:
    """Two blocks on the table, one slot, the gripper empty: enough for build_state."""
    geom = Geometry(12, 2, 7, 4.5, 1.5, 8, 20, 1, (0, 60, -40, 40, 0, 30), 5)
    objs = {"block_1": Obj("block_1", "block 1", "red", 1, "4 cm", 20, 0, 2, "table"),
            "block_7": Obj("block_7", "block 7", "red", 7, "4 cm", 30, 0, 2, "table")}
    places = {"tray_slot_1": Place("tray_slot_1", "tray slot 1", "slot", 40, 0, 1)}
    return WorldState(0, objs, places, None, ArmObs(0, 0, 12, None), geom)


def plain_view(**kw) -> View:
    base = dict(t=0, task=SORT_TASK, user_messages=[], state=world(), plan=None, queue=[], status={}, current=None,
                phase="not started on any step yet", mode="running", pause_reason=None, replan_pending=False, names={},
                path_to=None, hand=None, runnable_other=None)
    return View(**{**base, **kw})


# ---------------------------------------------------------------- P1: the user's words in the request


def test_user_words_default_to_the_task_and_sit_beside_the_plans_reading():
    v = plain_view()
    s = build_state(Event(1, 0, "user_text", 'the user typed: "hi"', data={"text": "hi"}), v)
    assert s["user_words"] == [SORT_TASK] and s["task"] == SORT_TASK          # no plan yet: both are the task
    v2 = plain_view(user_words=[SORT_TASK, CORRECTION], plan={"reading": "Place the blocks in descending order",
                                                              "steps": [], "done_when": [], "constraints": []})
    s2 = build_state(Event(1, 0, "step_done", "step finished"), v2)
    assert s2["user_words"] == [SORT_TASK, CORRECTION]                         # verbatim, newest last
    assert s2["task"] == "Place the blocks in descending order"               # the restatement stays
    assert list(s2).index("user_words") < list(s2).index("task")


def test_questions_judge_a_correction_against_the_users_words():
    v = plain_view(user_words=[SORT_TASK])
    qs = build_questions(Event(1, 0, "user_text", 'the user typed: "x"', data={"text": "x"}), v)
    rn, route = json.dumps(qs["right_now"]["instructions"]), json.dumps(qs["route"]["instructions"])
    for text in (rn, route):
        assert "`user_words`" in text and "`task`" in text
        assert "in their" in text and "own words" in text.replace("their words", "their own words")
    assert "how the current plan reads it" in rn and "how the current plan reads it" in route
    assert "not against `task`" in route


def test_the_brain_tracks_the_users_words_through_a_correction_and_a_wait(servers):
    """The correction's own decision sees only the task in `user_words`; after it, every request
    carries [task, correction]; the wait and the go-on never enter it."""
    srv = servers(pick_s=1.2)
    nums = numbers(srv)
    user = ScriptedUser([Line(carrying(srv, "block_1"), WAIT), Line(1.4, GO_ON), Line(carrying(srv, "block_1"), CORRECTION)])
    decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)],
                                 {CORRECTION: Utt("correction", version=1), WAIT: Utt("wait"), GO_ON: Utt("go_on")})
    rec = Recording(decider)
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=25)))
    assert r.outcome == "done", r.log
    assert [t for _, t in user.said] == [WAIT, GO_ON, CORRECTION]
    by_text = {ev.data.get("text"): (ev, view, state) for ev, view, state, _ in rec.seen if ev.kind == "user_text"}
    assert by_text[WAIT][2]["user_words"] == [SORT_TASK] and by_text[GO_ON][2]["user_words"] == [SORT_TASK]
    assert by_text[CORRECTION][2]["user_words"] == [SORT_TASK]                 # judged against the task as typed
    assert by_text[CORRECTION][2]["event"]["user_just_said"] == CORRECTION
    t_corr = by_text[CORRECTION][0].t
    later = [state["user_words"] for ev, _, state, _ in rec.seen if ev.t > t_corr and ev.kind != "user_text"]
    assert later and all(w == [SORT_TASK, CORRECTION] for w in later), later
    assert all(WAIT not in w and GO_ON not in w for _, _, state, _ in rec.seen for w in [state["user_words"]])
    assert [e["words"] for e in r.log if e["note"] == "user_words"] == [[SORT_TASK, CORRECTION]]
    assert where(srv) == {"block_5": "tray_slot_1", "block_3": "tray_slot_2", "block_1": "tray_slot_3"}


def test_user_words_keep_two_corrections_in_order_newest_last(servers):
    srv = servers(pick_s=1.2)
    nums = numbers(srv)
    user = ScriptedUser([Line(carrying(srv, "block_1"), CORRECTION), Line(carrying(srv, "block_5"), SECOND)])
    decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)],
                                 {CORRECTION: Utt("correction", version=1), SECOND: Utt("correction", version=0)})
    rec = Recording(decider)
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=30)))
    assert r.outcome == "done", r.log
    assert [t for _, t in user.said] == [CORRECTION, SECOND]
    second = next(state for ev, _, state, _ in rec.seen if ev.kind == "user_text" and ev.data.get("text") == SECOND)
    assert second["user_words"] == [SORT_TASK, CORRECTION]
    last = [state["user_words"] for ev, _, state, _ in rec.seen][-1]
    assert last == [SORT_TASK, CORRECTION, SECOND]
    assert where(srv) == {"block_1": "tray_slot_1", "block_3": "tray_slot_2", "block_5": "tray_slot_3"}


# ---------------------------------------------------------------- P2: a new plan is routed by the plan check alone


def _choice(pick, opts, conf):
    return {"type": "choice", "choice": pick, "confidence": conf, "probabilities": {o: (conf if o == pick else 0.0) for o in opts}}


def _plan(reading="sort the blocks"):
    return {"reading": reading, "steps": [{"id": "s1", "skill": "move_object", "object": "block_1", "target": "tray_slot_1", "direction": "none"}],
            "done_when": [{"id": "d1", "text": "block_1 is in tray_slot_1", "object": "block_1", "relation": "in", "target": "tray_slot_1"}],
            "constraints": []}


def test_plan_arrived_asks_no_route_and_no_fix_only_right_now_and_the_plan_check():
    v = plain_view(plan=_plan(), queue=["s1"], status={"s1": "pending"}, names={"block_1": "block 1", "block_7": "block 7"})
    qs = build_questions(Event(1, 0, "plan_arrived", "the first plan arrived: sort the blocks"), v)
    assert set(qs) == {"right_now", "check.block_7"}
    qs2 = build_questions(Event(2, 0, "step_done", "step finished"), v)
    assert "route" in qs2                                                      # every other event still asks the router


@pytest.mark.parametrize("route, conf, fix_conf", [("stay_local", 0.3, 1.0), ("fast_llm", 0.95, 1.0), ("capable_llm", 0.9, 1.0),
                                                   ("stay_local", 0.9, 0.2), ("ask_user", 0.9, 1.0)])
def test_a_new_plan_with_a_clean_plan_check_stays_local_whatever_the_route_answer(route, conf, fix_conf):
    """The recorded send-backs (unsure stay_local, confident replan) no longer send a good plan back."""
    answers = {"right_now": _choice("carry_on", ["carry_on", "hold", "pause"], 0.9),
               "in_plan_fix": _choice("skip", ["none", "skip"], fix_conf),
               "route": _choice(route, ["stay_local", "fast_llm", "capable_llm", "ask_user"], conf),
               "check.block_7": {"type": "noul", "noul": 0.03}}
    c = combine(Event(1, 0, "plan_arrived", "a new plan arrived"), answers, plain_view(), Gates())
    assert (c.route, c.fix, c.problems) == ("stay_local", "none", [])
    assert c.conf["route"] == 1.0 and c.conf["in_plan_fix"] == 1.0            # code's certainty: the loop's gate cannot escalate it
    assert enforce_gate(c, 0.8) is c
    assert "plan check" in c.reason


@pytest.mark.parametrize("p", [0.97, 0.6, 0.5, 0.4, 0.26])                   # yes, and the unsure band
def test_a_plan_check_yes_or_an_unsure_answer_still_sends_the_plan_back(p):
    answers = {"right_now": _choice("carry_on", ["carry_on", "hold", "pause"], 0.9),
               "route": _choice("stay_local", ["stay_local", "fast_llm", "capable_llm", "ask_user"], 0.99),
               "check.block_7": {"type": "noul", "noul": p}}
    c = combine(Event(1, 0, "plan_arrived", "a new plan arrived"), answers, plain_view(names={"block_7": "block 7"}), Gates())
    assert c.route == "fast_llm" and c.reason == "plan check"
    assert len(c.problems) == 1 and "block 7" in c.problems[0]
    assert ("unsure" in c.problems[0]) == (p < 0.5)


def test_a_clean_plan_check_without_a_check_question_stays_local_too():
    c = combine(Event(1, 0, "plan_arrived", "a new plan arrived"), {"right_now": _choice("carry_on", ["carry_on", "hold"], 0.9)},
                plain_view(), Gates())
    assert c.route == "stay_local"


def test_other_events_still_route_by_the_answer():
    answers = {"right_now": _choice("carry_on", ["carry_on", "hold", "pause"], 0.9),
               "route": _choice("fast_llm", ["stay_local", "fast_llm", "capable_llm", "ask_user"], 0.9)}
    assert combine(Event(1, 0, "user_text", "x", data={"text": "x"}), answers, plain_view(), Gates()).route == "fast_llm"
    low = {**answers, "route": _choice("stay_local", ["stay_local", "fast_llm", "capable_llm", "ask_user"], 0.5)}
    assert combine(Event(1, 0, "scene_change", "x"), low, plain_view(), Gates()).route == "fast_llm"


def test_a_plan_check_that_always_finds_a_problem_still_hits_the_replan_loop_backstop(servers):
    """The backstop (rules.ReplanGuard): three plans sent back by the plan check since the user last
    spoke, and later plans run as they come."""
    srv = servers()
    decider, llm = oracle_models(srv, [sort_goal(numbers(srv))])
    inner = decider.backend

    class Flagging:
        live = False

        def answer(self, state, questions, ref=None):
            r = inner.answer(state, questions, ref=ref)
            if ref[0].kind == "plan_arrived":
                r["answers"]["check.block_9"] = {"noul": 0.95}                # an object the plan leaves out, says the check
            return r
    decider.backend = Flagging()
    r = episode(Brain(srv, decider, llm, None, BrainConfig(max_s=20)))
    assert r.outcome == "done", r.log
    sent = [d for d in r.decisions if d["event"] == "plan_arrived"]
    assert len(sent) == 3 and all(d["combined"].route == "fast_llm" and d["combined"].reason == "plan check" for d in sent)
    assert r.loops == 1 and r.events["plan_arrived"] == 4
    assert sum(1 for e in r.log if e["note"] == "plan_trusted") == 1
    assert ReplanGuard().limit == 3


# ---------------------------------------------------------------- P3: a transient command never enters the task


@pytest.mark.parametrize("text", [WAIT, GO_ON, "wait", "Wait!", "one sec", "hold on", "resume", "continue please", "pause",
                                  "ok carry on", "go ahead", "stop"])
def test_transient_commands_are_recognised(text):
    assert is_transient(text)


@pytest.mark.parametrize("text", [CORRECTION, SECOND, "wait, put the red one on the left", "hold the block over the tray",
                                  "put block 7 on the right", "continue with the green one last"])
def test_task_words_are_not_transient(text):
    assert not is_transient(text)


def test_transient_mentions_finds_what_a_reading_carries_of_a_wait():
    leaked = ("Line up block_1, block_3 and block_5 in ascending numerical order; after the user said "
              "“hang on a second” and then “ok, go on,” continue the arrangement.")      # the live leak (H5)
    assert transient_mentions(leaked, [WAIT, GO_ON]) == [WAIT, GO_ON, "continue"]
    # a paraphrase counts once the user has typed a transient line (the September handover readings)
    assert transient_mentions("Hand block_4 to the person, then wait for a second.", ["hang on a sec"]) == ["wait"]
    assert transient_mentions("Pause the handover; the person has asked the robot to wait.", ["hang on a sec"]) == ["pause", "wait"]
    assert transient_mentions("Continue sorting the blocks, lowest on the left", ["ok, continue"]) == ["continue"]
    assert transient_mentions("Line the blocks up, lowest number on the left.", [WAIT, GO_ON, CORRECTION]) == []
    assert transient_mentions("Continue sorting the blocks", [CORRECTION]) == []    # no transient line typed: nothing to leak
    assert transient_mentions("Hand block_4 to the person and wait until they take it", []) == []


class LeakingLLM:
    """The mock planner forced to leak: every restated task repeats the user's wait and go-on, as the
    live Decisions API run did (docs/live-results.md H5)."""

    def __init__(self, backend):
        self.backend, self.readings = backend, []

    def call(self, req, ref=None):
        r = self.backend.call(req, ref=ref)
        out = json.loads(r.raw)
        out["reading"] = out["reading"] + f"; after the user said “{WAIT}” and then “{GO_ON},” continue the arrangement."
        self.readings.append(out["reading"])
        return LLMReply(json.dumps(out), r.latency_ms, r.in_tok, r.out_tok, None, getattr(r, "model", "mock"))


def test_a_leaking_restatement_is_not_adopted_and_the_log_says_why(servers):
    """The wait is typed before the first plan arrives (so the first plan's reading leaks) and the
    go-on after; a correction mid-carry makes the replan's reading leak too. Neither reading is kept:
    the first plan keeps the task, the replan keeps the plan's previous reading, and both are logged."""
    srv = servers(pick_s=1.2)
    nums = numbers(srv)
    user = ScriptedUser([Line(0.0, WAIT), Line(0.8, GO_ON), Line(carrying(srv, "block_1"), CORRECTION)])
    decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)],
                                 {CORRECTION: Utt("correction", version=1), WAIT: Utt("wait"), GO_ON: Utt("go_on")})
    leaky = LeakingLLM(llm["fast_llm"])
    rec = Recording(decider)
    r = episode(Brain(srv, rec, {"fast_llm": leaky, "capable_llm": leaky}, user, BrainConfig(max_s=30)))
    assert r.outcome == "done", r.log
    assert leaky.readings and all(WAIT in x and GO_ON in x for x in leaky.readings)   # the planner really leaked
    kept = [e for e in r.log if e["note"] == "reading_kept"]
    assert [e["request"] for e in kept] == ["plan"] + ["replan"] * (len(leaky.readings) - 1)
    # only lines the user had typed by then count: the first plan (0.3 s) came before the go-on (0.8 s)
    assert all(WAIT in e["leaked"] and set(e["leaked"]) <= {WAIT, GO_ON, "go on", "continue"} and "transient" in e["why"] for e in kept)
    assert kept[-1]["leaked"] == [WAIT, GO_ON, "continue"]
    assert kept[0]["kept"] == SORT_TASK and kept[0]["rejected"] == leaky.readings[0]
    arrived = [ev.text for ev, _, _, _ in rec.seen if ev.kind == "plan_arrived"]
    assert arrived and all(WAIT not in t and GO_ON not in t for t in arrived)
    tasks = [state["task"] for _, _, state, _ in rec.seen]
    assert tasks and all(WAIT not in t and GO_ON not in t for t in tasks)          # nothing a decider saw mentioned the wait
    assert where(srv) == {"block_5": "tray_slot_1", "block_3": "tray_slot_2", "block_1": "tray_slot_3"}


def test_a_clean_restatement_is_adopted_unchanged(servers):
    srv = servers(pick_s=1.2)
    nums = numbers(srv)
    user = ScriptedUser([Line(carrying(srv, "block_1"), CORRECTION)])
    decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)], {CORRECTION: Utt("correction", version=1)})
    rec = Recording(decider)
    r = episode(Brain(srv, rec, llm, user, BrainConfig(max_s=25)))
    assert r.outcome == "done", r.log
    assert not [e for e in r.log if e["note"] == "reading_kept"]
    readings = [ev.text for ev, _, _, _ in rec.seen if ev.kind == "plan_arrived"]
    assert len(readings) == 2 and CORRECTION in readings[1]                         # the mock folds the correction in


def test_the_planner_is_told_the_restatement_holds_only_the_end_result_in_the_users_words():
    text = instructions(S.widowx_like(), "arm_0")
    for want in ("- reading:", "ONLY the end result", "user's own words", '"lowest number on the left"', 'not "ascending numeric order"',
                 "Never put a pause, wait, resume or acknowledgement in it"):
        assert want in text, want
        assert want in INSTRUCTIONS
    ids = {"objects": ["block_1"], "places": ["tray_slot_1"]}
    for schema in (P.plan_schema(ids), P.diff_schema(ids)):
        d = schema["properties"]["reading"]["description"]
        assert "ONLY the end result" in d and "user's own words" in d and "never a pause, wait, resume or acknowledgement" in d


# ---------------------------------------------------------------- P4: the default live decider


def test_the_live_decider_defaults_to_jev_and_the_decisions_api_is_selectable():
    assert LM.decider_name(None) == "mock"                   # the default test run needs no keys
    assert LM.decider_name("live") == "jev" == LM.DEFAULT_LIVE_DECIDER
    assert LM.decider_name("jev") == "jev" and LM.decider_name("decisions") == "decisions"
    with pytest.raises(ValueError):
        LM.decider_name("gpt")
