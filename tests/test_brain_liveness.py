"""The rules the live spelling runs asked for (docs/live-results.md, "Live re-run after the fixes"):

- a pause must end: a hand pause ends in code once the hand has gone; any other pause the decider
  keeps re-asking about becomes a question to the user, whose answer ends it;
- a failed replan keeps the hold (the old plan is the one the event said is wrong), asks the planner
  once more, then pauses the holding arms and asks the user;
- `user_words` takes a typed line only when the decider itself routed it off the plan;
- a correction the plan has folded in is shown as such, so later steps are not re-judged against it;
- a routine step_done stays local on the decider's choice alone (combine.routine);
- the plan check groups spare objects by label and asks about counts.

Each runs on the physics-free stub with the oracle mocks, under asyncio.run (no pytest-asyncio).
"""
import asyncio
import json
import time
from dataclasses import replace

from robojev.brain import Brain, BrainConfig
from robojev.brain.core.combine import Gates, combine, routine
from robojev.brain.core.data import ArmObs, Combined, Event, Geometry, Obj, Place, WorldState
from robojev.brain.core.decision import View, build_questions, build_state, unmentioned_groups
from robojev.brain.eval.scenarios import SORT_TASK, Utt, sort_goal
from robojev.brain.mocks import LLMReply, Line, ScriptedUser, oracle_models
from robojev.brain.rules import enforce_gate
from robojev.robots import stub as S

CORRECTION = "actually, highest on the left"
WAIT, GO_ON = "hang on a second", "ok, go on"
FAST = dict(paused_idle_s=0.5, hold_s=0.5)


# ---------------------------------------------------------------- helpers


def make(numbers=(5, 1, 3), pick_s=1.0, **kw) -> S.StubRobotServer:
    objs, places = S.sort_scene(numbers)
    return S.StubRobotServer(S.widowx_like(), objs, places, durations={"pick_and_place": pick_s}, **kw).open()


def numbers(srv) -> dict:
    return {o.id: int(o.label) for o in srv.world().objects.values()}


def where(srv) -> dict:
    return {o.id: o.where for o in srv.world().objects.values()}


def carrying(srv, oid):
    return lambda *_: srv.world().arms["arm_0"].holding == oid


def episode(brain, task=SORT_TASK):
    return asyncio.run(brain.run(task))


def notes(r, note):
    return [e for e in r.log if e["note"] == note]


class Unsure:
    """A decision backend that answers `resume` with no confidence (under the right-now gate), so the
    cautious fallback keeps the arm paused: the live ep 5 pattern, both deciders, 41 asks in a row."""

    def __init__(self, inner):
        self.inner, self.live = inner, False

    def answer(self, state, questions, ref=None):
        r = self.inner.answer(state, questions, ref=ref)
        rn = r["answers"].get("right_now")
        if rn and rn.get("choice") == "resume":
            rn["confidence"] = 0.25
        return r


# ---------------------------------------------------------------- a pause must end


def test_a_hand_pause_ends_in_code_once_the_hand_has_gone_when_the_decider_stays_unsure():
    srv = make()
    try:
        srv.person_hand(carrying(srv, "block_1"), for_s=0.8, dist=0.15)
        decider, llm = oracle_models(srv, [sort_goal(numbers(srv))])
        decider.backend = Unsure(decider.backend)
        r = episode(Brain(srv, decider, llm, None, BrainConfig(max_s=20, **FAST)))
        assert r.outcome == "done", r.log
        left = next(t for t, what in srv.script_log if what == "person's hand leaves")
        resumed = notes(r, "resumed")
        assert len(resumed) == 1 and "hand that has gone" in resumed[0]["why"]
        t_resume = next(t for t, _, mode in srv.history if mode == "running" and t > left)
        assert left + FAST["paused_idle_s"] <= t_resume < left + 3 * FAST["paused_idle_s"]
        # no decision resumed it: every answer after the hand left was the cautious carry_on
        gone = next(d for d in r.decisions if d["event"] == "scene_change" and "gone" in d["text"])
        assert (gone["combined"].right_now, gone["combined"].right_now_raw) == ("carry_on", "resume")   # unsure: stayed paused
        assert not notes(r, "ask")
        assert not [d for d in r.decisions if d["event"] == "paused_idle" and d["t_event"] > left]   # code answered, not a re-ask
    finally:
        srv.close()


def test_a_wait_pause_the_decider_keeps_re_asking_about_becomes_a_question_and_the_answer_ends_it():
    srv = make(pick_s=1.2)
    try:
        user = ScriptedUser([Line(carrying(srv, "block_1"), WAIT), Line(lambda: bool(user.asked), GO_ON)])
        decider, llm = oracle_models(srv, [sort_goal(numbers(srv))], {WAIT: Utt("wait"), GO_ON: Utt("go_on")})
        r = episode(Brain(srv, decider, llm, user, BrainConfig(max_s=20, **FAST)))
        assert r.outcome == "done", r.log
        assert len(user.asked) == 1 and "Should it go on?" in user.asked[0] and WAIT in user.asked[0]
        asked = notes(r, "ask")
        assert len(asked) == 1 and asked[0]["why"].startswith("paused for 1 s")
        idle = [d for d in r.decisions if d["event"] == "paused_idle"]
        assert len(idle) == 2 and all(d["combined"].right_now == "carry_on" for d in idle)   # the wait held, twice
        t_wait = next(t for t, text in user.said if text == WAIT)
        t_go = next(t for t, text in user.said if text == GO_ON)
        assert t_go - t_wait >= 2 * FAST["paused_idle_s"]
        # paused all the way from the wait to the answer, and running after it
        modes = [(t, m) for t, _, m in srv.history]
        assert [m for t, m in modes if t_wait < t < t_go] == ["paused"]
        assert any(m == "running" and t > t_go for t, m in modes)
        assert not notes(r, "resumed")
        assert where(srv) == {"block_1": "tray_slot_1", "block_3": "tray_slot_2", "block_5": "tray_slot_3"}
    finally:
        srv.close()


def test_a_remark_made_while_a_hand_is_in_view_is_a_hand_pause():
    """Live ep 5 with Jev: the remark "Sorry, just reaching over." was decided before the hand's own
    event and its pause was recorded as the user's; code now attributes a non-transient line decided
    with a hand in view to the hand, so the pause ends when the hand goes."""
    srv = make()
    try:
        remark = "Sorry, just reaching over."
        srv.person_hand(carrying(srv, "block_1"), for_s=0.8, dist=0.15)
        user = ScriptedUser([Line(lambda: bool(srv.world().hands), remark)])
        decider, llm = oracle_models(srv, [sort_goal(numbers(srv))], {remark: Utt("chatter")})
        decider.backend = Unsure(decider.backend)
        r = episode(Brain(srv, decider, llm, user, BrainConfig(max_s=20, **FAST)))
        assert r.outcome == "done", r.log
        assert [t for t, _ in user.said]
        assert len(notes(r, "resumed")) == 1 and not user.asked
    finally:
        srv.close()


# ---------------------------------------------------------------- a failed replan keeps the hold


class FailingReplans:
    """An LLM backend whose first `n` replan calls come back unusable (not a plan at all)."""

    def __init__(self, backend, n: int):
        self.backend, self.n, self.replans = backend, n, 0

    def call(self, req, ref=None):
        if req.kind == "replan":
            self.replans += 1
            if self.replans <= self.n:
                return LLMReply("not a plan", 50.0, 10, 2, None, "mock")
        return self.backend.call(req, ref=ref)


def test_a_failed_replan_keeps_the_hold_retries_once_then_pauses_and_asks_the_user():
    srv = make(pick_s=1.2)
    try:
        nums = numbers(srv)
        user = ScriptedUser([Line(carrying(srv, "block_1"), CORRECTION), Line(lambda: bool(user.asked), CORRECTION)])
        decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)], {CORRECTION: Utt("correction", version=1)})
        # the planner validates and retries once per request, so 4 bad calls = 2 unusable replans
        failing = FailingReplans(llm["fast_llm"], 4)
        r = episode(Brain(srv, decider, {"fast_llm": failing, "capable_llm": failing}, user, BrainConfig(max_s=25, **FAST)))
        assert r.outcome == "done", r.log
        assert len(notes(r, "plan_retry")) == 1 and len(notes(r, "plan_abandoned")) == 1
        assert notes(r, "plan_abandoned")[0]["paused"] == ["arm_0"]
        assert len(user.asked) == 1 and user.asked[0].startswith("I could not work out a new plan for") and CORRECTION in user.asked[0]
        assert r.failed_plans == 2
        # the corrected move never completed: block 1 went from the gripper straight to its new slot
        assert srv.status("sk1").state == "cancelled"
        starts = [c[2][2] for c in srv.calls if c[1] == "start"]
        assert starts[:2] == [{"object": "block_1", "place": "tray_slot_1"}, {"object": "block_1", "place": "tray_slot_3"}]
        assert where(srv) == {"block_5": "tray_slot_1", "block_3": "tray_slot_2", "block_1": "tray_slot_3"}
        # held from the correction until the user's answer, never running the old plan in between
        t_corr = next(t for t, text in user.said if text == CORRECTION)
        t_ans = [t for t, text in user.said if text == CORRECTION][1]
        between = [m for t, _, m in srv.history if t_corr < t < t_ans]
        assert between and "running" not in between, between
    finally:
        srv.close()


# ---------------------------------------------------------------- user_words


def _brain_for_unit():
    srv = make()
    decider, llm = oracle_models(srv, [sort_goal(numbers(srv))])
    return srv, Brain(srv, decider, llm, None, BrainConfig(max_s=5))


def test_user_words_take_a_line_only_when_the_decider_itself_routed_it_off_the_plan():
    srv, brain = _brain_for_unit()
    try:
        brain.user_words = [SORT_TASK]
        conf = {"right_now": 0.9, "in_plan_fix": 1.0, "route": 0.6}

        def decided(i, text, route, route_raw, reason):
            ev = Event(i, 0.0, "user_text", f'the user typed: "{text}"', data={"text": text})
            brain._apply_decision(ev, {"combined": Combined("carry_on", "none", route, reason, route_raw=route_raw, conf=conf), "latency_ms": 1.0})
        decided(1, "It slid a little.", "fast_llm", "stay_local", "low confidence")     # the gate escalated an unsure remark
        assert brain.user_words == [SORT_TASK]
        decided(2, CORRECTION, "fast_llm", "fast_llm", "fast_llm")                     # the decider's own routing
        assert brain.user_words == [SORT_TASK, CORRECTION]
        decided(3, "the ones on the left", "ask_user", "ask_user", "ask_user")
        assert brain.user_words == [SORT_TASK, CORRECTION, "the ones on the left"]
        decided(4, WAIT, "fast_llm", "fast_llm", "fast_llm")                           # a transient never does
        assert brain.user_words == [SORT_TASK, CORRECTION, "the ones on the left"]
    finally:
        srv.close()


def test_a_correction_the_gate_escalated_joins_user_words_once_its_plan_changes_the_steps():
    """Handoff 8: the Decisions API's route on the correction was an unsure stay_local (0.40) that the
    gate escalated; the replan read the line and changed the plan, but rule 4 kept it out of
    `user_words`. A line the planner changed the plan for is a correction, whatever the route's
    confidence. A remark whose replan changes nothing stays out."""
    srv = make(pick_s=1.2)
    try:
        nums = numbers(srv)
        remark = "It slid a little."
        user = ScriptedUser([Line(carrying(srv, "block_1"), CORRECTION), Line(carrying(srv, "block_5"), remark)])
        decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)],
                                     {CORRECTION: Utt("correction", version=1), remark: Utt("chatter")})

        class Unsure:
            name = "unsure"

            def decide(self, ev, view):
                out = decider.decide(ev, view)
                if ev.kind == "user_text":       # the decider leans stay_local but under the gate: escalated to the planner
                    out["combined"] = replace(out["combined"], route="stay_local", route_raw="stay_local", fix="none",
                                              conf={**out["combined"].conf, "route": 0.4})
                return out
        r = episode(Brain(srv, Unsure(), llm, user, BrainConfig(max_s=25)))
        assert r.outcome == "done", r.log
        said = [d for d in r.decisions if d["event"] == "user_text"]
        assert len(said) == 2 and all(d["combined"].route == "fast_llm" and d["combined"].route_raw == "stay_local" for d in said)
        words = [e["words"] for e in r.log if e["note"] == "user_words"]
        assert words == [[SORT_TASK, CORRECTION]], words            # the correction once, the remark never
        assert where(srv) == {"block_5": "tray_slot_1", "block_3": "tray_slot_2", "block_1": "tray_slot_3"}
    finally:
        srv.close()


def test_diff_changes_ignores_a_new_reading_alone():
    from robojev.brain.core.plan import diff_changes
    same = {"reading": "new words", "new_steps": [], "pending_order": ["s2", "s3"], "drop_done_when": [], "new_done_when": [],
            "drop_constraints": [], "new_constraints": []}
    assert not diff_changes(same, ["s2", "s3"])
    assert diff_changes({**same, "pending_order": ["s3", "s2"]}, ["s2", "s3"])
    assert diff_changes({**same, "new_steps": [{"id": "n1"}]}, ["s2", "s3"])
    assert diff_changes({**same, "drop_done_when": ["d1"]}, ["s2", "s3"])


def test_the_scripted_user_can_answer_the_brains_question():
    """The slice's user never answered "Should it go on?" (Handoff 8: with nobody answering, every
    re-ask went to the planner until the request cap). ScriptedUser(on_ask=...) answers like a person."""
    srv = make(pick_s=1.2)
    try:
        user = ScriptedUser([Line(carrying(srv, "block_1"), WAIT)], on_ask=GO_ON, ask_delay_s=0.2)
        decider, llm = oracle_models(srv, [sort_goal(numbers(srv))], {WAIT: Utt("wait"), GO_ON: Utt("go_on")})
        r = episode(Brain(srv, decider, llm, user, BrainConfig(max_s=20, **FAST)))
        assert r.outcome == "done", r.log
        assert len(user.asked) == 1 and [t for _, t in user.said] == [WAIT, GO_ON]
        t_ask = next(e["t"] for e in r.log if e["note"] == "ask")
        t_go = next(t for t, text in user.said if text == GO_ON)
        assert t_ask <= t_go < t_ask + 1.0
    finally:
        srv.close()


def _view(**kw) -> View:

    geom = Geometry(12, 2, 7, 4.5, 1.5, 8, 20, 1, (0, 60, -40, 40, 0, 30), 5)
    objs = {"block_1": Obj("block_1", "block 1", "red", 1, "4 cm", 20, 0, 2, "table")}
    state = WorldState(0, objs, {"tray_slot_1": Place("tray_slot_1", "tray slot 1", "slot", 40, 0, 1)}, None, ArmObs(0, 0, 12, None), geom)
    plan = {"reading": "Line the blocks up, highest on the left.", "steps": [], "done_when": [], "constraints": []}
    base = dict(t=0, task=SORT_TASK, user_messages=[], state=state, plan=plan, queue=[], status={}, current=None,
                phase="between steps", mode="running", pause_reason=None, replan_pending=False, names={},
                path_to=None, hand=None, runnable_other=None, user_words=[SORT_TASK, CORRECTION])
    return View(**{**base, **kw})


def test_the_decider_is_told_which_user_words_the_plan_has_already_folded_in():
    ev = Event(1, 0, "step_done", "step finished")
    s = build_state(ev, _view())
    assert s["user_words"] == [SORT_TASK, CORRECTION] and s["user_words_in_task"].startswith("all of them")
    s2 = build_state(ev, _view(unplanned_words=[CORRECTION], replan_pending=True))
    assert s2["user_words_in_task"]["not_yet"] == [CORRECTION]
    qs = build_questions(ev, _view())
    assert "`user_words_in_task`" in qs["route"]["instructions"]["ask_about"]
    assert "A correction that `task` already states" in qs["route"]["instructions"]["ask_about"]
    assert "user_words_in_task" not in json.dumps(qs["right_now"])                        # the right-now question is as before
    assert "user_words_in_task" not in build_state(ev, _view(plan=None))                  # no plan yet: nothing is folded in
    # never on a typed line or a pause re-ask: there the question is the line, or whether to go on
    # (Handoff 8: the Decisions API read the field on "ok, go on" as a reason not to resume)
    for kind, data in (("user_text", {"text": "ok, go on"}), ("paused_idle", {})):
        e = Event(2, 0, kind, "x", data=data)
        assert "user_words_in_task" not in build_state(e, _view(unplanned_words=[CORRECTION]))
        assert "user_words_in_task" not in json.dumps(build_questions(e, _view()))


def test_the_brain_marks_a_correction_folded_in_once_its_plan_arrives():
    srv = make(pick_s=1.2)
    try:
        nums = numbers(srv)
        user = ScriptedUser([Line(carrying(srv, "block_1"), CORRECTION)])
        srv.person_hand(lambda w: bool(user.said), for_s=0.6, dist=0.3)          # a scene event while the new plan is on its way
        decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)],
                                     {CORRECTION: Utt("correction", version=1)}, llm_median_ms=1500.0)
        seen = []

        class Rec:
            name = "rec"

            def decide(self, ev, view):
                out = decider.decide(ev, view)
                seen.append((ev, out["request"]["state"]))
                return out
        r = episode(Brain(srv, Rec(), llm, user, BrainConfig(max_s=25)))
        assert r.outcome == "done", r.log
        t_plan = next(e["t"] for e in r.log if e["note"] == "plan_applied" and "diff" in e)
        # scene events while the new plan was on its way (the first may predate the correction's own decision)
        during = [s for ev, s in seen if ev.kind == "scene_change" and ev.t < t_plan and len(s["user_words"]) == 2]
        after = [s for ev, s in seen if ev.t > t_plan + 0.05 and ev.kind not in ("user_text", "paused_idle")]
        assert during and all(s["user_words_in_task"]["not_yet"] == [CORRECTION] for s in during), during
        assert during[0]["change_being_planned"]["user_said"] == CORRECTION
        assert after and all(s["user_words_in_task"].startswith("all of them") for s in after)
        assert all(s["user_words"] == [SORT_TASK, CORRECTION] for s in after)
    finally:
        srv.close()


# ---------------------------------------------------------------- a routine step_done stays local


def _choice(pick, opts, conf):
    return {"choice": pick, "probabilities": {o: (conf if o == pick else (1 - conf) / (len(opts) - 1)) for o in opts}, "confidence": conf}


def test_a_routine_step_done_keeps_the_deciders_stay_local_under_the_gate():
    answers = {"right_now": _choice("carry_on", ["carry_on", "hold", "pause"], 0.9),
               "route": _choice("stay_local", ["stay_local", "fast_llm", "capable_llm", "ask_user"], 0.45)}
    v = _view()
    c = combine(Event(1, 0, "step_done", "step finished"), answers, v, Gates())
    assert (c.route, c.fix) == ("stay_local", "none") and c.reason.startswith("routine")
    # the same confidence on any other event, or with a fix in play, still escalates
    assert combine(Event(1, 0, "scene_change", "x"), answers, v, Gates()).route == "fast_llm"
    fixed = {**answers, "in_plan_fix": _choice("skip", ["none", "skip"], 0.45)}
    assert combine(Event(1, 0, "step_done", "step finished"), fixed, v, Gates()).route == "fast_llm"
    # a plan-check problem is never routine
    assert not routine("step_done", "none", ["x is left out"]) and routine("step_done", "none", [])
    # the loop's own gate agrees
    low = Combined("carry_on", "none", "stay_local", "stay_local", conf={"route": 0.4, "in_plan_fix": 1.0})
    assert enforce_gate(low, 0.8, "step_done").route == "stay_local"
    assert enforce_gate(low, 0.8, "scene_change").route == "fast_llm"
    assert enforce_gate(replace(low, fix="skip"), 0.8, "step_done").route == "fast_llm"


# ---------------------------------------------------------------- the plan check and spare letters


def _letters_view() -> View:
    geom = Geometry(12, 2, 7, 4.5, 1.5, 8, 20, 1, (0, 60, -40, 40, 0, 30), 5)
    objs = {f"letter_{k}": Obj(f"letter_{k}", nm, "wood", None, "4 cm", 20 + 5 * i, 0, 2, "table", label=lab)
            for i, (k, nm, lab) in enumerate([("A1", "the letter A", "A"), ("A2", "the second letter A", "A"),
                                               ("A3", "the third letter A", "A"), ("B1", "the letter B", "B"),
                                               ("P1", "the letter P", "P")])}
    objs["cup"] = Obj("cup", "the cup", "blue", None, "8 cm", 50, 10, 4, "table")
    places = {f"slot_{i}": Place(f"slot_{i}", f"position {i + 1}", "slot", 40, 10 * i, 1) for i in range(3)}
    state = WorldState(0, objs, places, None, ArmObs(0, 0, 12, None), geom)
    plan = {"reading": "Spell PA.", "constraints": [], "done_when": [],
            "steps": [{"id": "s1", "skill": "move_object", "object": "letter_P1", "target": "slot_0", "direction": "none", "distance": None, "seconds": None},
                      {"id": "s2", "skill": "move_object", "object": "letter_A1", "target": "slot_1", "direction": "none", "distance": None, "seconds": None}]}
    return View(t=0, task="Spell PA.", user_messages=[], state=state, plan=plan, queue=["s1", "s2"], status={"s1": "pending", "s2": "pending"},
                current=None, phase="not started", mode="running", pause_reason=None, replan_pending=False,
                names={o.id: o.name for o in objs.values()}, path_to=None, hand=None, runnable_other=None)


def test_the_plan_check_groups_spare_objects_by_label_and_asks_about_counts():
    v = _letters_view()
    groups = unmentioned_groups(v)
    assert groups == [{"ids": ["cup"], "label": None, "used": []},                       # in id order
                      {"ids": ["letter_A2", "letter_A3"], "label": "A", "used": ["letter_A1"]},
                      {"ids": ["letter_B1"], "label": "B", "used": []}]
    ev = Event(1, 0, "plan_arrived", "a new plan arrived: Spell PA.")
    qs = build_questions(ev, v)
    assert set(qs) == {"right_now", "check.letter_A2", "check.letter_B1", "check.cup"}       # one per group, not per object
    a = qs["check.letter_A2"]
    assert "Of the 3 objects labelled A in view, `plan` already uses 1 (the letter A)" in a["instructions"]

    assert "the second letter A (and 1 more labelled A)" in a["instructions"]
    assert "more objects labelled A than the 1 the plan uses" in a["instructions"]
    assert "spare" in a["criteria"]["false"] and "needs more than 1 object labelled A" in a["criteria"]["true"]
    b = qs["check.letter_B1"]
    assert "the letter B (labelled B)" in b["instructions"] and "already uses" not in b["instructions"]
    assert "the cup" in qs["check.cup"]["instructions"] and "labelled" not in qs["check.cup"]["instructions"]
    s = build_state(ev, v)
    assert s["plan"]["objects_the_plan_never_mentions"] == [
        "the cup", "the second letter A, the third letter A (labelled A; the plan uses 1 other object labelled A)",
        "the letter B (labelled B; the plan uses 0 other objects labelled B)"]

    # a confident "spare" keeps the plan; a count the task does need sends it back, naming the group
    keep = {"right_now": _choice("carry_on", ["carry_on", "hold", "pause"], 0.9), "check.letter_A2": {"noul": 0.05},
            "check.letter_B1": {"noul": 0.04}, "check.cup": {"noul": 0.02}}
    assert combine(ev, keep, v, Gates()).route == "stay_local"
    back = {**keep, "check.letter_A2": {"noul": 0.9}}
    c = combine(ev, back, v, Gates())
    assert c.route == "fast_llm" and c.problems == ["the second letter A is something the task asks about but the plan leaves out"]


def test_labels_reach_the_brain_from_the_robots_world():
    from robojev.brain.adapt import core_state, geometry
    srv = make()
    try:
        spec = srv.manifest().arms[0]
        st = core_state(srv.world(), "arm_0", geometry(spec, 0.2), 0.2)
        assert {o.label for o in st.objects.values()} == {"1", "3", "5"}
    finally:
        srv.close()


# ---------------------------------------------------------------- a correction before the first plan


class SlowFirstPlan:
    """An LLM backend whose first plan call takes `delay_s` (the live rules ep 7: an 11 s first plan)."""

    def __init__(self, backend, delay_s: float):
        self.backend, self.delay_s, self.plans = backend, delay_s, 0

    def call(self, req, ref=None):
        if req.kind == "plan":
            self.plans += 1
            if self.plans == 1:
                time.sleep(self.delay_s)
        return self.backend.call(req, ref=ref)


def test_a_correction_typed_before_the_first_plan_arrives_is_planned_for():
    """The first plan request is built when the run starts; a correction typed while it is on its way
    is not in it. Live (battery, rules, ep 7): "Could you spell PANE instead, please?" at 9.8 s, the
    first plan at 11.0 s read OPENAI, and the arm spelled OPENAI. The brain now asks for the first plan
    again with the line in it; the earlier request goes stale."""
    srv = make(pick_s=0.8)
    try:
        nums = numbers(srv)
        start = time.monotonic()
        user = ScriptedUser([Line(lambda *_: time.monotonic() - start > 0.3, CORRECTION)])
        decider, llm = oracle_models(srv, [sort_goal(nums), sort_goal(nums, desc=True)], {CORRECTION: Utt("correction", version=1)})
        slow = SlowFirstPlan(llm["fast_llm"], 1.0)
        r = episode(Brain(srv, decider, {"fast_llm": slow, "capable_llm": slow}, user, BrainConfig(max_s=20, **FAST)))
        assert r.outcome == "done", r.log
        assert [e["request"] for e in notes(r, "llm_request")][:2] == ["plan", "plan"]
        assert notes(r, "plan_stale")
        assert where(srv) == {"block_5": "tray_slot_1", "block_3": "tray_slot_2", "block_1": "tray_slot_3"}
    finally:
        srv.close()
