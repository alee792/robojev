"""E12 update (experiments/e12v2/): the v2 design closed loop in a text world, with controls. Offline:
mock Jev and mock LLM (both read the oracle). Live runs need Jev and OpenAI keys."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
from e12v2.core import plan as P  # noqa: E402
from e12v2.core.combine import Ablation, Gates, combine  # noqa: E402
from e12v2.core.data import Combined, Event  # noqa: E402
from e12v2.core.decision import RIGHT_NOW_CRITERIA, View, build_questions, build_state  # noqa: E402
from e12v2.core.harness import Episode, HarnessConfig, LoopDetector, is_stop  # noqa: E402
from e12v2.core.planner import PlannerClient  # noqa: E402
from e12v2.eval import cli  # noqa: E402
from e12v2.eval import oracle as O  # noqa: E402
from e12v2.eval import scenarios as S  # noqa: E402
from e12v2.eval.cli import RunOpts, run_episode  # noqa: E402
from e12v2.eval.mocks import MockJev, MockLLM  # noqa: E402
from e12v2.sim.person import Beat, Person, carrying, say  # noqa: E402
from e12v2.sim.skills import make_skills  # noqa: E402
from e12v2.sim.world import Noise  # noqa: E402

ALL = list(S.ALL)
CORE = list(S.CORE)


def run(arm, name, seed=12, **kw):
    return run_episode(arm, name, seed, RunOpts(**kw))


# ---------------------------------------------------------------- arms, closed loop


@pytest.mark.parametrize("name", ALL)
def test_oracle_arm_completes_every_scenario_with_zero_violations(name):
    rec, res, obs, _ = run("oracle", name)
    assert rec["completed"], (name, res.outcome)
    assert rec["contacts"] == 0 and rec["violations"] == 0
    assert all(ok == n for ok, n in rec["agree"].values())
    assert rec["corr_missed"] == 0 and rec["spurious"] == 0


@pytest.mark.parametrize("name", ALL)
def test_mock_jev_at_zero_error_matches_the_oracle(name):
    a, *_ = run("oracle", name)
    b, *_ = run("jev", name, mock_jev_error=0.0, mock_conf_noise=0.05)   # right answers stay above the 0.7 gate
    for f in ("completed", "contacts", "violations", "llm_calls", "disturbances", "recovered", "corr_missed", "spurious"):
        assert a[f] == b[f], f
    assert all(ok == n for ok, n in b["agree"].values())


@pytest.mark.parametrize("name", ALL)
def test_mock_at_nonzero_error_runs_to_the_end(name):
    rec, res, _, _ = run("jev", name, mock_jev_error=0.3, mock_conf_noise=0.2, mock_llm_error=0.2, noise=Noise(0.0, 0.05, 1.0), max_s=120)
    assert res.outcome in ("done", "timeout", "gave_up") and rec["jev_requests"] > 0
    assert rec["violations"] == 0          # constraints are code: no model error can break them


@pytest.mark.parametrize("name", ALL)
def test_rules_arm_runs(name):
    rec, res, _, _ = run("rules", name)
    assert rec["jev_requests"] == 0 and rec["violations"] == 0
    assert rec["completed"]


@pytest.mark.parametrize("name", ALL)
def test_always_llm_arm_runs_with_the_mock_llm(name):
    rec, res, _, _ = run("always_llm", name)
    assert rec["jev_requests"] == 0 and rec["llm_calls"] >= 1 and rec["violations"] == 0
    assert res.outcome in ("done", "timeout", "gave_up")


@pytest.mark.parametrize("arm", ["jev-no-right-now", "jev-no-in-plan-fix", "jev-no-router"])
def test_ablations_run(arm):
    for name in ("sort_hand_in_path", "sort_take_back", "sort_correction"):
        rec, res, _, _ = run(arm, name)
        assert res.outcome in ("done", "timeout", "gave_up")


def test_right_now_keeps_the_hand_clear_and_its_absence_does_not():
    assert run("jev", "sort_hand_in_path", mock_jev_error=0.0)[0]["contacts"] == 0
    assert run("jev-no-right-now", "sort_hand_in_path", mock_jev_error=0.0)[0]["contacts"] >= 1
    assert run("always_llm", "sort_hand_in_path")[0]["contacts"] >= 1


def test_correction_changes_behaviour_at_jev_speed_not_llm_speed():
    j = run("jev", "sort_correction", mock_jev_error=0.0)[0]
    a = run("always_llm", "sort_correction")[0]
    assert j["completed"] and a["completed"]
    assert j["corr_s"][0] <= 0.5 < a["corr_s"][0]


def test_no_router_sends_hands_and_chatter_to_the_llm():
    a = run("jev", "sort_hand_in_path", mock_jev_error=0.0)[0]
    b = run("jev-no-router", "sort_hand_in_path", mock_jev_error=0.0)[0]
    assert b["llm_calls"] > a["llm_calls"]


def test_plan_check_catches_a_left_out_block():
    sc = S.make("sort_static", 12)
    llm = MockLLM(sc, sc.world, seed=0)
    llm._inject = lambda plan, kind: MockLLM._inject(llm, plan, "omit") if not hasattr(llm, "_done") and not setattr(llm, "_done", 1) else plan
    from e12v2.core.combine import JevDecider
    ep = Episode(sc.world, sc.person, make_skills(sc.world), JevDecider(MockJev(sc, sc.world, 0.0, 0.0, 1, name="oracle")),
                 PlannerClient({"fast_llm": llm}), HarnessConfig(max_s=200))
    res = ep.run(sc.task)
    first = res.decisions[0]
    assert first["event"] == "plan_arrived" and first["combined"].route == "fast_llm" and first["combined"].problems
    assert res.outcome == "done" and O.achieved(sc.goal_now(), sc.world.truth())


# ---------------------------------------------------------------- combiner and gates (unit)


def view(hand=False, mode="running", current=None):
    h = {"band": "near", "approaching": True, "in_path": True, "held_out": False} if hand else None
    return NS(hand=h, mode=mode, current=current, name=lambda o: o)


def ch(c, conf):
    return {"choice": c, "confidence": conf}


def ans(rn="carry_on", rc=0.95, fix="none", fc=0.95, route="stay_local", roc=0.95, **extra):
    return {"right_now": ch(rn, rc), "in_plan_fix": ch(fix, fc), "route": ch(route, roc), **extra}


EV = Event(1, 0, "user_text", "the user typed: x", data={"text": "x"})
FAIL = Event(2, 0, "step_failed", "step failed", "b", "s1", {"reason": "nothing grasped"})


def test_right_now_decides_hold_or_carry_on_while_the_llm_replans():
    c = combine(EV, ans("hold", route="fast_llm"), view(), Gates())
    assert (c.right_now, c.route, c.fix) == ("hold", "fast_llm", "none")
    c = combine(EV, ans("carry_on", route="fast_llm"), view(), Gates())
    assert (c.right_now, c.route) == ("carry_on", "fast_llm")


def test_cautious_fallback_on_low_right_now_confidence():
    g = Gates(right_now=0.5)
    assert combine(EV, ans("carry_on", 0.3), view(), g).right_now == "hold"
    assert combine(EV, ans("carry_on", 0.3), view(hand=True), g).right_now == "pause"
    assert combine(EV, ans("re_target", 0.3), view(), g).right_now == "hold"
    assert combine(EV, ans("resume", 0.3), view(mode="paused"), g).right_now == "carry_on"   # stay paused
    assert combine(EV, ans("back_off", 0.3), view(hand=True), g).right_now == "pause"
    assert combine(EV, ans("back_off", 0.6), view(hand=True), g).right_now == "back_off"
    assert combine(EV, None, view(hand=True), g).right_now == "pause"                         # Jev error


def test_in_plan_fix_needs_stay_local_and_the_stay_local_gate_on_the_weakest_confidence():
    g = Gates(stay_local=0.7)
    c = combine(FAIL, ans(fix="retry", fc=0.9, roc=0.9), view(), g)
    assert (c.route, c.fix) == ("stay_local", "retry")
    c = combine(FAIL, ans(fix="retry", fc=0.6, roc=0.9), view(), g)
    assert (c.route, c.fix, c.reason) == ("fast_llm", "none", "low confidence")
    c = combine(FAIL, ans(fix="retry", fc=0.9, roc=0.65), view(), g)
    assert (c.route, c.fix) == ("fast_llm", "none")
    c = combine(FAIL, ans(fix="retry", fc=0.9, route="fast_llm"), view(), g)
    assert (c.route, c.fix) == ("fast_llm", "none")                  # a fix applies only if the route stays local
    assert combine(FAIL, ans(fix="retry", fc=0.6, roc=0.9), view(), Gates(stay_local=0.5)).fix == "retry"


def test_thresholds_are_per_route():
    g = Gates(stay_local=0.7, fast_llm=0.3, ask_user=0.5)
    assert combine(EV, ans(route="fast_llm", roc=0.4), view(), g).route == "fast_llm"
    assert combine(EV, ans(route="fast_llm", roc=0.2), view(), g).route == "capable_llm"
    assert combine(EV, ans(route="ask_user", roc=0.4), view(), g).route == "capable_llm"
    assert combine(EV, ans(route="ask_user", roc=0.6), view(), g).route == "ask_user"
    assert combine(EV, ans(route="capable_llm", roc=0.05), view(), g).route == "capable_llm"
    assert combine(EV, ans(roc=0.69), view(), g).route == "fast_llm"


def test_code_rules_on_top_of_the_answers():
    g = Gates()
    assert combine(FAIL, ans(), view(), g).route == "fast_llm"           # a failure with no fix can't stay local
    pa = Event(3, 0, "plan_arrived", "plan")
    assert combine(pa, ans(**{"check.block_7": {"noul": 0.9}}), view(), g).route == "fast_llm"
    assert combine(pa, ans(**{"check.block_7": {"noul": 0.4}}), view(), g).route == "fast_llm"   # unsure: cautious
    assert combine(pa, ans(**{"check.block_7": {"noul": 0.05}}), view(), g).route == "stay_local"
    del_fix = ans()
    del del_fix["in_plan_fix"]                                            # not asked: none, certain
    assert combine(Event(4, 0, "step_done", "done"), del_fix, view(), g).route == "stay_local"


def test_ablations_in_the_combiner():
    c = combine(EV, ans("pause", route="fast_llm"), view(), Gates(), Ablation(right_now=False))
    assert c.right_now == "carry_on"
    c = combine(FAIL, ans(fix="retry"), view(), Gates(), Ablation(in_plan_fix=False))
    assert (c.fix, c.route) == ("none", "fast_llm")
    c = combine(FAIL, ans(fix="retry", route="fast_llm"), view(), Gates(), Ablation(router=False))
    assert (c.fix, c.route) == ("retry", "stay_local")
    c = combine(EV, ans(route="stay_local"), view(), Gates(), Ablation(router=False))
    assert c.route == "fast_llm"
    assert combine(Event(5, 0, "step_done", "d"), ans(), view(), Gates(), Ablation(router=False)).route == "stay_local"


def test_loop_detector():
    d = LoopDetector(3)
    assert not any(d.record(x) for x in "ABABA") and d.record("B")
    d = LoopDetector(3)
    assert not any(d.record(x) for x in "AAAAAAAA")
    assert not any(LoopDetector(3).record(x) for x in "ABABAC")


def test_typed_stop_is_code():
    assert is_stop("stop") and is_stop("Stop!") and is_stop("  STOP now")
    assert not is_stop("don't stop") and not is_stop("stop at the tray") and not is_stop("wait")
    sc = S.make("sort_static", 12)
    sc.person.beats.append(Beat("user: stop", lambda w, t: carrying(w, 1), say("stop"), disturbs=False))
    seen = []

    class Spy:
        name = "spy"

        def decide(self, ev, v):
            seen.append(ev)
            return {"combined": Combined("carry_on", "none", "stay_local"), "latency_ms": 1.0, "jev": False}

    llm = {"fast_llm": MockLLM(sc, sc.world)}
    ep = Episode(sc.world, sc.person, make_skills(sc.world), Spy(), PlannerClient(llm), HarnessConfig(max_s=120))
    res = ep.run(sc.task)
    assert res.outcome == "stopped" and not any(e.kind == "user_text" for e in seen)


def test_stop_button_stops():
    sc = S.make("sort_static", 12)
    sc.person.stop_at = 80
    rec, res, *_ = (lambda: None, None)
    ep = Episode(sc.world, sc.person, make_skills(sc.world), cli.RulesDecider(), PlannerClient({"fast_llm": MockLLM(sc, sc.world)}),
                 HarnessConfig(max_s=60))
    assert ep.run(sc.task).outcome == "stopped" and ep.t == 80


class Scripted:
    """A decider that answers every event the same way, except the user's text."""
    name = "scripted"

    def __init__(self, on_text: Combined):
        self.on_text = on_text

    def decide(self, ev, v):
        c = self.on_text if ev.kind == "user_text" else Combined("carry_on", "none", "stay_local")
        return {"combined": c, "latency_ms": 150.0, "jev": False}


@pytest.mark.parametrize("rn, moves", [("hold", False), ("carry_on", True)])
def test_the_arm_holds_or_carries_on_while_the_llm_replans(rn, moves):
    sc = S.make("sort_correction", 12)
    ep = Episode(sc.world, sc.person, make_skills(sc.world), Scripted(Combined(rn, "none", "fast_llm")),
                 PlannerClient({"fast_llm": MockLLM(sc, sc.world)}), HarnessConfig(max_s=200))
    track = []

    class Obs:
        def on_event(self, ep, ev): pass
        def on_decision(self, ep, ev, out, v): pass
        def on_llm(self, ep, res, ev): pass
        def on_tick(self, ep):
            track.append((ep.t, ep.mode, ep.replan_pending, (ep.state.arm.x, ep.state.arm.y)))

    ep.obs = Obs()
    res = ep.run(sc.task)
    assert res.outcome == "done" and O.achieved(sc.goal_now(), sc.world.truth())
    t_text = next(b.fired_t for b in sc.person.beats if b.text)
    during = [x for x in track if x[0] > t_text + 3 and x[2]]
    assert during, "a replan was pending after the correction"
    modes = {m for _, m, _, _ in during}
    xy = {p for *_, p in during}
    if moves:
        assert modes == {"running"} and len(xy) > 3
    else:
        assert modes == {"holding"} and len(xy) <= 3      # a hold may first lift to a safe point
    after = [m for t, m, pend, _ in track if t > t_text + 3 and not pend]
    assert after and after[0] == "running"                  # the new plan ends the hold


# ---------------------------------------------------------------- the plan: schema, validator, diffs


def state_of(name="sort_static", seed=12):
    sc = S.make(name, seed)
    return sc, sc.world.truth()


def good(name="sort_static"):
    sc, st = state_of(name)
    return O.oracle_plan(sc.goal_now(), st), st, sc


def test_validator_accepts_oracle_plans_for_every_scenario():
    for name in ALL:
        p, st, sc = good(name)
        assert P.validate(p, st) == [], name


def _err(p, st, needle, holding=None):
    errs = P.validate(p, st, holding)
    assert any(needle in e for e in errs), errs


def test_validator_rejects():
    p, st, _ = good()
    q = json.loads(json.dumps(p))
    q["steps"][1]["target"] = "tray_slot_1"
    _err(q, st, "still occupied by")
    q = json.loads(json.dumps(p))
    q["steps"].pop()
    _err(q, st, "would still be false")
    q = json.loads(json.dumps(p))
    q["steps"][0]["id"] = q["steps"][1]["id"]
    _err(q, st, "unique")
    q = json.loads(json.dumps(p))
    q["steps"][0]["target"] = "tray"
    _err(q, st, "needs a slot, a bin or table")
    q = json.loads(json.dumps(p))
    q["steps"][0]["object"] = "nope"
    _err(q, st, "unknown object")
    q = json.loads(json.dumps(p))
    q["done_when"] = []
    _err(q, st, "at least one done condition")
    _err(p, st, "first step must put it somewhere", holding="block_12")
    q = json.loads(json.dumps(p))
    q["constraints"] = [{"id": "c1", "text": "", "kind": "dont_touch", "object": "block_3", "place": "none"}]
    _err(q, st, "must not be touched")
    q = json.loads(json.dumps(p))
    q["constraints"] = [{"id": "c1", "text": "", "kind": "keep_out_of", "object": "block_3", "place": "tray"}]
    _err(q, st, "a constraint forbids")


def test_validator_on_stacks():
    p, st, sc = good("tower")
    assert [s["skill"] for s in p["steps"]] == ["stack_on", "stack_on"]
    q = json.loads(json.dumps(p))
    q["steps"].reverse()                              # yellow on red before red is on blue: fine for the validator...
    q["steps"].append({"id": "s9", "skill": "stack_on", "object": "block_2", "target": "red_block", "direction": "none"})
    _err(q, st, "already has")                        # ...but a second block on red is not


def test_diff_roundtrip():
    p, st, sc = good("sort_correction")
    queue = [s["id"] for s in p["steps"]][2:]
    new = O.oracle_plan(sc.versions[1], st)
    d = P.make_diff(p, queue, new)
    assert P.diff_errors(p, d) == []
    p2, q2 = P.apply_diff(p, d)
    byid = P.steps_by_id(p2)
    assert [P.step_goal(byid[i]) for i in q2] == [P.step_goal(s) for s in new["steps"]]
    assert {(c["object"], c["target"]) for c in p2["done_when"]} == {(c["object"], c["target"]) for c in new["done_when"]}
    assert P.validate(p2, st, None, q2) == []
    kept = P.make_diff(p, [s["id"] for s in p["steps"]], p)
    assert kept["new_steps"] == [] and kept["drop_done_when"] == [] and kept["pending_order"] == [s["id"] for s in p["steps"]]


def test_schemas_are_strict_everywhere_with_enums_from_the_world():
    _, st = state_of("sort_constraint")
    ids = P.schema_ids(st)

    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    for schema in (P.plan_schema(ids), P.diff_schema(ids), P.react_schema(ids, ("carry_on", "hold"))):
        walk(schema)
    step = P.plan_schema(ids)["properties"]["steps"]["items"]["properties"]
    assert "green_block" in step["object"]["enum"] and "tray_slot_1" in step["target"]["enum"]


def test_planner_prompt_prefix_is_stable_and_openai_request_is_strict(monkeypatch):
    from e12v2.core import planner as PL
    sc, st = state_of()
    names = sc.world.names()
    a = PL.plan_request("t", [], st, names)
    p, _, _ = good()
    b = PL.replan_request("replan", "t", ["x"], st, names, p, [s["id"] for s in p["steps"]], {}, {}, "why")
    assert a.instructions == b.instructions                        # the cacheable prefix
    sent = {}

    class Responses:
        def create(self, **kw):
            sent.update(kw)
            return NS(status="completed", output_text=json.dumps(p), model="m", incomplete_details=None,
                      usage=NS(input_tokens=100, output_tokens=50, input_tokens_details=NS(cached_tokens=80)))

    from e12v2 import backends
    monkeypatch.setattr("e13_branches.planner._openai", lambda: NS(OpenAI=lambda **kw: NS(responses=Responses())))
    be = backends.openai_backend("some-model", effort="low")
    out = PlannerClient({"fast_llm": be}).plan("t", [], st, names)
    assert out.ok and out.attempts[0]["cached_tok"] == 80
    assert sent["text"]["format"]["strict"] is True and sent["prompt_cache_key"] == "e12v2-planner"
    assert sent["reasoning"] == {"effort": "low"} and sent["store"] is False and "temperature" not in sent


def test_invalid_plan_gets_one_retry_with_the_errors():
    sc, st = state_of()
    p, _, _ = good()
    bad = json.loads(json.dumps(p))
    bad["done_when"] = []
    outs, reqs = [json.dumps(bad), json.dumps(p)], []

    class Be:
        def call(self, req, ref=None):
            reqs.append(req)
            return NS(raw=outs.pop(0), latency_ms=5.0, in_tok=1, out_tok=1, error=None, model="m")

    r = PlannerClient({"fast_llm": Be()}).plan("t", [], st, sc.world.names())
    assert r.ok and len(r.attempts) == 2 and "at least one done condition" in reqs[1].input


# ---------------------------------------------------------------- the decision request


def test_decision_request_has_three_groups_and_literal_facts():
    sc = S.make("sort_moved_target", 12)
    box = {}

    class Grab(MockJev):
        def answer(self, state, questions, ref=None):
            if ref[0].kind == "scene_change" and "state" not in box:
                box.update(state=state, questions=questions)
            return super().answer(state, questions, ref)

    from e12v2.core.combine import JevDecider
    ep = Episode(sc.world, sc.person, make_skills(sc.world), JevDecider(Grab(sc, sc.world, 0.0, 0.0)),
                 PlannerClient({"fast_llm": MockLLM(sc, sc.world)}), HarnessConfig(max_s=200))
    ep.run(sc.task)
    st, qs = box["state"], box["questions"]
    assert {"right_now", "route"} <= set(qs)
    assert "re_target" in qs["right_now"]["criteria"] and "resume" not in qs["right_now"]["criteria"]
    assert all("not_for" in c for q in qs.values() if q["type"] == "choice" for c in q["criteria"].values())
    assert "nothing grasped yet" in st["plan_position"]["arm_in_step"]
    assert st["event_object"]["is_the_current_steps_object"] == "yes"
    assert "hand" in st and "nearby_objects" in st


def test_spotter_criteria_do_not_restate_the_oracles_thresholds():
    txt = json.dumps(RIGHT_NOW_CRITERIA)
    assert not any(ch.isdigit() for ch in txt) and "cm" not in txt


# ---------------------------------------------------------------- the split: core / sim / eval / oracle


def _imports(path: Path) -> set:
    mods = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom):
            mods.add(("." * n.level) + (n.module or ""))
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
    return mods


CORE_FILES = sorted((EXP / "e12v2" / "core").glob("*.py"))


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_imports_nothing_from_sim_or_eval(path):
    for m in _imports(path):
        assert "sim" not in m.split(".") and "eval" not in m.split("."), (path.name, m)
        assert not m.startswith(("e12_blocksworld", "e13_branches", "e11")), (path.name, m)


@pytest.mark.parametrize("path", CORE_FILES + [EXP / "e12v2" / "eval" / "controls.py"], ids=lambda p: p.name)
def test_system_under_test_does_not_import_the_oracle(path):
    for m in _imports(path):
        parts = set(m.strip(".").split("."))
        assert not parts & {"oracle", "mocks", "scenarios", "metrics"}, (path.name, m)


def test_core_has_no_task_specific_words():
    """No task logic in core, and no scenario's objects or phrasing in its prompts (held-out stays held out)."""
    for path in CORE_FILES:
        src = path.read_text().lower()
        for w in ("ascending", "descending", "highest", "lowest", "green", "yellow", "tower", "block 4"):
            assert w not in src, (path.name, w)


# ---------------------------------------------------------------- CLI: dry run, replay


def test_dry_run_writes_samples_and_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(["--dry-run", "--scenarios", "sort_moved_target,sort_take_back,sort_hand_in_path,sort_correction,sort_wait_go"])
    out = capsys.readouterr().out
    files = {p.name for p in (tmp_path / "e12v2_samples").iterdir()}
    for k in ("plan_arrived", "step_done", "step_failed", "scene_change_hand", "scene_change_object", "user_text", "plan_done_unmet"):
        assert f"jev_{k}.json" in files, k
    assert {"llm_plan.json", "llm_replan.json", "llm_react.json"} <= files
    body = json.loads((tmp_path / "e12v2_samples" / "jev_user_text.json").read_text())["request"]
    assert set(body) == {"state", "model", "questions"}
    assert "Jev (live arms" in out and "LLM:" in out and "$" in out


def test_mock_run_logs_and_replay_sweeps_gates(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(["--arms", "jev,rules", "--scenarios", "sort_hand_in_path,sort_correction", "--log", "--mock-jev-error", "0.3"])
    out = capsys.readouterr().out
    assert "Pass criteria" in out and "Gate sweep" in out
    log = next(tmp_path.glob("e12v2_*.jsonl"))
    cli.main(["--replay", str(log)])
    out = capsys.readouterr().out
    assert "| 0.3 |" in out and "| 0.9 |" in out


def test_show_rules(capsys):
    cli.main(["--show-rules"])
    out = capsys.readouterr().out
    assert "U1" in out and "H3" in out and "M4" in out and "P3" in out


def test_live_jev_client_shape_errors_and_budget(monkeypatch):
    import common
    from e12v2 import backends
    calls = []
    monkeypatch.setattr(common, "make_client", lambda: NS(close=lambda: None))

    def fake_ask(client, state, questions):
        calls.append((state, questions))
        if len(calls) == 2:
            return common.Call(529, 80.0, {"error": "overloaded"}, "r2", None, None)
        return common.Call(200, 140.0, {"answers": {"route": {"choice": "stay_local", "confidence": 0.9}}}, "r1", 120.0, 900)
    monkeypatch.setattr(common, "ask", fake_ask)
    j = backends.JevHTTP(max_requests=2)
    a = j.answer({"s": 1}, {"q": {}})
    assert a["answers"]["route"]["choice"] == "stay_local" and a["in_tok"] == 900 and a["latency_ms"] == 140.0
    b = j.answer({"s": 1}, {"q": {}})
    assert b["answers"] is None and j.errors == 1                         # -> the combiner's jev_error path
    with pytest.raises(backends.BudgetExceeded):
        j.answer({"s": 1}, {"q": {}})


def test_a_single_earlier_correction_reaches_every_later_decision():
    # With exactly one user message, the state used to drop `user_said_earlier` (a `> 1` check), so
    # after "actually, highest on the left" Jev saw only the original task and a plan contradicting it.
    sc = S.make("sort_correction", 12)
    seen = []

    class Grab(MockJev):
        def answer(self, state, questions, ref=None):
            if ref[0].kind != "user_text":
                seen.append(state)
            return super().answer(state, questions, ref)

    from e12v2.core.combine import JevDecider
    ep = Episode(sc.world, sc.person, make_skills(sc.world), JevDecider(Grab(sc, sc.world, 0.0, 0.0)),
                 PlannerClient({"fast_llm": MockLLM(sc, sc.world)}), HarnessConfig(max_s=200))
    ep.run(sc.task)
    assert len(ep.user_messages) == 1
    after = [s for s in seen if "user_said_earlier" in s]
    assert after and all(ep.user_messages[0] in s["user_said_earlier"] for s in after)
    assert "user_said_earlier" not in seen[0]      # nothing said yet at the first plan
