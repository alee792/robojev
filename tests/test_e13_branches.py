"""E13 harness (experiments/e13_branches/): the evaluation oracle on hand-worked cases, the generic plan
validator, the router's decision rule, a full offline run with mock LLM + mock Jev, and the boundary
between the system under test and the oracle. Offline; live runs need OpenAI and Jev keys."""
import ast
import json
import random
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import common  # noqa: E402
from e13_branches import cli, oracle  # noqa: E402
from e13_branches import plan as P  # noqa: E402
from e13_branches import planner as PL  # noqa: E402
from e13_branches import router as R  # noqa: E402
from e13_branches import score as S  # noqa: E402
from e13_branches.mocks import MockLLM  # noqa: E402
from e13_branches.world import FAMILIES, TABLE, Block, Correction, Scene, Task, make_task, make_tasks, scene_view  # noqa: E402


def scene(*specs, stickers=None):
    """specs: (colour, number, size_cm)."""
    return Scene([Block(f"{c}_{n}", n, c, s) for c, n, s in specs], stickers)


def corr(op, cat="coverable", text="x"):
    return Correction("c", text, cat, op)


SC = scene(("red", 5, 3.0), ("blue", 2, 5.0), ("red", 9, 2.0), ("green", 4, 4.0))


# ---------------------------------------------------------------- oracle, hand-worked

def test_sort_by_number_lowest_left_and_reverse():
    t = Task("t", "sort_number", "", {"key": "number", "desc": False}, SC)
    assert oracle.target(t).canonical == {"blue_2": "tray_slot_1", "green_4": "tray_slot_2", "red_5": "tray_slot_3", "red_9": "tray_slot_4"}
    rev = oracle.target(t, corr({"op": "flip"})).canonical
    assert rev == {"red_9": "tray_slot_1", "red_5": "tray_slot_2", "green_4": "tray_slot_3", "blue_2": "tray_slot_4"}
    assert not oracle.target(t).check(rev)


def test_biggest_first_by_size():
    t = Task("t", "sort_size", "", {"key": "size", "desc": True}, SC)
    assert oracle.target(t).canonical == {"blue_2": "tray_slot_1", "green_4": "tray_slot_2", "red_5": "tray_slot_3", "red_9": "tray_slot_4"}
    small = oracle.target(t, corr({"op": "set_order", "desc": False})).canonical
    assert small["red_9"] == "tray_slot_1" and small["blue_2"] == "tray_slot_4"


def test_leave_a_colour_out_keeps_order_and_allows_gaps():
    t = Task("t", "sort_number", "", {"key": "number", "desc": False}, SC)
    tg = oracle.target(t, corr({"op": "assign_first", "select": {"colour": "red"}, "dest": TABLE}, "uncoverable"))
    assert tg.canonical == {"blue_2": "tray_slot_1", "green_4": "tray_slot_2", "red_5": TABLE, "red_9": TABLE}
    assert tg.check({"blue_2": "tray_slot_1", "green_4": "tray_slot_4", "red_5": TABLE, "red_9": TABLE})       # gap is fine
    assert not tg.check({"blue_2": "tray_slot_2", "green_4": "tray_slot_1", "red_5": TABLE, "red_9": TABLE})   # wrong order
    assert not tg.check({"blue_2": "tray_slot_1", "green_4": "tray_slot_1", "red_5": TABLE, "red_9": TABLE})   # shared slot


def test_group_by_colour_into_matching_bins_and_swap():
    sc = scene(("red", 1, 2.0), ("blue", 2, 3.0), ("red", 3, 4.0), stickers={"left_bin": "blue", "right_bin": "red"})
    t = Task("t", "group_colour", "", {"stickers": dict(sc.stickers)}, sc)
    assert oracle.target(t).canonical == {"red_1": "right_bin", "blue_2": "left_bin", "red_3": "right_bin"}
    assert oracle.target(t, corr({"op": "swap_bins"})).canonical == {"red_1": "left_bin", "blue_2": "right_bin", "red_3": "left_bin"}


def test_all_red_to_left_bin_rest_stay():
    t = Task("t", "colour_to_bin", "", {"colour": "red", "bin": "left_bin"}, SC)
    assert oracle.target(t).canonical == {"red_5": "left_bin", "blue_2": TABLE, "red_9": "left_bin", "green_4": TABLE}
    assert oracle.target(t, corr({"op": "set_bin", "bin": "right_bin"})).canonical["red_9"] == "right_bin"


def test_everything_except_the_green_one_any_order():
    t = Task("t", "all_except", "", {"except": "green"}, SC)
    tg = oracle.target(t)
    assert tg.canonical["green_4"] == TABLE
    assert tg.check({"red_5": "tray_slot_4", "blue_2": "tray_slot_2", "red_9": "tray_slot_1", "green_4": TABLE})
    assert not tg.check({"red_5": "tray_slot_4", "blue_2": "tray_slot_2", "red_9": "tray_slot_1", "green_4": "tray_slot_3"})
    assert oracle.target(t, corr({"op": "include_all"})).canonical["green_4"].startswith("tray_slot_")


def test_alternate_colours_start_and_flip():
    sc = scene(("red", 1, 2.0), ("blue", 2, 3.0), ("red", 3, 4.0), ("blue", 4, 5.0))
    t = Task("t", "alternate", "", {"first": "red", "second": "blue"}, sc)
    tg = oracle.target(t)
    assert tg.check({"red_3": "tray_slot_1", "blue_4": "tray_slot_2", "red_1": "tray_slot_3", "blue_2": "tray_slot_4"})  # any order within a colour
    assert not tg.check({"blue_2": "tray_slot_1", "red_1": "tray_slot_2", "blue_4": "tray_slot_3", "red_3": "tray_slot_4"})
    assert oracle.target(t, corr({"op": "flip"})).canonical["blue_2"] == "tray_slot_1"


def test_parity_bins_and_swap():
    t = Task("t", "parity_bins", "", {"even_bin": "left_bin"}, SC)
    assert oracle.target(t).canonical == {"red_5": "right_bin", "blue_2": "left_bin", "red_9": "right_bin", "green_4": "left_bin"}
    assert oracle.target(t, corr({"op": "swap_bins"})).canonical["blue_2"] == "right_bin"


@pytest.mark.parametrize("seed", range(5))
def test_generated_tasks_are_consistent(seed):
    """Every correction has a well-defined target; coverable ones are exactly the natural parameter's
    other branch, and uncoverable ones are not (under the task's natural parameter)."""
    coincide = {True: 0, False: 0}
    for t in make_tasks(21, 5, seed):
        assert 4 <= len(t.scene.blocks) <= 8 and len(set(t.scene.ids())) == len(t.scene.blocks)
        plan = json.loads(MockLLM(seed).call(PL.plan_request(t.text, scene_view(t.scene)), ref=({"task": t}, 0)).raw)
        assert P.validate(plan, scene_view(t.scene)) == []
        goals = list(P.branch_goals(plan).values())
        assert oracle.target(t).check(goals[0])
        for c in t.corrections:
            tg = oracle.target(t, c)
            if c.category == "coverable":
                assert tg.check(goals[1]) and not tg.check(goals[0]), (t.text, c.text)
            elif c.category in ("chatter", "pause", "restate"):
                assert tg.check(goals[0]), (t.text, c.text)
            else:   # switching the natural parameter never fixes it ("by number" can coincide with the current goal)
                assert tg.check(goals[0]) or not tg.check(goals[1]), (t.text, c.text)
                coincide[tg.check(goals[0])] += 1
    assert coincide[True] <= 0.1 * sum(coincide.values()), coincide


# ---------------------------------------------------------------- generic plan validator

SV = scene_view(SC)


def good_plan():
    g1 = [{"block": b, "destination": f"tray_slot_{i}"} for i, b in enumerate(["blue_2", "green_4", "red_5", "red_9"], 1)]
    g2 = [{"block": b, "destination": f"tray_slot_{i}"} for i, b in enumerate(["red_9", "red_5", "green_4", "blue_2"], 1)]
    return {"task_reading": "line up", "parameters": [{"name": "dir", "about": "which end starts", "default": "low_left",
                                                       "values": [{"value": "low_left", "meaning": "a"}, {"value": "high_left", "meaning": "b"}]}],
            "branches": [{"settings": [{"parameter": "dir", "value": "low_left"}], "goal": g1},
                         {"settings": [{"parameter": "dir", "value": "high_left"}], "goal": g2}]}


def test_validator_accepts_a_good_plan():
    assert P.validate(good_plan(), SV) == []
    assert P.branch_keys(good_plan()) == ["dir=low_left", "dir=high_left"]


@pytest.mark.parametrize("breaker, needle", [
    (lambda p: p["branches"][0]["goal"].pop(), "no destination for"),
    (lambda p: p["branches"][0]["goal"][1].update(destination="tray_slot_1"), "more than one block in tray_slot_1"),
    (lambda p: p["branches"][0]["goal"][1].update(destination="shelf"), "unknown destinations"),
    (lambda p: p["branches"][0]["goal"].append({"block": "red_5", "destination": "left_bin"}), "more than one destination for red_5"),
    (lambda p: p["parameters"][0].update(default="middle"), "is not one of its values"),
    (lambda p: p["branches"].pop(0), "no branch has every parameter at its default"),
    (lambda p: p["branches"][1]["settings"][0].update(value="low_left"), "same settings"),
    (lambda p: p["branches"][1]["settings"][0].update(value="sideways"), "is not a value of"),
    (lambda p: p["branches"][1].update(settings=[]), "exactly once"),
    (lambda p: p["branches"].extend([p["branches"][0]] * 4), "branches; needs 1-4"),
])
def test_validator_rejects(breaker, needle):
    p = good_plan()
    breaker(p)
    errs = P.validate(p, SV)
    assert any(needle in e for e in errs), errs


def test_bins_hold_many_blocks():
    p = good_plan()
    for g in p["branches"][0]["goal"]:
        g["destination"] = "left_bin"
    assert P.validate(p, SV) == []


def test_schema_is_strict_everywhere():
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    schema = P.plan_schema(SV)
    walk(schema)
    kw = PL.openai_kwargs(PL.plan_request("t", SV), "some-model")
    assert kw["text"]["format"]["type"] == "json_schema" and kw["text"]["format"]["strict"] is True
    assert "temperature" not in kw and kw["store"] is False


class Scripted:
    def __init__(self, outputs):
        self.outputs, self.reqs = list(outputs), []

    def call(self, req, ref=None):
        self.reqs.append(req)
        return PL.LLMResult(self.outputs.pop(0), 10.0, 1, 1)


def test_invalid_plan_gets_one_retry_with_the_errors():
    bad = good_plan()
    bad["branches"][0]["goal"].pop()
    be = Scripted([json.dumps(bad), json.dumps(good_plan())])
    out = PL.get_plan(be, PL.plan_request("t", SV), SV)
    assert out.plan and not out.valid_first and out.valid_after_retry and len(out.attempts) == 2
    assert "no destination for red_9" in be.reqs[1].input
    out = PL.get_plan(Scripted([json.dumps(bad), "not json"]), PL.plan_request("t", SV), SV)
    assert out.plan is None and len(out.attempts) == 2


# ---------------------------------------------------------------- router decision

def answers(route="adjust", rconf=0.95, value="high_left", vconf=0.9, stated=0.97):
    return {"route": {"choice": route, "confidence": rconf},
            "p0.value": {"choice": value, "confidence": vconf},
            "p0.stated": {"noul": stated}}


CUR = {"dir": "low_left"}


def test_adjust_picks_the_prepared_branch():
    d = R.decide(good_plan(), CUR, answers(), 0.7)
    assert (d.action, d.key, d.reason) == ("branch", "dir=high_left", "adjust")


def test_missing_branch_escalates():
    p = good_plan()
    p["branches"].pop()
    d = R.decide(p, CUR, answers(), 0.7)
    assert (d.action, d.reason) == ("escalate", "missing_branch")


def test_low_confidence_escalates_on_the_weakest_question():
    assert R.decide(good_plan(), CUR, answers(vconf=0.5), 0.7).reason == "low_confidence"
    assert R.decide(good_plan(), CUR, answers(stated=0.7), 0.7).reason == "low_confidence"   # noul conf |2p-1| = 0.4
    assert R.decide(good_plan(), CUR, answers(rconf=0.6), 0.7).reason == "low_confidence"
    assert R.decide(good_plan(), CUR, answers(vconf=0.5), 0.4).action == "branch"


def test_unstated_parameter_keeps_its_value_and_other_routes():
    assert R.decide(good_plan(), CUR, answers("continue", stated=0.03), 0.7).action == "keep"
    assert R.decide(good_plan(), CUR, answers("adjust", stated=0.03), 0.7).reason == "adjust_without_change"
    assert R.decide(good_plan(), CUR, answers("new_plan_needed"), 0.7).reason == "route_new_plan"
    assert R.decide(good_plan(), CUR, answers("pause", rconf=0.4), 0.7).action == "keep"      # pause has its own low gate
    assert R.decide(good_plan(), CUR, answers("continue", stated=0.97), 0.7).reason == "continue_but_value_changed"
    assert R.decide(good_plan(), CUR, None, 0.7).reason == "jev_error"


def test_jev_request_has_no_goals_and_no_unchanged_option():
    qs = R.jev_questions(good_plan())
    st = R.jev_state("task", "actually, the other way", good_plan(), CUR)
    assert "goal" not in json.dumps(st) and "tray_slot" not in json.dumps(st)
    assert set(qs) == {"route", "p0.value", "p0.stated"}
    assert set(qs["route"]["criteria"]) == set(R.ROUTES)
    assert all("not_for" in c for c in qs["route"]["criteria"].values())
    assert set(qs["p0.value"]["criteria"]) == {"low_left", "high_left"} and qs["p0.stated"]["type"] == "noul"


# ---------------------------------------------------------------- full offline run

def test_mock_run_completes_and_scores(capsys):
    cli.main(["--llm", "mock", "--jev", "mock", "--tasks", "14", "--corrections", "5", "--seed", "3"])
    out = capsys.readouterr().out
    for s in ("planner (step 1)", "Jev router (step 2)", "gate sweep", "end to end per correction", "always-LLM right"):
        assert s in out
    assert "all            70" in out


def test_perfect_mocks_are_right_end_to_end():
    import argparse
    args = argparse.Namespace(llm="mock", jev="mock", llm_model=None, llm_timeout=1, llm_max_output=None, llm_effort=None,
                              gate=0.7, baseline=True, no_retry=False, max_llm_calls=None, mock_llm_error=0.0,
                              mock_jev_error=0.0, log=False, seed=1)
    plans, recs, info = cli.run(make_tasks(14, 5, 1), args)
    assert all(p["valid_first"] and p["default_ok"] for p in plans)
    outs = [S.outcome(r, 0.7) for r in recs]
    assert all(o["correct"] for o in outs)
    assert all(o["path"] == "jev" for r, o in zip(recs, outs) if r["category"] in ("coverable", "chatter", "pause"))
    assert all(o["path"] == "escalated" for r, o in zip(recs, outs) if r["truth"]["route"] == ["new_plan_needed"])
    assert info["llm_calls"] == 14 + 70


def test_max_llm_calls_stops_cleanly(capsys):
    cli.main(["--tasks", "7", "--max-llm-calls", "10"])
    assert "STOPPED: --max-llm-calls 10" in capsys.readouterr().out


def test_log_and_replay(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    cli.main(["--tasks", "7", "--corrections", "3", "--log"])
    first = capsys.readouterr().out
    log = tmp_path / "e13_branches_mock_mock.jsonl"
    cli.main(["--replay", str(log)])
    again = capsys.readouterr().out
    cut = lambda s: [ln for ln in s[s.index("route accuracy"):s.index("system accuracy by task family")].splitlines() if "====" not in ln]  # noqa: E731
    assert cut(first) == cut(again)


def test_dry_run_writes_samples(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(["--dry-run", "--tasks", "7"])
    out = capsys.readouterr().out
    assert "Jev: 35 requests" in out and "no model price is assumed" in out
    kw = json.loads((tmp_path / "e13_samples" / "llm_plan_t00_sort_number.json").read_text())["openai_kwargs"]
    assert kw["model"] == "<OPENAI_MODEL>"


def test_openai_without_a_model_lists_and_exits(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(PL, "list_models", lambda s="terra": ["terra-a", "terra-b"])
    with pytest.raises(SystemExit, match="--llm-model"):
        cli.main(["--llm", "openai", "--tasks", "1"])


# ---------------------------------------------------------------- the system under test stays generic

SUT = ("plan.py", "planner.py", "router.py")


@pytest.mark.parametrize("name", SUT)
def test_system_under_test_does_not_import_the_evaluation_side(name):
    tree = ast.parse((EXP / "e13_branches" / name).read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
    assert not mods & {"oracle", "world", "mocks", "score", "e13_branches.oracle", "e13_branches.world"}, mods


@pytest.mark.parametrize("name", SUT)
def test_system_under_test_has_no_task_specific_logic(name):
    src = (EXP / "e13_branches" / name).read_text().lower()
    for word in ("\"number\"", "ascending", "descending", "reverse", "size_cm", "colour", "color", "parity", "biggest", "lowest"):
        assert word not in src, (name, word)


def test_generator_is_seeded():
    a = [(t.text, [c.text for c in t.corrections]) for t in make_tasks(14, 5, 7)]
    b = [(t.text, [c.text for c in t.corrections]) for t in make_tasks(14, 5, 7)]
    assert a == b and len({t for t, _ in a}) > 5
    assert {make_task(random.Random(0), f, "x").family for f in FAMILIES} == set(FAMILIES)


def test_openai_backend_reads_a_responses_api_result():
    from types import SimpleNamespace as NS
    sent = {}

    class Responses:
        def create(self, **kw):
            sent.update(kw)
            return NS(status="completed", output_text=json.dumps(good_plan()), model="m-1",
                      usage=NS(input_tokens=120, output_tokens=80), incomplete_details=None)

    be = PL.OpenAIBackend.__new__(PL.OpenAIBackend)
    be.model, be.max_output_tokens, be.effort = "m", None, "low"
    be.client = NS(responses=Responses())
    out = PL.get_plan(be, PL.plan_request("t", SV), SV)
    assert out.plan == good_plan() and out.in_tok == 120 and out.out_tok == 80
    assert sent["reasoning"] == {"effort": "low"} and sent["text"]["format"]["strict"] is True
