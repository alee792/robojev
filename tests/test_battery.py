"""The spelling battery (robojev.battery, docs/battery.md): the generator, the judge, the offline
planner and decider stand-ins, the controls, the runner on the stub and on the physics sim, the
two-arm stub pair, the CLI, the paraphrase hook.

Everything is judged on outcomes (the row the arm leaves, times, calls), never on which decision was
made; the tests here check the machinery and that the deciders finish, not that any one of them was
"right" at an event.

Live models (tests/live_models.py): ROBOJEV_DECIDER=jev|decisions and/or ROBOJEV_PLANNER=openai run the
physics episodes with a live decider or planner on held-out phrasings, writing the episode files to
experiments/results/live/. Offline (the default) nothing needs keys.
"""
from __future__ import annotations

import dataclasses
import json
import re

import httpx
import pytest

from robojev.battery import cli, paraphrase
from robojev.battery.controls import AlwaysLLM, RulesOnly
from robojev.battery.decider import SpellTruth, truth
from robojev.battery.generate import EVENT_KINDS, Episode, Event, Generator, Grammar, When, episodes, first_with, spellable
from robojev.battery.judge import read_row, score_row
from robojev.battery.phrasings import OPENING, PHRASINGS, SPLITS, check_tables
from robojev.battery.run import CONTROLS, Options, compare, run, run_episode, table
from robojev.battery.spell import Cube, Table, asked_word, spell_plan, word_in
from robojev.brain.core.data import Event as CoreEvent
from robojev.brain.core.decision import View
from robojev.protocol import ArmObs, ObjectObs, PlaceObs, WorldState
from robojev.robots.widowx_sim import scene

import live_models as LM

POSITIONS = [f"slot_{i}" for i in range(6)]
FORBIDDEN = re.compile(r"\b(expect(ed)?_action|right_now|route|fix|should|oracle|correct_answer)\b", re.I)


# ---------------------------------------------------------------- the generator


def test_phrasing_tables_have_four_per_half_and_the_halves_are_disjoint():
    assert check_tables() == []
    for kind, halves in PHRASINGS.items():
        for split in SPLITS:
            assert len(halves[split]) >= 4, (kind, split)
        assert not set(halves["train"]) & set(halves["held_out"]), kind


def test_generator_is_deterministic_and_seeds_differ():
    assert episodes(8, 0) == episodes(8, 0)
    assert episodes(8, 0) != episodes(8, 1)
    assert episodes(8, 0, "held_out") != episodes(8, 0, "train")
    eps = episodes(40, 0)
    kinds = {e.kind for ep in eps for e in ep.events}
    assert kinds >= set(EVENT_KINDS) - {"stop"}, kinds                 # every event type turns up in 40 episodes
    assert all(0 <= len([e for e in ep.events if e.kind != "stop"]) <= 3 for ep in eps)


def _phrasing_of(text: str, kind: str, split: str, word: str | None) -> bool:
    table = OPENING if kind == "opening" else PHRASINGS[kind]
    return any(text == t.replace("{W}", word or "") for t in table[split])


def test_every_line_comes_from_its_split_and_the_end_state_is_the_latest_word():
    for split in SPLITS:
        for ep in episodes(30, 2, split):
            assert ep.split == split
            assert _phrasing_of(ep.opening, "opening", split, ep.word)
            assert _phrasing_of(ep.reply, "reply", split, None)
            latest = ep.word
            for e in ep.events:
                if e.kind == "new_word":
                    assert _phrasing_of(e.text, "new_word", split, e.word) and e.word != latest
                    latest = e.word
                elif e.kind == "chatter":
                    assert _phrasing_of(e.text, "chatter", split, None)
                elif e.kind == "wait_go_on":
                    assert _phrasing_of(e.text, "wait", split, None) and _phrasing_of(e.go_on_text, "go_on", split, None)
                elif e.text is not None:
                    assert _phrasing_of(e.text, e.kind, split, None), (e.kind, e.text)
                if e.return_text:
                    assert _phrasing_of(e.return_text, "letter_returned", split, None)
            assert ep.expected_word == latest
            assert spellable(ep.expected_word, ep.letters())            # the end state is always reachable
            assert [e.kind for e in ep.events].count("stop") <= 1
            assert all(e.kind != "stop" for e in ep.events[:-1])


def test_an_episode_carries_no_expected_action():
    """Only what happens, when, and the end state: no field says what the robot should do."""
    for cls in (Episode, Event, When):
        for f in dataclasses.fields(cls):
            assert not FORBIDDEN.search(f.name), (cls.__name__, f.name)
    ep = episodes(1, 0)[0]
    dumped = json.dumps(ep.as_dict())
    assert not FORBIDDEN.search(dumped.replace("letter_returned", ""))


def test_grammar_refuses_words_its_letters_cannot_spell():
    with pytest.raises(ValueError):
        Grammar(words=("ZEBRA",))
    assert spellable("PANDA", "OPENAAID") and not spellable("PANDA", "OPENAID")


def test_first_with_finds_shapes_without_listing_them():
    ep = first_with(["new_word", "letter_taken"], seed=1)
    assert {"new_word", "letter_taken"} <= {e.kind for e in ep.events}


def test_paraphrase_hook_is_a_no_op_offline_and_keeps_the_word_when_live():
    assert paraphrase.none("spell OPENAI", "opening") == "spell OPENAI"
    ep = Generator().episode(0, 4, "held_out", paraphrase.none)
    assert ep == Generator().episode(0, 4, "held_out")

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        handler.sent.append(body)
        text = body["input"]
        new = "could you make that PANDA" if "PANDA" in text else "ooh, do take care" if "hand" in body["instructions"] else "LOST THE WORD"
        return httpx.Response(200, json={"output": [{"type": "message", "content": [{"type": "output_text", "text": new}]}]})
    handler.sent = []
    para = paraphrase.OpenAIParaphraser(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert para("actually, spell PANDA", "new_word") == "could you make that PANDA"
    assert handler.sent[0]["model"] and "PANDA" in handler.sent[0]["input"]
    e = Event(0, "new_word", When("at", 1.0, 9.0), text="actually, spell PINE", word="PINE")
    ep = Episode(0, 0, "held_out", "spell OPENAI", "OPENAI", "OPENAAID", "", 6, (e,), "whatever you can manage", "PINE")
    from robojev.battery.generate import _paraphrased
    out = _paraphrased(ep, para)
    assert out.events[0].text == "actually, spell PINE"               # the rewrite lost the word: the original stands
    assert out.lines["actually, spell PINE"].kind == "new_word"


# ---------------------------------------------------------------- the judge, on synthetic worlds


def _world(row: dict[int, str], holding: str | None = None, extra_table: tuple[str, ...] = ()) -> WorldState:
    places = {p: PlaceObs(p, f"position {i}", "slot", 0.36, 0.2 - 0.08 * i, 1) for i, p in enumerate(POSITIONS)}
    objs = {}
    for i, lab in row.items():
        oid = f"letter_{lab}{i}"
        objs[oid] = ObjectObs(oid, f"the letter {lab}", 0.36, 0.2 - 0.08 * i, 0.0, POSITIONS[i], label=lab)
    for k, lab in enumerate(extra_table):
        oid = f"letter_{lab}_t" if k == 0 or lab not in extra_table[:k] else f"letter_{lab}_t{k}"
        objs[oid] = ObjectObs(oid, f"the letter {lab}", 0.2, 0.0, 0.0, "table", label=lab)
    arms = {"arm_0": ArmObs("arm_0", 0.2, 0.0, 0.1, holding, 0.08, "idle")}
    return WorldState(0.0, objs, places, [], arms)


@pytest.mark.parametrize("row, word, want", [
    ({0: "O", 1: "P", 2: "E", 3: "N", 4: "A", 5: "I"}, "OPENAI", (6, 6, 1.0)),       # the full word
    ({0: "O", 1: "P", 2: "E"}, "OPENAI", (3, 6, 0.5)),                               # half spelled
    ({0: "P", 1: "O", 2: "E", 3: "N", 4: "A", 5: "I"}, "OPENAI", (0, 6, 0.0)),       # wrong order: nothing from the left
    ({0: "O", 1: "P", 2: "E", 4: "A", 5: "I"}, "OPENAI", (3, 6, 0.5)),               # a letter missing in the middle
    ({0: "P", 1: "A", 2: "N", 3: "D", 4: "A", 5: "I"}, "PANDA", (5, 6, 5 / 6)),      # the word, then a stray cube
    ({}, "OPENAI", (0, 6, 0.0)),                                                      # nothing placed
    ({0: "P", 1: "A", 2: "N", 3: "D", 4: "A"}, "PANDA", (5, 5, 1.0)),
])
def test_judge_scores_the_row_left_to_right(row, word, want):
    w = _world(row, extra_table=("E",))
    r = read_row(w, POSITIONS)
    assert r == [row.get(i) for i in range(6)]
    prefix, span, score = score_row(r, word)
    assert (prefix, span) == want[:2] and score == pytest.approx(want[2])


# ---------------------------------------------------------------- the offline planner


def _table(**where) -> Table:
    cubes = [Cube("letter_O", "O", where.get("O", "table")), Cube("letter_P", "P", where.get("P", "table")),
             Cube("letter_E", "E", where.get("E", "table")), Cube("letter_N", "N", where.get("N", "table")),
             Cube("letter_A", "A", where.get("A", "table")), Cube("letter_A2", "A", where.get("A2", "table")),
             Cube("letter_I", "I", where.get("I", "table")), Cube("letter_D", "D", where.get("D", "table"))]
    held = next((c.id for c in cubes if c.where == "gripper"), None)
    return Table(cubes, POSITIONS, "table", {"arm": held})


def test_spell_plan_goes_left_to_right_reuses_what_is_placed_and_clears_what_is_wrong():
    plan, missing = spell_plan("OPENAI", _table())
    assert missing == [] and [(s["object"], s["target"]) for s in plan["steps"]] == [
        ("letter_O", "slot_0"), ("letter_P", "slot_1"), ("letter_E", "slot_2"), ("letter_N", "slot_3"), ("letter_A", "slot_4"), ("letter_I", "slot_5")]
    assert [c["target"] for c in plan["done_when"]] == POSITIONS
    # O and P placed, then PANDA: O is cleared to the table, P moves straight to position 0, the rest follow
    plan, missing = spell_plan("PANDA", _table(O="slot_0", P="slot_1"))
    assert [(s["object"], s["target"]) for s in plan["steps"]] == [
        ("letter_O", "table"), ("letter_P", "slot_0"), ("letter_A", "slot_1"), ("letter_N", "slot_2"), ("letter_D", "slot_3"), ("letter_A2", "slot_4")]
    # already right: nothing to do but the conditions
    plan, _ = spell_plan("OPEN", _table(O="slot_0", P="slot_1", E="slot_2", N="slot_3"))
    assert plan["steps"] == [] and len(plan["done_when"]) == 4
    # a stray cube past the word's end is cleared
    plan, _ = spell_plan("PEN", _table(P="slot_0", E="slot_1", N="slot_2", I="slot_3"))
    assert [(s["object"], s["target"]) for s in plan["steps"]] == [("letter_I", "table")]


def test_spell_plan_puts_a_held_cube_down_first_and_names_missing_letters():
    plan, missing = spell_plan("PANDA", _table(O="gripper"))
    assert plan["steps"][0] == {"id": "s1", "skill": "move_object", "object": "letter_O", "target": "table", "direction": "none"}
    plan, missing = spell_plan("OPEN", _table(P="gripper"))
    assert (plan["steps"][0]["object"], plan["steps"][0]["target"]) == ("letter_P", "slot_1")
    plan, missing = spell_plan("PIZZA", _table())
    assert missing == ["Z", "Z"] and "no cube in view for the letter" in plan["reading"]
    assert [(s["object"], s["target"]) for s in plan["steps"]] == [("letter_P", "slot_0"), ("letter_I", "slot_1"), ("letter_A", "slot_4")]


def test_the_word_is_read_from_the_newest_message_that_names_one():
    assert word_in("actually, spell PANDA") == "PANDA" and word_in("nice day for it") is None and word_in("I like it") is None
    assert asked_word("spell OPENAI", ["hang on a second", "scrap that, spell PANDA instead", "ok, go on"]) == "PANDA"
    assert asked_word("spell OPENAI", []) == "OPENAI"


# ---------------------------------------------------------------- the controls and the mock decider


def test_rules_were_written_from_the_train_phrasings():
    """Every train phrasing lands on the rule its type was written for (held-out ones may not: that
    is the point of the split)."""
    r = RulesOnly()

    def decide(text: str):
        c = r.decide(CoreEvent(1, 0, "user_text", f'the user typed: "{text}"', data={"text": text}), None)["combined"]
        return c.right_now, c.route
    for t in PHRASINGS["new_word"]["train"]:
        assert decide(t.replace("{W}", "PANDA")) == ("hold", "fast_llm"), t
    for t in PHRASINGS["wait"]["train"]:
        assert decide(t) == ("pause", "stay_local"), t
    for t in PHRASINGS["go_on"]["train"]:
        assert decide(t) == ("resume", "stay_local"), t
    for kind in ("chatter", "letter_taken", "letter_returned", "letter_added", "letter_nudged", "letter_placed", "hand_near"):
        for t in PHRASINGS[kind]["train"]:
            assert decide(t) == ("carry_on", "stay_local"), (kind, t)
    for t in PHRASINGS["reply"]["train"]:
        assert decide(t) == ("carry_on", "fast_llm"), t


def test_always_llm_sends_every_non_routine_event_to_the_planner():
    a = AlwaysLLM()
    c = a.decide(CoreEvent(1, 0, "user_text", "x", data={"text": "hang on"}), None)["combined"]
    assert (c.right_now, c.route) == ("carry_on", "fast_llm") and c.react
    for routine in ("step_done", "plan_arrived"):
        c = a.decide(CoreEvent(2, 0, routine, "x"), None)["combined"]
        assert c.route == "stay_local"


def _view(world: WorldState, plan: dict, status: dict, queue: list, current=None, messages=(), mode="running") -> View:
    from robojev.brain.adapt import core_state, geometry
    from robojev.protocol import ArmSpec, GripperSpec
    spec = ArmSpec("arm_0", "base", ((0.1, 0.5), (-0.3, 0.3), (-0.02, 0.35)), GripperSpec(0.08), 0.12)
    st = core_state(world, "arm_0", geometry(spec, 0.2), 0.2)
    return View(0, "spell OPENAI", list(messages), st, plan, queue, status, current, "between steps", mode, None, False,
                {o.id: o.name for o in world.objects.values()}, None, None, None, user_words=["spell OPENAI", *messages])


def test_mock_decider_routes_a_taken_letter_to_the_user_only_when_no_spare_carries_it():
    ep = Episode(0, 0, "train", "spell OPENAI", "OPENAI", "OPENAAID", "", 6, (), "whatever you can manage", "OPENAI")
    step = {"id": "s4", "skill": "move_object", "object": "letter_N", "target": "slot_3", "direction": "none"}
    plan = {"reading": "spell OPENAI", "steps": [step], "done_when": [], "constraints": []}
    ev = CoreEvent(5, 0, "scene_change", "the letter N is no longer in view", "letter_N",
                   data={"change": {"what": "object_gone", "object": "letter_N", "from": "table"}})
    world = _world({0: "O", 1: "P", 2: "E"}, extra_table=("A", "I"))          # N gone, no spare
    T = truth(ev, _view(world, plan, {"s4": "pending"}, ["s4"]), SpellTruth(ep, lambda: world))
    assert T["route"][0] == "ask_user"
    world = _world({0: "O", 1: "P", 2: "E"}, extra_table=("A", "I", "N"))     # another N on the table
    T = truth(ev, _view(world, plan, {"s4": "pending"}, ["s4"]), SpellTruth(ep, lambda: world))
    assert T["route"][0] == "fast_llm"
    # a new word: hold when the step in progress no longer fits it, carry on when it still does
    text = "actually, spell PANDA"
    ep2 = Episode(0, 0, "train", "spell OPENAI", "OPENAI", "OPENAAID", "", 6,
                  (Event(0, "new_word", When("at", 1.0, 9.0), text=text, word="PANDA"),), "whatever you can manage", "PANDA")
    ev = CoreEvent(6, 0, "user_text", f'the user typed: "{text}"', data={"text": text})
    world = _world({0: "O", 1: "P"}, extra_table=("E", "N", "A", "A", "I", "D"))
    T = truth(ev, _view(world, plan, {"s4": "running"}, ["s4"], current=step, messages=[text]), SpellTruth(ep2, lambda: world))
    assert T["right_now"][0] == "hold" and T["route"][0] == "fast_llm"       # N to position 3: PANDA wants D there
    step_a = {"id": "s5", "skill": "move_object", "object": "letter_A_t", "target": "slot_1", "direction": "none"}
    plan_a = {"reading": "spell OPENAI", "steps": [step_a], "done_when": [], "constraints": []}
    T = truth(ev, _view(world, plan_a, {"s5": "running"}, ["s5"], current=step_a, messages=[text]), SpellTruth(ep2, lambda: world))
    assert T["right_now"][0] == "carry_on"                                    # A to position 1 fits PANDA too


# ---------------------------------------------------------------- episodes on the stub


def test_the_offline_planner_spells_the_word_on_the_stub():
    ep = Episode(0, 0, "train", "spell OPENAI", "OPENAI", "OPENAAID", "", 6, (), "whatever you can manage", "OPENAI")
    rec, dump = run_episode(ep, Options(world="stub", quiet=True, pick_s=0.4))
    assert rec.outcome == "done" and rec.exact and rec.row == list("OPENAI"), rec
    assert rec.planner_calls == 1 and rec.steps_done == 6 and rec.drops == 0
    assert dump["judge"]["score"] == 1.0 and len(dump["brain_log"]) > 10


def test_six_episode_compare_on_the_stub_all_three_deciders_finish_most(tmp_path):
    opts = Options(n=6, seed=0, world="stub", out=tmp_path, quiet=True, pick_s=0.5)
    eps = episodes(6, 0)
    summaries = compare(opts, CONTROLS, eps)
    assert [s.decider for s in summaries] == list(CONTROLS)
    for s in summaries:
        done = sum(r.outcome == "done" for r in s.records)
        assert done >= 4, (s.decider, [(r.index, r.outcome, r.row) for r in s.records])
        assert s.mean_score >= 0.6, (s.decider, s.mean_score)
        assert s.drops == 0.0
        assert (tmp_path / f"summary_{s.decider}.json").exists()
    files = sorted(tmp_path.glob("ep_*.jsonl"))
    assert len(files) == 18
    lines = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert {x["type"] for x in lines} >= {"episode", "judge", "brain_log", "decisions", "llm"}
    judge = next(x for x in lines if x["type"] == "judge")
    assert not FORBIDDEN.search(" ".join(judge))                 # the record's fields grade outcomes, not answers
    text = table(summaries)
    assert "mean outcome score" in text and "always_llm + mock" in text
    print("\n" + text)


def test_a_generated_episode_on_the_two_arm_stub_pair_crosses_the_zone():
    g = Grammar(arms=2)
    ep = episodes(1, 0, grammar=g)[0]
    assert ep.arms == 2
    rec, dump = run_episode(ep, Options(world="stub", arms=2, quiet=True, grammar=g, pick_s=0.5))
    assert rec.outcome == "done" and rec.score == 1.0, (rec.row, rec.outcome, rec.notes)
    starts = [x for x in dump["brain_log"] if x.get("note") == "step_start"]
    assert {x["arm"] for x in starts} == {"arm_0", "arm_1"}       # letters in arm_0's half, positions in arm_1's: both arms work
    assert any("zone" in x["step"] for x in starts)


def test_cli_runs_a_tiny_battery_and_writes_files(tmp_path, capsys):
    assert cli.main(["--n", "1", "--seed", "3", "--world", "stub", "--out", str(tmp_path), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "mean outcome score" in out and "train phrasings" in out
    assert (tmp_path / "ep_3_000_mock.jsonl").exists() and (tmp_path / "summary_mock.json").exists()
    a = cli.parse(["--decider", "jev", "--planner", "openai", "--paraphrase"])
    assert a.decider == "jev" and a.paraphrase
    assert LM.battery_split() in SPLITS


def test_options_refuse_two_arms_on_the_sim():
    with pytest.raises(ValueError):
        Options(world="sim", arms=2)


# ---------------------------------------------------------------- the physics sim

needs_sim = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(), reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")


@needs_sim
@pytest.mark.parametrize("kinds", [["letter_taken"], ["new_word"]])
def test_generated_episodes_on_the_physics_sim(kinds, request):
    """One episode with a letter taken away and one with a new-word correction (the first generated
    ones of seed 1 with those events), on the MuJoCo WidowX, judged 0.9 or better. Live
    (tests/live_models.py) the same episodes run with the live decider or planner on held-out phrasings."""
    pytest.importorskip("mujoco")
    decider, planner = LM.battery_models()
    split = LM.battery_split()
    ep = first_with(kinds, seed=1, split=split)
    out = LM.battery_out(request.node.name.replace("[", "_").replace("]", ""))
    opts = Options(n=1, seed=1, world="sim", decider=decider, planner=planner, split=split, out=out, quiet=True)
    s = run(opts, [ep])
    r = s.records[0]
    print(f"\nBATTERY sim [{decider}+{planner}, {split}] ep {ep.index} {[e.kind for e in ep.events]}: word {r.word} row {r.row} "
          f"score {r.score} {r.outcome} {r.wall_s} s; plans {r.planner_calls}; drops {r.drops}; "
          f"events {[(e.kind, e.what, e.time_to_change_s, e.change) for e in r.events]}; notes {r.notes}")
    assert {e.kind for e in ep.events} >= set(kinds)
    assert r.outcome == "done", r
    assert r.score >= 0.9, (r.row, r.word)
    assert r.drops == 0
    assert all(e.what != "skipped" for e in r.events), r.notes
