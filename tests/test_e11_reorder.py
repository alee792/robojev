"""E11 probe (experiments/e11_reorder.py): the ground-truth solver on hand-worked cases, and that every
request it builds is well formed. Offline; the probe itself needs a Jev key."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e11_reorder as e  # noqa: E402


def scene(row, held=None, lifted=None, old=e.ASC, phrase="actually, reverse it", values=None):
    text, kind, intent = next(p for p in e.PHRASES if p[0] == phrase)
    values = values or sorted([b for b in row if b is not None] + ([held] if held else []))
    return e.Scene("t", len(row), values, row, held, lifted, old, text, kind, intent, values == list(range(1, len(row) + 1)))


# -- canonical: [3,1,2,5,4], sorting ascending, holding 5 lifted from slot 4, "actually, reverse it" --

def test_canonical_reverse_carries_5_left_to_slot_1():
    t = e.truth(e.canonical())
    assert t["new_order"] == {e.DESC}
    assert t["intent"] == {"adjust_order"}
    assert t["direction"] == {e.LEFT}
    assert t["target_slot"] == {"slot 1"}
    assert "next_pick" not in t


def test_canonical_under_the_old_order_would_carry_right_to_slot_5():
    sc = e.canonical()
    assert e.home_slot(5, sc.values, e.ASC) == 5
    assert e.direction(5, 4, sc.values, e.ASC) == e.RIGHT


def test_keep_going_keeps_old_order_and_accepts_unchanged():
    t = e.truth(scene([3, 1, 2, None, 4], held=5, lifted=4, phrase="keep going the same way"))
    assert t["new_order"] == {e.ASC, "unchanged"}
    assert t["intent"] == {"continue"}
    assert t["direction"] == {e.RIGHT} and t["target_slot"] == {"slot 5"}


def test_explicit_order_equal_to_old_accepts_adjust_or_continue():
    t = e.truth(scene([4, 5, 3, 2, 1], old=e.DESC, phrase="biggest first"))
    assert t["new_order"] == {e.DESC, "unchanged"}
    assert t["intent"] == {"adjust_order", "continue"}


def test_put_back_when_lifted_from_its_own_slot():
    # descending over {2, 5, 9}: 9 belongs in slot 1, and that is where it came from
    t = e.truth(scene([None, 2, 5], held=9, lifted=1, old=e.ASC, phrase="sort them high to low"))
    assert t["direction"] == {e.BACK} and t["target_slot"] == {"slot 1"}


def test_sparse_numbers_use_rank_not_value():
    # blocks {4, 11, 17, 20}; 17 is the 3rd smallest -> slot 3 ascending, slot 2 descending
    sc = scene([20, None, 4, 11], held=17, lifted=2, old=e.DESC, phrase="sort them low to high")
    t = e.truth(sc)
    assert t["target_slot"] == {"slot 3"} and t["direction"] == {e.RIGHT}
    assert e.home_slot(17, sc.values, e.DESC) == 2


# -- next_pick: the block that belongs in the leftmost wrong slot --

def test_next_pick_leftmost_wrong_slot():
    row = [1, 2, 5, 3, 4]
    assert e.next_pick(row, [1, 2, 3, 4, 5], e.ASC) == 3      # slots 1-2 right, slot 3 wants 3
    assert e.next_pick(row, [1, 2, 3, 4, 5], e.DESC) == 5     # slot 1 wants 5
    t = e.truth(scene(row, old=e.ASC, phrase="wait, do it the other way round"))
    assert t["next_pick"] == {"block 5"} and "direction" not in t


def test_next_pick_nothing_when_already_sorted():
    t = e.truth(scene([1, 2, 3], old=e.ASC, phrase="go a bit slower"))
    assert t["next_pick"] == {e.NOTHING}
    assert e.truth(scene([1, 2, 3], old=e.ASC, phrase="actually, reverse it"))["next_pick"] == {"block 3"}


def test_stop_pause_new_task_skip_action_questions():
    for phrase in ("stop!", "hold on a sec", "forget the blocks, put the cup in the bowl"):
        sc = scene([3, 1, 2, None, 4], held=5, lifted=4, phrase=phrase)
        assert set(e.truth(sc)) == {"intent", "new_order"}
        assert set(e.build_questions(sc)) == {"intent", "new_order", "escalate"}


# -- generated requests are consistent with the solver --

def test_generated_scenes_are_well_formed_and_truth_is_among_options():
    for sc in e.scenes([3, 5, 7, 9], 30, seed=0):
        blocks = [b for b in sc.row if b is not None] + ([sc.held] if sc.held is not None else [])
        assert sorted(blocks) == sc.values and len(sc.row) == sc.n
        assert (sc.held is None) == (None not in sc.row)
        if sc.held is not None:
            assert sc.row[sc.lifted_from - 1] is None
        qs = e.build_questions(sc)
        t = e.truth(sc)
        assert set(t) | {"escalate"} == set(qs)
        for k, ok in t.items():
            assert ok <= set(qs[k]["criteria"]), (sc, k)
        for v in e.VARIANTS:
            state = e.render_state(sc, v)
            assert state["user_just_said"] == sc.text


def test_solved_state_carries_the_answer_under_both_orders():
    s = e.render_state(e.canonical(), "solved")
    assert s["facts_if_descending"]["held_block"]["belongs_in"] == "slot 1"
    assert s["facts_if_descending"]["held_block"]["which_way_from_where_it_was_lifted"] == "left"
    assert s["facts_if_ascending"]["held_block"]["belongs_in"] == "slot 5"
    assert "facts_if_ascending" not in e.render_state(e.canonical(), "facts")


def test_mock_pipeline_scores_every_request():
    rng = random.Random(0)
    recs = []
    for sc in e.scenes([3, 5], 6, seed=1):
        for v in e.VARIANTS:
            qs = e.build_questions(sc)
            recs.append(e.score(sc, v, e.mock_answers(sc, qs, rng), 120.0))
    assert len(recs) == 36 and all("ask_smarter" in r for r in recs)
    e.print_tables(recs, [3, 5], e.VARIANTS, "mock")
