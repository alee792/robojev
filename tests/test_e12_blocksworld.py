"""E12 harness (experiments/e12_blocksworld/): closed-loop runs with offline backends, the Limiter, the
policy, and the dry run. Offline; live runs need a Jev key."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from e12_blocksworld import backends as B  # noqa: E402
from e12_blocksworld import cli  # noqa: E402
from e12_blocksworld import plan as P  # noqa: E402
from e12_blocksworld.harness import Runtime, Variant, run_one  # noqa: E402
from e12_blocksworld.scenarios import SCENARIOS, make  # noqa: E402
from e12_blocksworld.world import TRAVEL_Z, Noise, Perception  # noqa: E402

ALL = list(SCENARIOS)


def run(name, backend=None, variant=None, **kw):
    return run_one(make(name, 12), backend or B.Oracle(), variant or Variant(), **kw)


@pytest.mark.parametrize("name", ALL)
def test_oracle_completes_every_scenario_with_zero_violations(name):
    m = run(name)
    assert m.completed, name
    assert m.forbidden_touch == 0 and m.hand_contact == 0
    assert m.utt_missed == 0 and m.utt_spurious == 0
    assert all(ok == n for ok, n in m.agree.values())


@pytest.mark.parametrize("name", ALL)
def test_mock_at_zero_error_matches_oracle(name):
    a, b = run(name), run(name, B.Mock(error=0.0, conf_noise=0.1, seed=3))
    for f in ("completed", "done_tick", "skills", "failed", "requests", "escalations", "gated", "utt_to_change",
              "forbidden_touch", "hand_contact"):
        assert getattr(a, f) == getattr(b, f), f


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("variant", [Variant("solved"), Variant("raw", spotter=False, listener=False)], ids=lambda v: v.name)
def test_mock_with_errors_and_noise_runs_to_the_end(name, variant):
    m = run(name, B.Mock(error=0.3, conf_noise=0.2, seed=5), variant, noise=Noise(0.05, 0.05, 1.0), max_s=120, latency_ticks=3)
    assert m.ticks > 0 and sum(m.requests.values()) > 0
    assert m.forbidden_touch == 0   # always-on orders are code: no model error can break them


def test_no_spotter_lets_the_hand_get_touched_and_the_spotter_prevents_it():
    assert run("hand_in_path", variant=Variant("solved", spotter=False)).hand_contact >= 1
    m = run("hand_in_path")
    assert m.hand_contact == 0 and m.requests["spotter"] > 0


def test_no_listener_routes_utterances_to_the_planner_and_is_slower():
    fast = run("reverse_midway")
    slow = run("reverse_midway", variant=Variant("solved", listener=False))
    assert slow.completed and slow.escalations >= 1 and slow.requests["listener"] == 0
    assert slow.utt_to_change[0] > fast.utt_to_change[0]


def test_chatter_changes_nothing():
    m = run("chatter")
    assert m.utt_spurious == 0 and m.done_tick == run("crawl_sort").done_tick


def _rt_above(name, bid):
    rt = Runtime(make(name, 12), B.Oracle(), Variant())
    rt._perceive()
    b = rt.world.blocks[bid]
    rt.world.arm.x, rt.world.arm.y, rt.world.arm.z = b.x, b.y, TRAVEL_Z
    rt._perceive()
    return rt


def test_limiter_refuses_a_forbidden_pick_even_when_a_layer_picks_it():
    from e12_blocksworld.harness import Pending
    from e12_blocksworld.skills import offered
    rt = _rt_above("forbidden_moves", "g1")
    assert "pick:g1" not in offered(rt)          # code never offers it...
    picks = {"escalate": {"choice": "decide_now", "confidence": 1.0},
             "task_status": {"choice": "not_complete", "confidence": 1.0},
             "next_skill": {"choice": "pick:g1", "confidence": 1.0}}
    rt.apply(Pending(0, "sequencer", picks))       # ...a Sequencer pick of it is dropped as not offered...
    assert rt.skill is None
    rt.dispatch("pick:g1")                         # ...and a direct command is refused by the Limiter
    assert rt.skill is None and rt.world.arm.holding is None
    assert "refused by the Limiter" in rt.world.arm.last_result
    assert rt.limiter.refusals >= 1


def test_limiter_refuses_a_grasp_that_would_touch_the_forbidden_block():
    rt = _rt_above("forbidden_moves", "b1")
    g, b = rt.world.blocks["g1"], rt.world.blocks["b1"]
    g.x, g.y = b.x + 3.2, b.y
    rt._perceive()
    from e12_blocksworld.skills import offered
    opts = offered(rt)
    assert "pick:b1" not in opts and "nudge_clear:b1" in opts
    rt.dispatch("pick:b1")
    assert rt.skill is None and "refused" in rt.world.arm.last_result
    # a descent straight onto the green block is refused step by step
    why = rt.limiter.check_step(rt.scene, (g.x, g.y, 7.0), (g.x, g.y, 4.0), 4.5)
    assert why and "green block" in why


def test_policy_parks_the_held_block_when_its_slot_is_taken():
    sc = make("crawl_sort", 12)
    w = sc.world
    w.blocks["b3"].where, w.blocks["b3"].x, w.blocks["b3"].y = "slot:1", *w.slots[1]
    w.arm.holding, w.blocks["b1"].where = "b1", "gripper"
    ts = Perception.truth(w)
    assert P.goals(sc.plan, {"order": P.DESC}, ts)["b5"] == "slot:1"
    assert P.next_skill(sc.plan, {"order": P.ASC, "pace": "normal"}, ts, w.arm) == "carry_to:park"


def test_dry_run_writes_one_sample_per_layer_per_scenario(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(["--dry-run", "--scenario", "crawl_sort,reverse_midway,hand_in_path"])
    got = sorted(p.name for p in (tmp_path / "e12_samples").iterdir() if not p.name.startswith("plan_"))
    assert (tmp_path / "e12_samples" / "plan_sort_number.json").exists()
    assert got == ["crawl_sort_sequencer.json", "hand_in_path_sequencer.json", "hand_in_path_spotter.json",
                   "reverse_midway_listener.json", "reverse_midway_sequencer.json"]
    assert "$" in capsys.readouterr().out
