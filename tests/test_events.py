"""Event-driven requests: the change detector, the silence ladder's freshness rule, and the
2-of-3 majority on the `next` guard."""
from robojev.arm import ArmSnapshot
from robojev.brain import Brain
from robojev.config import DEFAULT
from robojev.events import ChangeDetector
from robojev.world import EntityView, World


def ent(eid="e1", label="white upright object A", xyz=(0.30, 0.0, 0.05)):
    return EntityView(eid, label, "a white upright object", xyz, 0.05, 0.0, 0.08, True, 0.0, 0.0, True, 0.08)


def world(entities=None, task="pick up the cup", orders=(), gripper_state="open",
          holding=None, above=None, prim=None):
    snap = ArmSnapshot(0.0, (0.25, 0.0, 0.19), 0.04, setpoint=(0.25, 0.0, 0.19), status="live")
    return World(0.0, snap, list(entities if entities is not None else [ent()]), 0.0, task, list(orders),
                 None, "hold", "directly_above", "high", "slow", "fresh",
                 gripper_state=gripper_state, holding_label=holding, above_label=above,
                 prim=prim or {"name": "move_above", "subject": "white upright object A",
                               "status": "running", "age_s": 0.4, "last_result": None})


# -- change detector -------------------------------------------------------------------------

def test_first_request_always_fires_then_a_static_scene_is_skipped():
    d = ChangeDetector(move_m=0.02, max_silence_s=1.0)
    w = world()
    ask, why = d.should_ask(w, ["hold"], 100.0)
    assert ask and why == "first request"
    d.mark_sent(100.0)
    assert d.should_ask(w, ["hold"], 100.1) == (False, "no change")
    assert d.should_ask(world(), ["hold"], 100.9)[0] is False


def test_max_silence_forces_a_request():
    d = ChangeDetector(move_m=0.02, max_silence_s=1.0)
    d.should_ask(world(), ["hold"], 100.0); d.mark_sent(100.0)
    assert d.should_ask(world(), ["hold"], 100.99)[0] is False
    ask, why = d.should_ask(world(), ["hold"], 101.0)
    assert ask and why == "max silence"


def test_material_changes_fire_and_a_small_wobble_does_not():
    d = ChangeDetector(move_m=0.02, max_silence_s=100.0)
    d.should_ask(world(), ["hold"], 0.0); d.mark_sent(0.0)
    # 1.5 cm of jitter is below the threshold
    assert d.should_ask(world([ent(xyz=(0.315, 0.0, 0.05))]), ["hold"], 0.1)[0] is False
    for w, frag in [
        (world([ent(xyz=(0.34, 0.0, 0.05))]), "moved"),
        (world([]), "disappeared"),
        (world([ent(), ent("e2", "black flat object B", (0.2, 0.1, 0.01))]), "appeared"),
        (world([ent(label="tan upright object A")]), "relabelled"),
        (world(task="put it down"), "task"),
        (world(orders=("stay 15 cm away from the laptop",)), "orders"),
        (world(gripper_state="closed on something"), "phase"),
        (world(holding="white upright object A"), "phase"),
        (world(above="white upright object A"), "phase"),
        (world(prim={"name": "lift", "subject": None, "status": "running", "age_s": 0.1, "last_result": None}), "phase"),
    ]:
        ask, why = d.should_ask(w, ["hold"], 0.1)
        assert ask and frag in why, (frag, why)
    ask, why = d.should_ask(world(), ["hold", "retreat"], 0.1)
    assert ask and "offered" in why


def test_primitive_age_alone_is_not_material():
    """prim.age_s ticks every tick; if it counted, nothing would ever be skipped."""
    d = ChangeDetector(move_m=0.02, max_silence_s=100.0)
    d.should_ask(world(), ["hold"], 0.0); d.mark_sent(0.0)
    aged = world(prim={"name": "move_above", "subject": "white upright object A",
                       "status": "running", "age_s": 3.7, "last_result": None})
    assert d.should_ask(aged, ["hold"], 0.5)[0] is False


def test_baseline_only_moves_on_an_actual_send():
    """A tick that wanted to ask but could not (in-flight cap, pause) leaves the change pending."""
    d = ChangeDetector(move_m=0.02, max_silence_s=100.0)
    d.should_ask(world(), ["hold"], 0.0); d.mark_sent(0.0)
    moved = world([ent(xyz=(0.34, 0.0, 0.05))])
    assert d.should_ask(moved, ["hold"], 0.1)[0] is True     # evaluated, but not sent
    assert d.should_ask(moved, ["hold"], 0.2)[0] is True     # still pending
    d.mark_sent(0.2)
    assert d.should_ask(moved, ["hold"], 0.3)[0] is False


def test_a_failed_request_re_asks_on_the_next_tick():
    """An error, timeout or stale drop leaves the situation unanswered: don't wait out max_silence."""
    d = ChangeDetector(move_m=0.02, max_silence_s=10.0)
    d.should_ask(world(), ["hold"], 0.0); d.mark_sent(0.0)
    assert d.should_ask(world(), ["hold"], 0.1)[0] is False
    d.retry = True                                       # loop._retry_if_newest sets this
    ask, why = d.should_ask(world(), ["hold"], 0.2)
    assert ask and why == "retry after failed request"
    d.mark_sent(0.2)
    assert d.retry is False
    assert d.should_ask(world(), ["hold"], 0.3)[0] is False


# -- silence ladder --------------------------------------------------------------------------

def ladder(brain, t):
    return brain.state(now=t)["ladder"]


def test_ladder_starts_at_rise_and_an_applied_answer_makes_it_fresh():
    b = Brain(DEFAULT)
    assert ladder(b, 100.0) == "rise"
    b.note_sent(1); b.note_applied(1, 100.0)
    assert ladder(b, 100.2) == "fresh"
    assert ladder(b, 100.6) == "hold"      # > silence_hold_s
    assert ladder(b, 102.6) == "rise"      # > silence_rise_s


def test_intentional_skips_keep_the_ladder_fresh():
    b = Brain(DEFAULT)
    b.note_sent(1); b.note_applied(1, 100.0)
    t = 100.1
    while t < 106.0:                        # 6 s of "nothing changed", no request sent
        assert b.note_skipped(t) is True
        assert ladder(b, t) == "fresh"
        t += 0.1


def test_a_sent_and_failed_request_still_walks_the_ladder():
    """Static scene + dead API: the forced max-silence request fails, so the skips in between stop
    counting as freshness and the ladder walks as it always did."""
    b = Brain(DEFAULT)
    b.note_sent(1); b.note_applied(1, 100.0)
    assert b.note_skipped(100.5) is True
    b.note_sent(11)                          # max-silence request at t=101.0; it will never come back
    for t in (101.1, 101.5, 102.0, 104.0):
        assert b.note_skipped(t) is False    # outstanding/failed request: no refresh
    assert ladder(b, 100.9) == "fresh"
    assert ladder(b, 101.4) == "hold"
    assert ladder(b, 103.0) == "rise"
    b.note_applied(11, 104.0)                # a late answer restores freshness
    assert ladder(b, 104.1) == "fresh"


def test_pause_counts_as_silence_not_as_no_change():
    """While paused the loop sends nothing and records no skip, so the ladder walks (the /api/pause
    path the README's 6 s outage test uses)."""
    b = Brain(DEFAULT)
    b.note_sent(1); b.note_applied(1, 100.0)
    assert ladder(b, 100.3) == "fresh"
    assert ladder(b, 100.8) == "hold"
    assert ladder(b, 103.0) == "rise"


# -- majority on `next` ----------------------------------------------------------------------

def choice(ch, p):
    """A distribution whose argmax is `ch` with p_max = p (the rest spread over two fillers)."""
    rest = round(1 - p, 2)
    a = round(rest / 2, 2)
    return {"type": "choice", "choice": ch, "confidence": p,
            "probabilities": {ch: p, "__a": a, "__b": round(rest - a, 2)}}


def vote(brain, a, b_, c, w=None, tag=1):
    answers = {"next": choice(*a), "next_b": choice(*b_), "next_c": choice(*c)}
    brain.apply(answers, w or world(), tag, 0.0, now=100.0 + tag)
    return brain.judgments["next"]


def test_two_of_three_agreeing_applies_with_the_mean_p():
    b = Brain(DEFAULT)
    b.offered_keys = ["retreat", "hold"]
    b.prim_status = None
    j = vote(b, ("retreat", 0.9), ("retreat", 0.7), ("hold", 0.8))
    assert j.chosen == "retreat" and abs(j.p - 0.8) < 1e-9
    assert j.votes == {"next": {"choice": "retreat", "p": 0.9},
                       "next_b": {"choice": "retreat", "p": 0.7},
                       "next_c": {"choice": "hold", "p": 0.8}}
    assert j.applied == "pending_confirmation"       # next_consecutive = 2
    j = vote(b, ("retreat", 0.9), ("retreat", 0.7), ("hold", 0.8), tag=2)
    assert j.applied == "applied" and b.prim == "retreat"


def test_no_majority_changes_nothing_and_does_not_count_a_streak():
    b = Brain(DEFAULT)
    b.offered_keys = ["retreat", "hold", "lift"]
    b.prim_status = None
    j = vote(b, ("retreat", 0.9), ("hold", 0.9), ("lift", 0.9))
    assert j.applied == "disagree" and b.prim is None
    # the disagreement must not have counted towards retreat's hysteresis
    vote(b, ("retreat", 0.9), ("retreat", 0.9), ("lift", 0.9), tag=2)
    assert b.prim is None                             # first consecutive retreat only
    vote(b, ("retreat", 0.9), ("retreat", 0.9), ("lift", 0.9), tag=3)
    assert b.prim == "retreat"


def test_a_lone_variant_still_counts_degraded():
    b = Brain(DEFAULT)
    b.offered_keys = ["hold"]
    b.prim_status = None
    b.apply({"next": choice("hold", 0.9)}, world(), 1, 0.0, now=100.0)
    j = b.judgments["next"]
    assert j.chosen == "hold" and j.applied == "applied" and j.votes == {"next": {"choice": "hold", "p": 0.9}}


def test_majority_below_the_gate_is_still_gated():
    b = Brain(DEFAULT)
    b.offered_keys = ["retreat", "hold"]
    b.prim_status = None
    j = vote(b, ("retreat", 0.4), ("retreat", 0.4), ("hold", 0.9))
    assert j.chosen == "retreat" and j.applied == "gated" and b.prim is None
