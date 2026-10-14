"""The crawl slice (docs/widowx-pair.md "Crawl: one brain, two arms"): the two-arm brain (H2,
robojev.brain, tested until now on the physics-free stub) driving the two-arm MuJoCo WidowX server
(K5, robojev.robots.widowx_sim, tested until now from the conformance script), realtime=True, with
e12v2's mock decider (one oracle per arm) and the hand-over mock planner (mocks.PairLLM); no API
keys. The layout is the pair scene's: blocks in arm_0's half, the tray in arm_1's, the shared zone
between with three hand-over spots, so every block crosses the zone (arm_0 to a spot, arm_1 from
there to the tray) and the server's zone lock makes one arm wait for the other. Like the one-arm
slice (tests/test_slice.py) this exists to surface mismatches between the halves; judged from the
server's world(), never from what the brain or a skill says about itself:

  (a) a 3-block numbered sort across the table, lowest on the left: every block crosses the zone;
      wall time, how many moments both arms had a skill in flight, the zone waits, no safety_trip;
  (b) the same with "actually, highest on the left" typed while arm_1 is carrying: both arms'
      decisions within 0.5 s, the order reversed, nothing dropped, nothing left in the zone;
  (c) a hand near arm_1 mid-carry: arm_1 pauses within 0.5 s and arm_0 keeps working; resume once
      the hand leaves; no hand contact;
  (d) STOP while both arms hold blocks: both park, nothing dropped from height.

The sim has no person: the hand in (c) is a perception override in front of the server
(mocks.ScriptedPerception, labelled as such there), not a body in the physics, so "no contact" is
the closest arm_1's gripper came to the reported hand position.

Measured numbers are printed per test (pytest -s) and recorded in experiments/results/crawl_slice.txt.
pytest-asyncio is not installed: each episode runs under asyncio.run in a plain test.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time

import pytest

from robojev import conformance as trials
from robojev.brain import Brain, BrainConfig
from robojev.brain.eval.oracle import Goal
from robojev.brain.eval.scenarios import Utt
from robojev.brain.mocks import PRESS_STOP, Line, ScriptedPerception, ScriptedUser, pair_models
from robojev.robots.widowx_sim import scene

pytestmark = pytest.mark.skipif(not scene.FOLLOWER_XML.exists(),
                                reason=f"{scene.FOLLOWER_XML} missing: set TROSSEN_ARM_MUJOCO_DIR")
pytest.importorskip("mujoco")
from robojev.robots.widowx_sim.server import PAIR_HOME_XY, TRAVEL_Z, SimRobotServer  # noqa: E402

A0, A1 = scene.ARM_0, scene.ARM_1
ARMS = (A0, A1)
TASK = "Line the blocks up in the tray, lowest number on the left."
CORRECTION = "actually, highest on the left"
REACT_S = 0.5                      # the bar: text or hand -> the server reports holding/paused on the arm concerned
REST_Z = scene.TABLE_Z + 0.02      # a 4 cm block's centre when it sits on the table
FALL_M = 0.02                      # higher than this, unheld, for longer than the grace = dropped from height
GRACE_S = 0.2                      # a flicker of the physics grasp detector is not a drop (skills.Params.drop_grace)
ZONE_WAIT = "waiting for the shared zone"
ZONE_SPOTS = ("zone_0", "zone_1", "zone_2")

# blocks 5, 1, 3 left to right as arm_0 sees them (+y is its left), in its half (conformance.PAIR_X_RANGE);
# the tray's three slots are in arm_1's half, slot_0 the leftmost as arm_1 sees them
BLOCKS = [scene.Block("block_5", (0.19, 0.10)), scene.Block("block_1", (0.19, 0.0)), scene.Block("block_3", (0.19, -0.10))]
ASCENDING = Goal({"block_1": "slot_0", "block_3": "slot_1", "block_5": "slot_2"})
DESCENDING = Goal({"block_5": "slot_0", "block_3": "slot_1", "block_1": "slot_2"})


# ---------------------------------------------------------------- helpers


@pytest.fixture
def pair():
    """Make the pair physics server on the wall clock behind the logging perception layer (`arm`:
    the arm whose history it keeps and whose gripper a scripted hand appears near)."""
    made = []

    def make(arm: str = A0) -> ScriptedPerception:
        sim = SimRobotServer(scene.pair(list(BLOCKS), n_slots=3), realtime=True).open()
        p = ScriptedPerception(sim, arm=arm).open()
        p.events = []
        p.subscribe(p.events.append)
        made.append((sim, p))
        return p
    yield make
    for sim, p in made:
        p.close()
        sim.close()


class Watch:
    """Polls world() (and status() of each skill on an arm) every 10 ms on a thread: per arm its mode,
    what it holds, its skill and whether that skill is waiting for the shared zone; every object's
    where. Samples are recorded on change, with the wall time."""

    def __init__(self, srv: ScriptedPerception):
        self.srv, self.samples = srv, []
        self._halt = threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self._t.join(timeout=1.0)

    def _poll(self):
        last = None
        while not self._halt.wait(0.01):
            w = self.srv.world()
            arms = {}
            for a in w.arms.values():
                waiting = a.skill is not None and self.srv.status(a.skill).phase_text.startswith(ZONE_WAIT)
                arms[a.id] = (a.mode, a.holding, a.skill, waiting)
            rec = (arms, {o.id: o.where for o in w.objects.values()})
            if rec != last:
                self.samples.append((time.monotonic(), *rec))
                last = rec

    def spans(self, cond) -> list[tuple[float, float]]:
        """(start, length) of every spell during which `cond(arms)` held."""
        out, start = [], None
        for t, arms, _ in self.samples:
            if cond(arms) and start is None:
                start = t
            elif not cond(arms) and start is not None:
                out.append((start, t - start))
                start = None
        if start is not None:
            out.append((start, self.samples[-1][0] - start))
        return out

    def both_in_flight(self) -> list[tuple[float, float]]:
        return self.spans(lambda arms: all(arms[a][2] is not None for a in ARMS))

    def zone_waits(self, arm: str) -> list[tuple[float, float]]:
        return self.spans(lambda arms: arms[arm][3])

    def crossed(self, oid: str) -> bool:
        """The object was seen resting on a shared-zone spot at some sample."""
        return any(objs[oid] in ZONE_SPOTS for _, _, objs in self.samples)

    def during(self, t0: float, t1: float, arm: str) -> set[str]:
        """Modes `arm` reported in (t0, t1): the one in force at t0 and every change inside."""
        before = [arms[arm][0] for t, arms, _ in self.samples if t <= t0]
        return set(before[-1:]) | {arms[arm][0] for t, arms, _ in self.samples if t0 < t < t1}


def brain(srv, versions, utts=None, user=None, **cfg) -> Brain:
    decider, llm = pair_models(srv, versions, utts)
    return Brain(srv, decider, llm, user, BrainConfig(max_s=150, **cfg))


def episode(b: Brain):
    t = time.monotonic()
    r = asyncio.run(b.run(TASK))
    return r, time.monotonic() - t


def carrying(srv, arm: str, oid: str | None = None, after_s: float = 0.3):
    """A condition: `arm`'s skill has been in its carry phase, `oid` (or anything) in the gripper,
    for `after_s`."""
    seen: list[float] = []

    def cond(*_) -> bool:
        a = srv.world().arms[arm]
        if a.holding is None or (oid is not None and a.holding != oid) or a.skill is None \
                or srv.status(a.skill).phase != "carry":
            seen.clear()
            return False
        if not seen:
            seen.append(time.monotonic())
        return time.monotonic() - seen[0] >= after_s
    return cond


def both_holding(srv):
    return lambda *_: all(a.holding is not None for a in srv.world().arms.values())


def judge_sort(srv, goal: Goal) -> tuple[bool, str]:
    """Every block resting in its slot, both grippers empty, both arms at travel height: from world() only."""
    w, m = srv.world(), srv.manifest()
    verdicts = [trials.judge(w, m, oid, place, A1) for oid, place in goal.assign.items()]
    problems = [f"{A0}: {p}" for p in trials._arm_checks(w, m, A0)]
    return all(ok for ok, _ in verdicts) and not problems, "; ".join([why for _, why in verdicts] + problems)


def tools(srv, *names, arm: str | None = None):
    return [c for c in srv.calls if c[1] in names and (arm is None or (c[2] and c[2][0] == arm))]


def first_mode(srv, arm: str, modes, after: float) -> float | None:
    """Wall time the server first reported `arm` in one of `modes` at or after `after`."""
    return next((t for t, a, mode, _ in srv.arm_log if a == arm and t >= after and mode in modes), None)


def drops(srv, oid: str, since: float = 0.0) -> list[tuple[float, float]]:
    """Spells in the history where `oid` was in no gripper and not near the table for longer than the
    grasp detector's grace: (when, how long). Empty = never dropped from height."""
    out, start = [], None
    for t, _, _, objs in srv.history:
        if t < since:
            continue
        where, z = objs[oid]
        high = not where.startswith("gripper:") and z > REST_Z + FALL_M
        if high and start is None:
            start = t
        elif not high and start is not None:
            if t - start > GRACE_S:
                out.append((start, t - start))
            start = None
    if start is not None and srv.history[-1][0] - start > GRACE_S:
        out.append((start, srv.history[-1][0] - start))
    return out


def no_drops(srv, since: float = 0.0) -> dict[str, list]:
    return {b.name: d for b in BLOCKS if (d := drops(srv, b.name, since))}


def decided_at(b: Brain, r, kind: str) -> dict[str, float]:
    """Wall time of each arm's decision on the (one) event of `kind`; the brain's clock starts at `_t0`."""
    return {d["arm"]: b._t0 + d["t"] for d in r.decisions if d["event"] == kind}


def holding_at(srv, t: float) -> dict[str, str | None]:
    """What each arm held at wall time `t`, from the perception layer's per-arm log."""
    return {a: next((h for tt, i, _, h in reversed(srv.arm_log) if i == a and tt <= t), None) for a in ARMS}


def summary(name: str, r, wall: float, **extra) -> None:
    bits = [f"wall {wall:.1f} s", f"events {dict(r.events)}", f"steps started/done/failed {r.steps_started}/{r.steps_done}/{r.steps_failed}",
            f"plans {len(r.llm)} (stale {r.stale_plans}, failed {r.failed_plans}, loops {r.loops})"]
    bits += [f"{k} {v}" for k, v in extra.items()]
    print(f"\nCRAWL {name}: " + "; ".join(bits))


def starts_by_arm(srv) -> dict[str, list[tuple[str, str]]]:
    return {a: [(c[2][2]["object"], c[2][2]["place"]) for c in tools(srv, "start", arm=a)] for a in ARMS}


# ---------------------------------------------------------------- (a) the sort across the table


def test_a_three_block_sort_across_the_table(pair):
    srv = pair()
    with Watch(srv) as watch:
        r, wall = episode(brain(srv, [ASCENDING]))
    ok, why = judge_sort(srv, ASCENDING)
    both = watch.both_in_flight()
    waits = {a: watch.zone_waits(a) for a in ARMS}
    summary("sort", r, wall, verdict=why, both_in_flight=f"{len(both)} spells, {sum(d for _, d in both):.1f} s",
            zone_waits={a: f"{len(w)} x {sum(d for _, d in w):.1f} s" for a, w in waits.items()}, starts=starts_by_arm(srv))
    assert r.outcome == "done", r.log
    assert ok, why
    assert r.steps_done == 6 and r.steps_failed == 0 and r.failed_plans == 0 and len(r.llm) == 1
    # the hand-over pattern on the wire: arm_0 to a zone spot, arm_1 from there to the tray, each block once
    starts = starts_by_arm(srv)
    assert sorted(o for o, _ in starts[A0]) == sorted(o for o, _ in starts[A1]) == sorted(b.name for b in BLOCKS)
    assert all(p in ZONE_SPOTS for _, p in starts[A0]) and all(p.startswith("slot_") for _, p in starts[A1])
    assert all(watch.crossed(b.name) for b in BLOCKS)          # seen resting on a spot, from world()
    assert both and sum(d for _, d in both) > 1.0              # both arms had a skill in flight at once
    assert any(waits.values())                                 # the lock made someone wait, and nothing worse
    assert not [e for e in srv.events if e.kind in ("safety_trip", "skill_failed")]
    assert r.events["scene_change"] == 0                       # no placement read as someone else's doing
    assert not tools(srv, "stop", "pause")
    assert wall < 120.0


# ---------------------------------------------------------------- (b) a correction while arm_1 carries


def test_b_correction_while_arm_1_carries_reverses_the_order(pair):
    srv = pair()
    user = ScriptedUser([Line(carrying(srv, A1), CORRECTION)])
    b = brain(srv, [ASCENDING, DESCENDING], {CORRECTION: Utt("correction", version=1)}, user)
    with Watch(srv) as watch:
        r, wall = episode(b)
    t_text = user.t0 + user.said[0][0]
    decided = decided_at(b, r, "user_text")
    t_hold = first_mode(srv, A1, ("holding", "paused"), t_text)
    ok, why = judge_sort(srv, DESCENDING)
    dropped = no_drops(srv)
    said = [d for d in r.decisions if d["event"] == "user_text"]
    summary("correction", r, wall, text_to_decisions_s={a: round(t - t_text, 3) for a, t in decided.items()},
            text_to_arm_1_hold_s=None if t_hold is None else round(t_hold - t_text, 3),
            arm_0_at_text=next((arms[A0][:2] for t, arms, _ in reversed(watch.samples) if t <= t_text), None),
            verdict=why, starts=starts_by_arm(srv), drops=dropped)
    assert r.outcome == "done", r.log
    assert sorted(decided) == list(ARMS) and all(t - t_text < REACT_S for t in decided.values()), (decided, t_text)
    assert t_hold is not None and t_hold - t_text < REACT_S, (t_hold, t_text)
    assert sorted(d["arm"] for d in said) == list(ARMS)         # one decision per arm
    assert all((d["combined"].right_now, d["combined"].route) == ("hold", "fast_llm") for d in said)
    replans = [x for x in r.llm if x["kind"] == "replan"]
    assert len(replans) == 1 and replans[0]["ok"]               # one replan for the event, not one per arm
    assert ok, why                                              # reversed, and so nothing left in the zone
    assert dropped == {} and r.steps_failed == 0 and r.stale_plans == 0
    assert not [e for e in srv.events if e.kind in ("safety_trip", "skill_failed")]
    assert not tools(srv, "stop")


# ---------------------------------------------------------------- (c) a hand near arm_1


def test_c_hand_near_arm_1_pauses_arm_1_only_and_arm_0_keeps_working(pair):
    srv = pair(arm=A1)
    # a second into arm_1's carry from the zone it is in its own half, so arm_0 may work meanwhile
    srv.hand_when(carrying(srv, A1, after_s=1.0), for_s=1.5, dist=0.15)
    with Watch(srv) as watch:
        r, wall = episode(brain(srv, [ASCENDING]))
    appeared = next(t for t, what in srv.hand_log if what == "appears")
    left = next(t for t, what in srv.hand_log if what == "leaves")
    t_pause = first_mode(srv, A1, ("paused",), appeared)
    resumed = [t for t, _, _ in tools(srv, "resume", arm=A1)]
    ok, why = judge_sort(srv, ASCENDING)
    arm_0_meanwhile = watch.during(t_pause or appeared, resumed[0] if resumed else left, A0)
    summary("hand", r, wall, hand_to_pause_s=None if t_pause is None else round(t_pause - appeared, 3),
            leave_to_resume_s=None if not resumed else round(resumed[0] - left, 3), min_hand_dist_m=round(srv.min_hand_dist, 3),
            arm_0_modes_meanwhile=sorted(arm_0_meanwhile), verdict=why)
    assert r.outcome == "done", r.log
    assert t_pause is not None and t_pause - appeared < REACT_S, (t_pause, appeared)
    assert t_pause < left
    assert resumed and resumed[0] >= left                      # resumed only once the hand had gone
    # arm_1 was the only arm paused, and the hand was arm_1's decision only
    assert all(c[2] == (A1,) for c in tools(srv, "pause", "resume"))
    assert not [t for t, a, mode, _ in srv.arm_log if a == A0 and mode == "paused"]
    hand = [d for d in r.decisions if d["event"] == "scene_change"]
    assert hand and {d["arm"] for d in hand} == {A1}
    # paused means paused: arm_1 reported nothing but "paused" between the pause and the resume
    between = {mode for t, a, mode, _ in srv.arm_log if a == A1 and t_pause < t < resumed[0]}
    assert between <= {"paused"}, between
    assert "running" in arm_0_meanwhile                        # arm_0 kept working while arm_1 was paused
    assert srv.min_hand_dist > 0.05                            # arm_1's gripper never reached the hand
    assert ok, why
    assert no_drops(srv) == {} and r.steps_failed == 0
    assert not tools(srv, "stop")


# ---------------------------------------------------------------- (d) STOP while both arms hold blocks


def test_d_stop_while_both_arms_hold_blocks_parks_both_and_drops_nothing(pair):
    srv = pair()
    user = ScriptedUser([Line(both_holding(srv), PRESS_STOP)])
    r, wall = episode(brain(srv, [ASCENDING], user=user))
    t_stop = tools(srv, "stop")[0][0]
    held_at_stop = holding_at(srv, t_stop)
    assert r.outcome == "stopped"
    assert not [c for c in srv.calls if c[0] > t_stop and c[1] != "stop"]       # nothing sent after STOP
    # the server parks both at crawl speed (0.03 m/s): wait for them
    homes = {a: srv.inner.arms[a].base.to_common((PAIR_HOME_XY[0], PAIR_HOME_XY[1])) for a in ARMS}

    def parked(a) -> bool:
        return math.hypot(a.x - homes[a.id][0], a.y - homes[a.id][1]) < 0.01 and abs(a.z - TRAVEL_Z) < 0.01
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline and not all(parked(a) for a in srv.world().arms.values()):
        time.sleep(0.05)
    w = srv.world()
    parked_after = time.monotonic() - t_stop
    dropped = no_drops(srv, since=t_stop)
    summary("stop", r, wall, stop_to_parked_s=round(parked_after, 2), held_at_stop=held_at_stop,
            arms={a: (x.mode, x.holding, round(x.z, 3)) for a, x in w.arms.items()},
            blocks={o.id: (o.where, round(o.z, 3)) for o in w.objects.values()}, drops=dropped)
    assert all(h is not None for h in held_at_stop.values()), held_at_stop     # STOP came with a block in each gripper
    for a in w.arms.values():
        assert a.mode == "stopped" and parked(a), (a.id, a.mode, a.x, a.y, a.z)
    for o in w.objects.values():                               # every block held, or resting on the table / a place
        assert o.where.startswith("gripper:") or abs(o.z - REST_Z) < 0.005, (o.id, o.where, o.z)
    assert dropped == {}, dropped
    failed = [e for e in srv.events if e.kind == "skill_failed"]
    assert len(failed) == 2 and all(e.text.startswith("stopped: ") for e in failed) and {e.arm for e in failed} == set(ARMS)
    assert r.events["user_text"] == 0                          # STOP is code: no decision is asked about it
