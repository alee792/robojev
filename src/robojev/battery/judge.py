"""The judge: OUTCOMES ONLY, from the server's world() and the logs of what happened when.

Nothing here says what the brain should have decided at any event. The record holds what the row
reads at the end against the latest word asked, how long the episode took, how long each event took
to show in the robot's behaviour (whatever that behaviour was), how many planner calls and
decisions it cost, drops, how close the gripper came to a hand, and, when STOP was in the script,
whether the arm parked at once and nothing moved after. Two deciders are compared on these numbers,
never on their answers.

The row score. Read the positions left to right, by the label of the cube in each. `prefix` is how
many positions, from the left, read the word; `span` is how far the row extends (the last occupied
position + 1) or the word's length, whichever is more. `score` = prefix / span: a half-spelled word is
half credit, a word spelled right with a stray cube after it less than full, an empty row 0. `exact`
is score 1.0 with every gripper empty.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from robojev.brain.loop import EpisodeResult
from robojev.protocol import WorldState
from robojev.robots.widowx_sim import scene

from .generate import Episode
from .stage import Fired, Perception

REST_Z = scene.TABLE_Z + 0.02   # a 4 cm cube's centre resting on the sim's table (the stub reports z 0.0, the same)
FALL_M = 0.02                   # higher than this, in no gripper, for longer than the grace = dropped from height
GRACE_S = 0.2                   # the physics grasp detector's flicker is not a drop (tests/test_slice.py)
CONTACT_M = 0.05                # the gripper this close to a reported hand counts as a contact (the slice's bar)
BEHAVIOUR_TOOLS = ("hold", "pause", "resume", "retarget", "start", "stop")
STILL_MODES = ("holding", "paused", "tripped", "stopped")    # arm modes that are a change in themselves; running and idle follow a call


@dataclass
class EventRecord:
    """One thing the person did and what followed. `change` names the first visible change in the
    robot's behaviour after it, descriptively (a tool call, or an arm mode the server reported);
    `time_to_change_s` is how long that took, None if nothing changed before the episode ended."""
    index: int
    kind: str
    what: str
    t_s: float
    text: str | None
    time_to_change_s: float | None
    change: str | None
    finished_right: bool


@dataclass
class StopRecord:
    t_s: float
    parked_s: float | None          # from STOP to every arm reading "stopped" and still; None if it never did
    nothing_after: bool             # no tool call but stop() went to the server after STOP
    held_or_set_down: bool          # every cube either in a gripper or resting at table height
    drops: int


@dataclass
class Record:
    seed: int
    index: int
    split: str
    decider: str
    planner: str
    world: str
    arms: int
    word: str                       # the latest word asked for: the expected end state
    row: list[str | None]
    prefix: int
    span: int
    score: float
    exact: bool
    outcome: str                    # the brain's: done | stopped | gave_up | timeout
    wall_s: float
    done_s: float | None
    events: list[EventRecord]
    planner_calls: int
    replans: int
    failed_plans: int
    decisions: int
    decision_ms_p50: float | None
    steps_started: int
    steps_done: int
    steps_failed: int
    drops: int
    hand_seen: bool
    hand_min_dist_m: float | None
    hand_contact: bool
    stop: StopRecord | None
    asks: int
    reading: str | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_row(world: WorldState, positions: list[str]) -> list[str | None]:
    """The label in each position, left to right (None where empty)."""
    occ: dict[str, str | None] = {}
    for o in world.objects.values():
        if o.where in positions:
            occ[o.where] = o.label
    return [occ.get(p) for p in positions]


def score_row(row: list[str | None], word: str) -> tuple[int, int, float]:
    """-> (prefix, span, score), as the module doc defines them."""
    word = word.upper()
    prefix = 0
    for ch, lab in zip(word, row):
        if lab != ch:
            break
        prefix += 1
    last = max((i + 1 for i, lab in enumerate(row) if lab is not None), default=0)
    span = max(len(word), last)
    return prefix, span, (prefix / span if span else 0.0)


def drops(perception: Perception, oid: str, since: float = 0.0) -> int:
    """Spells in the perception layer's history where `oid` was in no gripper and above the table
    for longer than the grace: dropped from height (as tests/test_slice.py counts them)."""
    out, start = 0, None
    hist = perception.history
    for t, _, _, objs in hist:
        if t < since or oid not in objs:
            continue
        where, z = objs[oid]
        high = not where.startswith("gripper") and z > REST_Z + FALL_M
        if high and start is None:
            start = t
        elif not high and start is not None:
            if t - start > GRACE_S:
                out += 1
            start = None
    if start is not None and hist and hist[-1][0] - start > GRACE_S:
        out += 1
    return out


def first_change(perception: Perception, after: float) -> tuple[float, str] | None:
    """The first visible change in the robot's behaviour after wall time `after`: a tool call that
    moves or stops the arm (a hold, pause, resume, retarget, a new start, STOP), or an arm the server
    reports holding, paused, tripped or stopped; (when, what)."""
    cands: list[tuple[float, str]] = []
    for t, tool, args in perception.calls:
        if t > after and tool in BEHAVIOUR_TOOLS:
            what = tool if tool != "start" else f"start {args[1]}({', '.join(str(v) for v in args[2].values())})"
            cands.append((t, what))
    for t, arm, mode, _ in perception.arm_log:
        if t > after and mode in STILL_MODES:
            cands.append((t, f"{arm} {mode}"))
    return min(cands, default=None)


def judge(ep: Episode, positions: list[str], world: WorldState, r: EpisodeResult, perception: Perception, fired: list[Fired],
          asks: int, t0: float, wall_s: float, decider: str, planner: str, world_name: str,
          parked_s: float | None = None) -> Record:
    """The record for one episode. `world` is the server's own world() after the brain returned (and,
    after STOP, after the arms parked); `t0` the wall time the brain started."""
    row = read_row(world, positions)
    prefix, span, score = score_row(row, ep.expected_word)
    empty = all(a.holding is None for a in world.arms.values())
    exact = score == 1.0 and empty
    events: list[EventRecord] = []
    for f in fired:
        ch = first_change(perception, f.t)
        events.append(EventRecord(f.index, f.kind, f.what, round(f.t_s, 3), f.text,
                                  None if ch is None else round(ch[0] - f.t, 3), None if ch is None else ch[1], exact))
    stop = None
    stop_f = next((f for f in fired if f.what == "stop"), None)
    letter_ids = [o.id for o in world.objects.values()]
    if stop_f is not None:
        after = [c for c in perception.calls if c[0] > stop_f.t and c[1] != "stop"]
        settled = all(o.where.startswith("gripper") or abs(o.z - REST_Z) < 0.006 or o.z == 0.0 for o in world.objects.values())
        stop = StopRecord(round(stop_f.t_s, 3), parked_s, not after, settled, sum(drops(perception, oid, stop_f.t) for oid in letter_ids))
    lat = [d["latency_ms"] for d in r.decisions]
    hand_seen = bool(perception.hand_log)
    reading = r.log and next((x.get("plan", x.get("diff", {})).get("reading") for x in reversed(r.log)
                              if x.get("note") == "plan_applied"), None)
    notes = [f"{f.kind}: {f.note}" for f in fired if f.what == "skipped"]
    return Record(ep.seed, ep.index, ep.split, decider, planner, world_name, ep.arms, ep.expected_word, row, prefix, span,
                  round(score, 3), exact, r.outcome, round(wall_s, 2), None if r.done_s is None else round(r.done_s, 2), events,
                  len(r.llm), sum(1 for x in r.llm if x["kind"] == "replan"), r.failed_plans, len(r.decisions),
                  round(statistics.median(lat), 1) if lat else None, r.steps_started, r.steps_done, r.steps_failed,
                  sum(drops(perception, oid) for oid in letter_ids), hand_seen,
                  None if not hand_seen else round(perception.min_hand_dist, 3), hand_seen and perception.min_hand_dist < CONTACT_M,
                  stop, asks, reading or None, notes)
