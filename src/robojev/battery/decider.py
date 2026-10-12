"""The OFFLINE STAND-IN for the decision model on the spelling task: SpellJev, a DecisionBackend that
answers the brain's decision request from GROUND TRUTH (the script's own account of each line, the
server's true world), with full confidence, so the fast loop can run end to end with no keys. It is
not a model and proves nothing about one; it is what the controls (controls.py) are compared with
offline, and what the live deciders replace on the Mac (models.py).

It runs behind e12v2's JevDecider, so the request building, combine() and the gates are the real
brain's; only the answers are oracle-like. `truth()` says which options are acceptable for a
question (preferred first); the backend picks the first the request offers, as eval/mocks.MockJev
does for the sort. Nothing here is read by the judge: the battery scores outcomes, not answers.

What it answers, for the record (not a specification of right answers, only this mock's):
  user text   a new word: hold if the move in progress does not fit it, else carry on; route to the
              planner, or to the user when no cube in view can spell the word. A wait: pause. A go-on:
              resume. Chatter, a remark, a reply: carry on; a reply goes to the planner.
  hand        very close: back off; coming closer or in the path: pause; gone: resume if paused for it.
  cubes       the current step's cube moved before the grasp: re-target. A cube taken: planner if a
              spare cube carries the letter, else the user. A cube put down: planner if the plan has no
              step for its letter, else carry on. A placed cube pulled out: re-queue its step. A cube
              the person put into a position: planner. Else carry on.
  steps       done: carry on. Failed: retry a grasp that missed, a drop or a stall (twice), run another
              step first when the place is occupied and one can run, otherwise the planner.
  plan check  a cube the plan never mentions is flagged when the plan does not spell the word and
              that cube's letter is in it.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable

from robojev.brain.core.combine import Gates, JevDecider
from robojev.brain.core.decision import View, object_step
from robojev.brain.core.plan import step_goal, steps_by_id
from robojev.brain.mocks import PerArmDecider, TimedJev
from robojev.protocol import WorldState

from .generate import Episode, Line, spellable
from .spell import SLOT_RE

LATENCY_MS = 150.0            # Jev's measured median in the loop (docs/live-results.md)


def positions_of(world: WorldState) -> list[str]:
    """The position ids of a world, left to right (slot_0, slot_1, ...)."""
    found = [(int(m.group(1)), p) for p in world.places for m in [SLOT_RE.match(p)] if m]
    return [p for _, p in sorted(found)]


def row_of(world: WorldState, positions: list[str]) -> list[str | None]:
    occ = {o.where: o.label for o in world.objects.values()}
    return [occ.get(p) for p in positions]


class SpellTruth:
    """Ground truth for one episode: what each line is (from the script), the word in force (from
    what has been said), the true world (from the perception layer's server, hidden cubes left out:
    what a perfect perception would see)."""

    def __init__(self, ep: Episode, world: Callable[[], WorldState]):
        self.ep, self.world = ep, world
        self.lines: dict[str, Line] = dict(ep.lines)

    def line(self, text: str) -> Line:
        return self.lines.get(text) or Line("chatter")

    def word_now(self, messages: list[str]) -> str:
        w = self.ep.word
        for m in messages:
            ln = self.lines.get(m)
            if ln is not None and ln.kind == "new_word" and ln.word:
                w = ln.word
        return w


def _step_wrong(step: dict, word: str, world: WorldState, positions: list[str]) -> bool:
    """Does `step` (a move) no longer fit `word`: its cube to a position whose letter is not the
    cube's, or past the word's end, or a clear of a cube that now sits where the word wants it."""
    g = step_goal(step)
    if g is None:
        return False
    oid, dest = g
    o = world.objects.get(oid)
    label = o.label if o is not None else None
    if dest in positions:
        i = positions.index(dest)
        return i >= len(word) or word[i] != label
    if o is not None and o.where in positions:
        j = positions.index(o.where)
        return j < len(word) and word[j] == label
    return False


def _plan_row(view: View, world: WorldState, positions: list[str]) -> dict[str, str | None]:
    """What each position would hold after the plan's queue, run from the true world (position -> label)."""
    if not view.plan:
        return {}
    byid = steps_by_id(view.plan)
    where = {o.id: o.where for o in world.objects.values()}
    for sid in view.queue:
        g = step_goal(byid.get(sid, {"skill": "hold"}))
        if g:
            where[g[0]] = g[1]
    label = {o.id: o.label for o in world.objects.values()}
    return {w: label.get(o) for o, w in where.items() if w in positions and o in label}   # a cube out of view spells nothing


def _plan_spells(view: View, word: str, world: WorldState, positions: list[str]) -> bool:
    """Would the plan's queue, run from the true world, leave the row reading `word`?"""
    row = _plan_row(view, world, positions)
    return bool(view.plan) and all(row.get(p) == ch for p, ch in zip(positions, word)) and \
        all(p not in row for p in positions[len(word):])


def _letters_short(view: View, word: str, world: WorldState, positions: list[str]) -> set[str]:
    """The letters the plan leaves unspelled: of the positions the word wants, those the plan's row
    does not give the right letter."""
    row = _plan_row(view, world, positions)
    return {ch for p, ch in zip(positions, word) if row.get(p) != ch}


def truth(ev, view: View, src: SpellTruth) -> dict:
    """Acceptable answers per question group (preferred first) and booleans for the plan-check and
    progress Nouls, from ground truth."""
    world = src.world()
    positions = positions_of(world)
    word = src.word_now(view.user_messages)
    T: dict = {"right_now": ["carry_on"], "in_plan_fix": ["none"], "route": ["stay_local"]}
    paused_for_hand = view.mode == "paused" and (view.pause_reason or "").startswith("a person")
    h = view.hand
    k = ev.kind
    if k == "user_text":
        ln = src.line(ev.data.get("text", ""))
        if ln.kind == "new_word":
            cur = view.current
            if cur is None:
                T["right_now"] = ["hold", "carry_on"] if view.state.arm.holding else ["carry_on", "hold"]
            else:
                T["right_now"] = ["hold"] if _step_wrong(cur, word, world, positions) else ["carry_on", "hold"]
            if view.mode == "paused":
                T["right_now"] = ["hold", "resume", "carry_on"]
            letters = [o.label for o in world.objects.values() if o.label]
            T["route"] = ["fast_llm", "capable_llm"] if spellable(word, letters) else ["ask_user", "fast_llm"]
        elif ln.kind == "wait":
            T["right_now"] = ["pause"] if view.mode != "paused" else ["carry_on", "pause"]
        elif ln.kind == "go_on":
            T["right_now"] = ["resume"] if view.mode in ("paused", "holding") else ["carry_on", "resume"]
        elif ln.kind == "reply":
            T["route"] = ["fast_llm", "capable_llm"]
    elif k == "scene_change":
        ch = ev.data.get("change", {})
        if ch.get("what") == "hand":
            if h is None:
                T["right_now"] = ["resume"] if paused_for_hand else ["carry_on"]
            elif h["band"] == "very close":
                T["right_now"] = ["back_off", "pause"]
            elif h["approaching"] or h["in_path"]:
                T["right_now"] = ["pause", "carry_on"] if view.mode == "paused" else ["pause"]
            else:
                T["right_now"] = ["resume", "carry_on"] if paused_for_hand else ["carry_on"]
        else:
            oid = ev.object or ch.get("object")
            st, step = object_step(view, oid) if oid else ("no step", None)
            o = world.objects.get(oid) if oid else None
            cur = view.current
            if cur is not None and cur.get("object") == oid and not view.grasped and ch.get("what") == "object_moved":
                T["right_now"] = ["re_target"]
            elif ch.get("what") == "object_gone":
                label = _label_of_gone(oid)
                if label is not None and label in word and st in ("pending", "running"):
                    have = Counter(x.label for x in world.objects.values() if x.label)
                    T["route"] = ["fast_llm", "capable_llm"] if have[label] >= word.count(label) else ["ask_user", "fast_llm"]
            elif ch.get("what") == "object_appeared":
                if o is not None and o.label in word and not _plan_spells(view, word, world, positions):
                    T["route"] = ["fast_llm", "capable_llm"]
            elif o is not None and o.where in positions:          # the person put it into a position
                T["route"] = ["fast_llm", "capable_llm"]
            elif step is not None and st == "done" and o is not None and o.where != step_goal(step)[1]:
                T["in_plan_fix"] = ["re_queue"]
    elif k == "step_done":
        for c in (view.plan or {}).get("done_when", []):
            if c["relation"] == "other":
                T[f"done.{c['id']}"] = True
    elif k == "step_failed":
        r = ev.data.get("reason", "")
        gone = ev.object is not None and ev.object not in world.objects
        if gone or "not_in_view" in r or "no longer in the world" in r:
            T.update(in_plan_fix=["skip"], route=["fast_llm", "capable_llm"])
        elif ("occupied" in r or "on top" in r) and view.runnable_other:
            T["in_plan_fix"] = ["another_step_first"]
        elif any(x in r for x in ("grasp_failed", "dropped", "stalled")) and view.tries.get(ev.step_id, 0) <= 2:
            T["in_plan_fix"] = ["retry"]
        else:                                                   # occupied, blocked, unreachable, anything else
            T["route"] = ["fast_llm", "capable_llm"]
        if h is not None and (h["approaching"] or h["in_path"] or h["band"] == "very close"):
            T["right_now"] = ["pause", "carry_on"]
    elif k == "plan_arrived":
        short = _letters_short(view, word, world, positions)     # a cube left out matters only if it could fill a gap
        for oid in world.objects:
            if view.plan and oid not in _mentioned(view.plan):
                lab = world.objects[oid].label
                T[f"check.{oid}"] = bool(lab) and lab in short
    elif k == "paused_idle":
        T["right_now"] = ["resume"] if paused_for_hand and h is None else ["carry_on"]
    elif k == "plan_done_unmet":
        T["route"] = ["fast_llm", "capable_llm"]
    return T


def _mentioned(plan: dict) -> set[str]:
    ids = set()
    for s in plan["steps"]:
        ids |= {s["object"], s["target"]}
    for c in plan["done_when"]:
        ids |= {c["object"], c["target"]}
    return ids


_GONE_RE = re.compile(r"letter_([A-Z])\d*$")


def _label_of_gone(oid: str | None) -> str | None:
    """A cube that is no longer in view carries its letter in its id (scene.letters)."""
    m = _GONE_RE.match(oid or "")
    return m.group(1) if m else None


def _shape(pick: str, opts: list[str], conf: float = 1.0) -> dict:
    n = len(opts)
    pm = conf * (1 - 1 / n) + 1 / n if n > 1 else 1.0
    rest = (1 - pm) / (n - 1) if n > 1 else 0.0
    return {"choice": pick, "probabilities": {o: round(pm if o == pick else rest, 3) for o in opts}, "confidence": round(conf, 3)}


class SpellJev:
    """The DecisionBackend: every question answered from truth(), full confidence, LATENCY_MS."""
    live = False
    name = "mock-spell"

    def __init__(self, src: SpellTruth, latency_ms: float = LATENCY_MS):
        self.src, self.latency_ms = src, latency_ms
        self.records: list[dict] = []

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        ev, view = ref
        T = truth(ev, view, self.src)
        out = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                opts = list(q["criteria"])
                out[key] = _shape(next((t for t in T.get(key, []) if t in opts), opts[0]), opts)
            else:
                out[key] = {"noul": 0.97 if T.get(key, False) else 0.03}
        self.records.append({"t": time.time(), "event": ev.kind, "answers": {k: v.get("choice", v.get("noul")) for k, v in out.items()}})
        tok = int(len(json.dumps({"state": state, "questions": questions})) / 3.5)
        return {"answers": out, "latency_ms": self.latency_ms, "in_tok": tok, "meta": {"mock": self.name}}


def mock_decider(ep: Episode, world: Callable[[], WorldState], arms: list[str], gates: Gates | None = None):
    """The offline fast loop: JevDecider (the brain's request building, combine and gates) over
    SpellJev, which sleeps its latency; one per arm behind mocks.PerArmDecider when there are several."""
    src = SpellTruth(ep, world)

    def one() -> JevDecider:
        d = JevDecider(TimedJev(SpellJev(src)), gates or Gates())
        d.name = "mock"
        return d
    if len(arms) == 1:
        return one()
    return PerArmDecider({a: one() for a in arms}, arms[0])
