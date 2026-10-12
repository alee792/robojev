"""The offline planner for spelling: SpellLLM, an LLMBackend that writes plans from the request it is
sent (the user's words, the letters in view with their labels, the positions), with no model and no
ground truth. OFFLINE ONLY: the stand-in for the OpenAI planner, so the battery runs with no keys; on
the Mac the real planner takes its place (--planner openai).

The plan for a word (`spell_plan`): position by position, left to right, a cube whose label is that
letter goes into that position, by pick_and_place. A cube already in its position stays. A cube in a
position the word does not want it in is cleared first (straight to a position that wants it and is
free, else to "table"), and so is anything in a position past the word's end. A held cube is dealt
with first (the planner rule). A letter with no cube in view is left out and the reading says so;
what to do about that is the brain's business (the mock decider routes it to the user: decider.py).

The word is the newest uppercase word in the user's messages, else the task's: a model would read
the sentence; this mock reads the capitals, which is why the generated phrasings keep the word in
capitals and nothing else in them capitalised.

Plans are written in the brain's core vocabulary (move_object, target); mocks.CatalogLLM puts them
into the catalog's names and the request's step fields, as it does for e12v2's mock.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field

from robojev.brain.core import plan as P
from robojev.brain.mocks import LLMReply, give_arms
from robojev.protocol import RobotServer

WORD_RE = re.compile(r"\b([A-Z]{2,})\b")
NOT_WORDS = frozenset({"OK", "STOP", "TV", "AI"})
SLOT_RE = re.compile(r"^slot_(\d+)$")


def word_in(text: str) -> str | None:
    """The last uppercase word of two or more letters in `text`, if any."""
    words = [w for w in WORD_RE.findall(text or "") if w not in NOT_WORDS]
    return words[-1] if words else None


def asked_word(task: str, messages: list[str]) -> str:
    """The word asked for now: from the newest message that names one, else the task."""
    for m in reversed(messages):
        w = word_in(m)
        if w:
            return w
    return word_in(task) or ""


@dataclass(frozen=True)
class Cube:
    id: str
    label: str | None
    where: str                              # position id | "table" | "gripper" | "gripper:<arm>" | ...


@dataclass
class Table:
    """What the planner reads of the world: the cubes, the positions left to right, whether there is
    a "table" place to put things down on, what the arm holds (one arm) or each arm holds."""
    cubes: list[Cube]
    positions: list[str]
    table: str | None
    holding: dict[str, str | None] = field(default_factory=dict)   # arm -> cube id

    def occupant(self, pos: str) -> Cube | None:
        return next((c for c in self.cubes if c.where == pos), None)


def read_request(body: dict) -> Table:
    """The planner request's `world` (schema.world_view or world_view_arms) as a Table."""
    w = body["world"]
    cubes = [Cube(o["id"], o.get("label"), o["where"]) for o in w["objects"]]
    slots = sorted((int(m.group(1)), p["id"]) for p in w["places"] if p["kind"] == "slot"
                   for m in [SLOT_RE.match(p["id"])] if m)
    table = next((p["id"] for p in w["places"] if p["kind"] == "table"), None)
    if "arms" in w:
        holding = {a: (h["holding"] if h["holding"] != "nothing" else None) for a, h in w["arms"].items()}
    else:
        h = w.get("arm", {}).get("holding", "nothing")
        holding = {"arm": None if h == "nothing" else h}
    return Table(cubes, [s for _, s in slots], table, holding)


def spell_plan(word: str, table: Table, reading: str | None = None) -> tuple[dict, list[str]]:
    """-> (a full plan in core names, the letters of `word` with no cube in view). See the module doc."""
    word = word.upper()
    pos = table.positions[:len(word)]
    occ = {c.where: c for c in table.cubes if c.where in table.positions}
    assigned: dict[int, str] = {}
    used: set[str] = set()
    for i, ch in enumerate(word[:len(pos)]):                       # cubes already right stay
        c = occ.get(pos[i])
        if c is not None and c.label == ch:
            assigned[i], used = c.id, used | {c.id}
    missing: list[str] = list(word[len(pos):])                     # letters past the last position

    def rank(c: Cube) -> int:                                      # the held cube, then cubes that must move anyway, then free ones
        return 0 if c.where.startswith("gripper") else 1 if c.where in table.positions else 2
    for i, ch in enumerate(word[:len(pos)]):
        if i in assigned:
            continue
        cands = sorted((c for c in table.cubes if c.label == ch and c.id not in used), key=rank)
        if cands:
            assigned[i], used = cands[0].id, used | {cands[0].id}
        else:
            missing.append(ch)
    target = {cid: pos[i] for i, cid in assigned.items()}
    where = {c.id: c.where for c in table.cubes}
    steps: list[dict] = []

    def add(o: str, t: str) -> None:
        steps.append({"id": f"s{len(steps) + 1}", "skill": "move_object", "object": o, "target": t, "direction": P.NONE})
        where[o] = t

    def free(p: str) -> bool:
        return not any(w == p for w in where.values())

    def park(o: str) -> None:
        """Somewhere out of the way: the table, else a free position past the word."""
        if table.table is not None:
            add(o, table.table)
            return
        spare = next((p for p in table.positions[len(word):] if free(p)), None)
        if spare is not None:
            add(o, spare)
    for held in table.holding.values():                            # a held cube is put down first
        if held and held in where:
            t = target.get(held)
            if t is not None and free(t):
                add(held, t)
            else:
                park(held)
    for _ in range(len(table.cubes) + 1):                          # clear positions of what does not belong there
        wrong = [c.id for c in table.cubes if where[c.id] in table.positions and target.get(c.id) != where[c.id]]
        if not wrong:
            break
        direct = next((o for o in wrong if target.get(o) is not None and free(target[o])), None)
        if direct is not None:
            add(direct, target[direct])
        else:
            park(wrong[0])
    for i in range(len(pos)):                                      # fill, left to right
        cid = assigned.get(i)
        if cid is not None and where[cid] != pos[i]:
            add(cid, pos[i])
    done = [{"id": f"d{i + 1}", "text": f"{cid} is in {pos[i]}", "object": cid, "relation": "in", "target": pos[i]}
            for i, cid in sorted(assigned.items())]
    if not done:
        done = [{"id": "d1", "text": f"no letter of {word} is in view; there is nothing to place", "object": P.NONE,
                 "relation": "other", "target": P.NONE}]
    text = reading or f"spell {word}"
    if missing:
        text += f" (no cube in view for the letter{'s' if len(missing) > 1 else ''} {', '.join(sorted(set(missing)))}; placing the rest)"
    return {"reading": text, "steps": steps, "done_when": done, "constraints": []}, missing


class SpellLLM:
    """An LLMBackend for the spelling task, from the request alone (module doc). With several arms
    the one-arm steps go through mocks.give_arms, which reads the server's world for reach, as
    e12v2's pair mock does. Reports a lognormal latency around `median_ms` (wrap in mocks.TimedLLM
    to wait it out). `calls` counts requests; `plans` keeps every plan written."""

    def __init__(self, server: RobotServer, seed: int = 0, median_ms: float = 300.0):
        self.server, self.median_ms = server, median_ms
        self.rng = random.Random(seed)
        self.calls = 0
        self.plans: list[dict] = []
        self.missing: list[list[str]] = []

    def call(self, req, ref=None) -> LLMReply:
        self.calls += 1
        body = json.loads(req.input.split("\n\nYour previous answer was:")[0])
        table = read_request(body)
        word = asked_word(body.get("task", ""), body.get("user_messages", []))
        plan, missing = spell_plan(word, table)
        m = self.server.manifest()
        if len(m.arms) > 1:
            plan["steps"] = give_arms(plan["steps"], self.server.world(), m)
        self.plans.append(plan)
        self.missing.append(missing)
        out: dict = plan
        if req.kind != "plan":
            ctx = ref[0] if ref else None
            ep = ctx["episode"] if ctx else None
            if ep is not None:
                out = P.make_diff(ep.plan, ep.queue, plan, reading=plan["reading"])
            else:
                out = P.make_diff({"steps": [], "done_when": [], "constraints": []}, [], plan, reading=plan["reading"])
        lat = math.exp(self.rng.gauss(math.log(self.median_ms), 0.35))
        return LLMReply(json.dumps(out), lat, model="mock-spell")
