"""The scenario truth e12v2's mocks read (experiments/e12v2/eval/scenarios.py, the data types only).

e12v2's scenarios build a toy world and a scripted person; here the world is a robot server and the
user is scripted by robojev.brain.mocks, so only what the oracle and the mocks read is kept: each
person line's truth (Utt), the task versions a Scenario maps user messages to (goal_for), the sort
task and its goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .oracle import Goal


@dataclass
class Utt:
    kind: str                     # correction | chatter | wait | go_on
    version: int | None = None    # corrections: the task version it switches to
    clarify: str | None = None    # what the person answers if asked what they meant

    @property
    def expect_change(self) -> bool:
        return self.kind != "chatter"


@dataclass
class Scenario:
    name: str
    about: str
    task: str
    world: Any                    # e12v2's SimWorld; None behind a robot server
    person: Any                   # e12v2's scripted Person; None behind a robot server
    versions: list
    utts: dict = field(default_factory=dict)
    held_out: bool = False
    interference: bool = False     # counts for pass criterion 2 (interference / correction)

    def version_for(self, messages: list) -> int:
        v = 0
        for m in messages:
            u = self.utts.get(m)
            if u is not None and u.version is not None:
                v = u.version
            elif u is None:
                for u2 in self.utts.values():   # an answer to "what did you mean?" carries its correction's version
                    if u2.clarify == m and u2.version is not None:
                        v = u2.version
        return v

    def goal_for(self, messages: list) -> Goal:
        return self.versions[self.version_for(messages)]

    def said(self) -> list:
        return [b.text for b in sorted(self.person.beats, key=lambda b: b.fired_t or 0) if b.text and b.fired_t is not None]

    def goal_now(self) -> Goal:
        return self.goal_for(self.said())

    def answer(self, question: str) -> str:
        said = self.said()
        for m in reversed(said):
            u = self.utts.get(m)
            if u and u.clarify:
                return u.clarify
        return "yes, just carry on as you were"


def sort_goal(ids_by_number: dict, desc: bool = False, constraints=()) -> Goal:
    ranked = sorted(ids_by_number, key=lambda b: ids_by_number[b], reverse=desc)
    return Goal({b: f"tray_slot_{k}" for k, b in enumerate(ranked, 1)}, list(constraints))


SORT_TASK = "Line the numbered blocks up in the tray, lowest number on the left."
