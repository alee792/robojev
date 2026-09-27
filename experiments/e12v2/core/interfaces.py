"""The seams between the design logic and everything it runs against.

World        perception: the current world state. Changes the robot didn't cause are worked out by
             core/changes.py from successive states, so every World gets them the same way.
UserChannel  typed text from the user, and questions back to them.
Skill        one motor capability (docs/v2.md "Skills"): precondition / start / step / status, plus the
             literal phase text Jev needs and two hooks the harness uses (re-target, safe point).
PlanFormat   how a plan is represented; the step list now, room for a rules format later.
LLMBackend   one structured-output LLM call (OpenAI, or a mock).
DecisionBackend  one Jev request (Jev over HTTP, or a mock).
Decider      turns an event into a Combined decision (Jev + combiner, or a control).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .data import Command, Event, Geometry, WorldState


@runtime_checkable
class World(Protocol):
    geometry: Geometry

    def advance(self, t: int) -> None:
        """Let the outside world move on one tick (people act, hands move)."""

    def state(self) -> WorldState:
        """What perception reports now."""

    def execute(self, cmd: Command) -> None:
        """Carry out one command that already passed the safety filter."""

    def names(self) -> dict:
        """object/place id -> display name (for literal text)."""


@runtime_checkable
class UserChannel(Protocol):
    def poll(self, t: int) -> list[str]:
        """Text the user typed since the last poll."""

    def ask(self, t: int, question: str) -> None:
        """Put a question to the user; any answer arrives later through poll()."""

    def stop_pressed(self, t: int) -> bool:
        """The STOP button."""


@runtime_checkable
class Skill(Protocol):
    name: str

    def precondition(self, state: WorldState, step: dict) -> str | None:
        """None if the step can run now, else why not (literal)."""

    def start(self, state: WorldState, step: dict) -> None: ...

    def step(self, state: WorldState) -> Command | None:
        """A short chunk of motion; the harness passes it through the safety filter."""

    def feedback(self, executed: bool, reason: str | None) -> None:
        """Whether the last command ran or the safety filter refused it."""

    def status(self) -> tuple[str, str | None]:
        """("running" | "done" | "failed", reason)."""

    def phase_text(self, state: WorldState) -> str:
        """Where the arm is in this step, literally ("lowered at tray slot 3 holding block 5, not yet released")."""

    def heading(self, state: WorldState) -> tuple | None:
        """Where the arm is heading in this step (x, y), for "in the arm's path" facts."""

    def retarget(self, state: WorldState) -> None:
        """Aim at where the step's object is now."""

    def safe_point(self, state: WorldState) -> Command | None:
        """One chunk toward a safe point to hold at (None when already there)."""


class PlanFormat(Protocol):
    """A plan representation. `StepListFormat` (core/plan.py) is the only one built; a rules format
    (docs/v2.md "Alternative to explore") would implement the same four methods."""
    name: str

    def schema(self, ids: dict) -> dict: ...

    def diff_schema(self, ids: dict) -> dict: ...

    def validate(self, plan: dict, state: WorldState, holding: str | None) -> list[str]: ...

    def apply_diff(self, plan: dict, diff: dict) -> dict: ...


class LLMBackend(Protocol):
    def call(self, req, ref=None):
        """-> an object with raw, latency_ms, in_tok, out_tok, error, model (e13 LLMResult shape)."""


class DecisionBackend(Protocol):
    live: bool

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        """-> {"answers": {...} | None, "latency_ms": float, "in_tok": int, "meta": {...}}"""


class Decider(Protocol):
    name: str

    def decide(self, event: Event, view) -> dict:
        """-> {"combined": Combined, "latency_ms": float, "request": dict|None, "answers": dict|None, "in_tok": int}"""
