"""Offline stand-ins for the brain's models and user: no API keys, no network. The Brain never
imports this module; tests and spike runs do.

e12v2's mock Jev and mock LLM are reused as they are. They read an evaluation oracle (the true world,
a goal per task version, the truth about each line the user types), so the adapters here give them
that from the stub server, and make them behave as models would behind this brain:
  - StubTruth      the stub's world() as e12v2's cm WorldState (`.truth()`, what the mocks read).
  - TimedJev/LLM   the mocks report a latency but return at once; these sleep it, in the worker
                   thread, so a real clock sees decisions take ~150 ms and plans seconds.
  - CatalogLLM     e12v2's mock LLM writes e12v2's step names and fields ("move_object", no push
                   distance); a real LLM given the brain's schema writes the catalog's. This renames
                   steps and fills fields the schema has and the mock lacks with null.
  - ScriptedLLM    fixed answers in turn, for robots the oracle cannot plan for (the push-only arm).
  - ScriptedUser   lines typed at times or when conditions hold; the STOP button as a line.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from e12v2.core.combine import Gates, JevDecider
from e12v2.core.data import WorldState
from e12v2.eval.mocks import MockJev, MockLLM
from e12v2.eval.oracle import Goal
from e12v2.eval.scenarios import Scenario, Utt

from ..protocol import RobotServer
from .adapt import core_state, geometry
from .steps import to_catalog

PRESS_STOP = "<the STOP button>"      # a ScriptedUser line that presses STOP instead of typing


class StubTruth:
    """Ground truth for e12v2's oracle, read from the server (a stub has no perception error)."""

    def __init__(self, server: RobotServer, arm: str = "arm_0", lookahead_s: float = 0.2):
        spec = next(a for a in server.manifest().arms if a.id == arm)
        self.server, self.arm, self.lookahead_s = server, arm, lookahead_s
        self.geom = geometry(spec, lookahead_s)

    def truth(self) -> WorldState:
        return core_state(self.server.world(), self.arm, self.geom, self.lookahead_s)


def oracle_scenario(versions: Sequence[Goal], utts: dict[str, Utt] | None = None, task: str = "") -> Scenario:
    """e12v2's Scenario, for what its mocks read: goal_for(user messages) and the utterance truths.
    Its world and scripted person belong to e12v2's toy world and are not used here."""
    return Scenario(name="stub", about="", task=task, world=None, person=None, versions=list(versions), utts=dict(utts or {}))


class TimedJev:
    """A DecisionBackend that takes as long as it says it does."""

    def __init__(self, backend):
        self.backend, self.live = backend, False

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        r = self.backend.answer(state, questions, ref=ref)
        time.sleep(r.get("latency_ms", 0.0) / 1000.0)
        return r


class TimedLLM:
    """An LLMBackend that takes as long as it says it does."""

    def __init__(self, backend):
        self.backend = backend

    def call(self, req, ref=None):
        r = self.backend.call(req, ref=ref)
        time.sleep(r.latency_ms / 1000.0)
        return r


@dataclass
class LLMReply:
    """An LLMBackend result in e12v2's shape."""
    raw: str | None
    latency_ms: float
    in_tok: int = 0
    out_tok: int = 0
    error: str | None = None
    model: str = "scripted"


class CatalogLLM:
    """Puts an e12v2-vocabulary LLM's answers into the catalog vocabulary of the request's schema."""

    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def _step_fields(schema: dict) -> list[str]:
        props = schema["properties"]
        steps = props.get("steps") or props.get("new_steps")
        return list(steps["items"]["properties"])

    def call(self, req, ref=None):
        r = self.backend.call(req, ref=ref)
        if r.error is not None or not r.raw:
            return r
        try:
            out = json.loads(r.raw)
        except json.JSONDecodeError:
            return r
        fields = self._step_fields(req.schema)
        key = "steps" if "steps" in out else "new_steps"
        out[key] = [{f: None for f in fields} | to_catalog(s) for s in out.get(key, [])]
        return LLMReply(json.dumps(out), r.latency_ms, r.in_tok, r.out_tok, None, getattr(r, "model", "mock"))


class ScriptedLLM:
    """Answers in turn (the last one repeats). `requests` records every request received."""

    def __init__(self, answers: Sequence[dict], latency_ms: float = 100.0):
        self.answers, self.latency_ms = list(answers), latency_ms
        self.requests: list = []

    def call(self, req, ref=None):
        self.requests.append(req)
        out = self.answers[min(len(self.requests), len(self.answers)) - 1]
        time.sleep(self.latency_ms / 1000.0)
        return LLMReply(json.dumps(out), self.latency_ms)


def oracle_decider(server: RobotServer, versions: Sequence[Goal], utts: dict[str, Utt] | None = None, *,
                   arm: str = "arm_0", seed: int = 0, lookahead_s: float = 0.2) -> JevDecider:
    """e12v2's real JevDecider (request building, combine(), its gates) over its mock Jev with no
    errors and full confidence (the oracle arm), taking Jev's measured latency."""
    sc, truth = oracle_scenario(versions, utts), StubTruth(server, arm, lookahead_s)
    return JevDecider(TimedJev(MockJev(sc, truth, error=0.0, conf_noise=0.0, seed=seed)), Gates())


def oracle_models(server: RobotServer, versions: Sequence[Goal], utts: dict[str, Utt] | None = None, *,
                  arm: str = "arm_0", llm_median_ms: float = 300.0, seed: int = 0,
                  lookahead_s: float = 0.2) -> tuple[JevDecider, dict]:
    """oracle_decider(), and e12v2's mock LLM for both planner routes, both reading the stub.
    -> (decider, planner backends)."""
    sc, truth = oracle_scenario(versions, utts), StubTruth(server, arm, lookahead_s)
    llm = CatalogLLM(TimedLLM(MockLLM(sc, truth, seed=seed, median_ms=llm_median_ms)))
    decider = oracle_decider(server, versions, utts, arm=arm, seed=seed, lookahead_s=lookahead_s)
    return decider, {"fast_llm": llm, "capable_llm": llm}


@dataclass
class Line:
    """One thing the user does: typed `text` (or PRESS_STOP) at `when` s after attach, or when `when()` holds."""
    when: float | Callable[[], bool]
    text: str


class ScriptedUser:
    """A UserChannel that plays its lines in order, each when its time or condition comes."""

    def __init__(self, lines: Iterable[Line | tuple] = (), poll_s: float = 0.02):
        self.lines = [x if isinstance(x, Line) else Line(*x) for x in lines]
        self.poll_s = poll_s
        self.said: list[tuple[float, str]] = []
        self.asked: list[str] = []
        self._task: asyncio.Task | None = None

    def attach(self, say: Callable[[str], None], press_stop: Callable[[], None]) -> None:
        self._say, self._press_stop, self._t0 = say, press_stop, time.monotonic()
        self._task = asyncio.get_running_loop().create_task(self._play())

    async def _play(self) -> None:
        for line in self.lines:
            while not (line.when() if callable(line.when) else time.monotonic() - self._t0 >= line.when):
                await asyncio.sleep(self.poll_s)
            self.said.append((time.monotonic() - self._t0, line.text))
            if line.text == PRESS_STOP:
                self._press_stop()
            else:
                self._say(line.text)

    def ask(self, question: str) -> None:
        self.asked.append(question)
