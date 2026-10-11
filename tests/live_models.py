"""Live models for the slice (tests/test_slice.py), chosen by environment variables:

    ROBOJEV_DECIDER = mock | live | jev | decisions   (default mock: the test run needs no keys; `live`
                                                   is the default live decider, Jev, per docs/live-results.md)
    ROBOJEV_PLANNER = mock | openai               (default mock)
    ROBOJEV_SHADOW  = 1    with a live decider, also send each decision request to the other live
                           backend in the background (not acted on) so both answer the same events
    ROBOJEV_LIVE_DIR       where the recorder writes (default experiments/results/live)

Anything live needs keys (TYPESAFE_API_KEY / OPENAI_API_KEY, env or .env) and is never part of the
default test run. Every live decision is logged with the oracle's truth for that moment (e12v2's
eval.oracle.truth, read from the server; the brain never sees it) so wrong answers can be counted.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path

from robojev.brain.core.combine import Gates, JevDecider
from robojev.brain.eval import oracle as O
from robojev.brain.mocks import StubTruth, oracle_models, oracle_scenario

DEFAULT_LIVE_DECIDER = "jev"      # the live recommendation (docs/live-results.md); the Decisions API stays selectable


def decider_name(value: str | None) -> str:
    """The decider an environment value selects: unset -> mock; "live" -> the default live decider."""
    name = value or "mock"
    if name == "live":
        name = DEFAULT_LIVE_DECIDER
    if name not in ("mock", "jev", "decisions"):
        raise ValueError(f"ROBOJEV_DECIDER must be mock, live, jev or decisions, not {value!r}")
    return name


DECIDER = decider_name(os.environ.get("ROBOJEV_DECIDER"))
PLANNER = os.environ.get("ROBOJEV_PLANNER", "mock")
SHADOW = os.environ.get("ROBOJEV_SHADOW", "") == "1"
LIVE = DECIDER != "mock" or PLANNER != "mock"
LIVE_DIR = Path(os.environ.get("ROBOJEV_LIVE_DIR", Path(__file__).resolve().parents[1] / "experiments/results/live"))
assert PLANNER in ("mock", "openai"), PLANNER


def tag() -> str:
    return f"{DECIDER}-{PLANNER}"


def _backend(name: str):
    if name == "jev":
        from robojev.brain.backends.jev import JevHTTP
        return JevHTTP(max_requests=200)
    from robojev.brain.backends.decisions_api import DecisionsAPI
    return DecisionsAPI(max_requests=200)


class Logged:
    """A DecisionBackend in front of a live one: logs every request with its answer, the oracle's
    truth at the moment it was asked, and (SHADOW) the other backend's answer to the same request."""

    def __init__(self, primary, truth, sc, shadow=None):
        self.primary, self.truth_src, self.sc, self.shadow = primary, truth, sc, shadow
        self.live = True
        self.records: list[dict] = []
        self.pool = ThreadPoolExecutor(max_workers=2) if shadow else None

    def answer(self, state, questions, ref=None):
        ev, view = ref
        try:
            truth = O.truth(ev, view, self.sc, self.truth_src)
        except Exception as e:                   # the oracle can't judge every live situation; say so
            truth = {"error": repr(e)[:200]}
        fut = self.pool.submit(self.shadow.answer, state, questions) if self.shadow else None
        r = self.primary.answer(state, questions)
        self.records.append({"t": time.time(), "event": ev.kind, "event_text": ev.text, "state": state, "questions": questions,
                             "truth": truth, "primary": r, "_shadow": fut})
        return r

    def finish(self) -> list[dict]:
        for rec in self.records:
            fut = rec.pop("_shadow", None)
            if fut is not None:
                try:
                    rec["shadow"] = fut.result(timeout=10)
                except Exception as e:
                    rec["shadow"] = {"answers": None, "meta": {"error": repr(e)[:200]}}
        if self.pool:
            self.pool.shutdown(wait=False)
        return self.records


def models(server, versions, utts=None):
    """-> (decider, planner backends, the Logged backend or None). Mock parts are e12v2's oracle mocks,
    exactly as the mock slice uses them."""
    decider, llm = oracle_models(server, versions, utts)
    logged = None
    if DECIDER != "mock":
        other = {"jev": "decisions", "decisions": "jev"}[DECIDER]
        primary = _backend(DECIDER)
        shadow = _backend(other) if SHADOW else None
        for b in (primary, shadow):
            if b is not None:
                b.warm()
        logged = Logged(primary, StubTruth(server), oracle_scenario(versions, utts), shadow)
        decider = JevDecider(logged, Gates())            # the same request building, combine() and gates as the mock
    if PLANNER == "openai":
        from robojev.brain.backends.openai_planner import OpenAIPlanner
        p = OpenAIPlanner(effort="low", max_calls=30)
        llm = {"fast_llm": p, "capable_llm": p}           # no CatalogLLM: a real LLM writes the catalog's names
    return decider, llm, logged


def jsonable(x):
    if is_dataclass(x) and not isinstance(x, type):
        return jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def record_episode(recorder, name: str, r, logged, measured: dict) -> None:
    """Everything a later reader needs next to the server's frames and events."""
    recorder.note(kind="episode", scenario=name, decider=DECIDER, planner=PLANNER, shadow=SHADOW, outcome=r.outcome,
                  measured=jsonable(measured), events=dict(r.events), steps=[r.steps_started, r.steps_done, r.steps_failed],
                  stale_plans=r.stale_plans, failed_plans=r.failed_plans, loops=r.loops)
    for d in r.decisions:
        recorder.note(kind="decision", rec=jsonable(d))
    for p in r.llm:
        recorder.note(kind="llm", rec=jsonable(p))
    for rec in (logged.finish() if logged else []):
        recorder.note(kind="decision_request", rec=jsonable(rec))
    for line in r.log:
        recorder.note(kind="brain_log", rec=jsonable(line))


def out_path(test_name: str) -> Path:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_DIR / f"slice_{tag()}_{test_name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def dumps(x) -> str:
    return json.dumps(jsonable(x))
