"""The deciders and planners the battery can run, by name.

    deciders   mock        the offline stand-in (decider.SpellJev behind the brain's JevDecider)
               rules       controls.RulesOnly
               always_llm  controls.AlwaysLLM
               jev         Jev over HTTP (brain.backends.jev; TYPESAFE_API_KEY)
               decisions   OpenAI's Decisions API (brain.backends.decisions_api; OPENAI_API_KEY)
    planners   mock        spell.SpellLLM, timed, in the catalog's vocabulary (offline only)
               openai      brain.backends.openai_planner (OPENAI_API_KEY)

The live ones are imported only when asked for, so the offline battery needs no keys and no network.
"""
from __future__ import annotations

from collections.abc import Callable

from robojev.brain.core.combine import Gates, JevDecider
from robojev.brain.mocks import CatalogLLM, TimedLLM
from robojev.protocol import RobotServer, WorldState

from .controls import AlwaysLLM, RulesOnly
from .decider import mock_decider
from .generate import Episode
from .spell import SpellLLM

DECIDERS = ("mock", "rules", "always_llm", "jev", "decisions")
PLANNERS = ("mock", "openai")
LIVE_DECIDERS = ("jev", "decisions")
LIVE_PLANNERS = ("openai",)


def make_decider(name: str, ep: Episode, truth: Callable[[], WorldState], arms: list[str]):
    if name == "mock":
        return mock_decider(ep, truth, arms)
    if name == "rules":
        return RulesOnly()
    if name == "always_llm":
        return AlwaysLLM()
    if name == "jev":
        from robojev.brain.backends.jev import JevHTTP
        backend = JevHTTP(max_requests=400)
    elif name == "decisions":
        from robojev.brain.backends.decisions_api import DecisionsAPI
        backend = DecisionsAPI(max_requests=400)
    else:
        raise ValueError(f"decider must be one of {DECIDERS}, not {name!r}")
    backend.warm()
    d = JevDecider(backend, Gates())
    d.name = name
    return d


def make_planner(name: str, server: RobotServer, seed: int = 0) -> dict[str, object]:
    """-> the brain's planner backends by route."""
    if name == "mock":
        llm = CatalogLLM(TimedLLM(SpellLLM(server, seed=seed)))
        return {"fast_llm": llm, "capable_llm": llm}
    if name == "openai":
        from robojev.brain.backends.openai_planner import OpenAIPlanner
        p = OpenAIPlanner(effort="low", max_calls=60)
        return {"fast_llm": p, "capable_llm": p}
    raise ValueError(f"planner must be one of {PLANNERS}, not {name!r}")


def is_live(decider: str, planner: str) -> bool:
    return decider in LIVE_DECIDERS or planner in LIVE_PLANNERS
