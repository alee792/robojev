"""The rules the loop enforces in code, whatever the models say (docs/v2.md "Rules that hold everywhere").

Priority: STOP > the robot's safety filter > right-now answer > route > in-plan fix > plan.

- STOP: the button, or a bare typed "stop" (`is_stop`, from e12v2: "don't stop" is text), calls the
  server's stop() at once, on the caller's thread, and ends the episode. Only this path calls stop().
- Models can only pause: a right-now answer maps to hold, pause, resume or retarget
  (RIGHT_NOW_ACTION), never stop. back_off is a pause: the protocol has no back-off motion, and the
  server's own stop distance for hands applies regardless.
- The in-plan fix applies only when the route is stay_local and the weakest confidence among the
  route and the fix clears the gate (0.8; e12v2 measured it catching 98% of wrong answers). e12v2's
  combine() already does this; `enforce_gate` re-applies it in the loop so a decider that does not
  (another backend, a control) cannot bypass it. A confidence the decider did not report counts as 0.
- Replan loop: a new plan sent back to the LLM by its plan_arrived decision is a rejection; at
  `limit` (3) rejections since the user last spoke, later plans run as they come, undecided, until
  the user says something new (ReplanGuard).
- A loop of steps (the same two alternating, one failing 3 times) escalates to the capable LLM
  (`LoopDetector`, from e12v2's harness).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .core.data import Combined

STAY_LOCAL_GATE = 0.8
STOP_RE = re.compile(r"^\s*(stop|stop!|stop now|stop it|stop stop)\W*$", re.I)


def is_stop(text: str) -> bool:
    """Typed "stop" is handled in code (docs/v2.md). Only a bare stop: "don't stop" or "stop at the tray" are text."""
    return bool(STOP_RE.match(text))


class LoopDetector:
    """A loop is a failure: the same two plan steps alternating more than twice escalates."""

    def __init__(self, repeats: int = 3):
        self.repeats, self.hist = repeats, []

    def record(self, sig) -> bool:
        self.hist.append(sig)
        n = 2 * self.repeats
        h = self.hist[-n:]
        if len(h) == n and h[0] != h[1] and all(h[i] == h[i % 2] for i in range(n)):
            self.hist.clear()
            return True
        return False


# right-now answer -> what the loop does to the arm (None = nothing). No answer maps to stop.
RIGHT_NOW_ACTION: dict[str, str | None] = {
    "carry_on": None, "hold": "holding", "pause": "paused", "back_off": "paused",
    "resume": "running", "re_target": "retarget",
}


def enforce_gate(c: Combined, gate: float = STAY_LOCAL_GATE) -> Combined:
    """Keep a decision local only on confidence: stay_local with min(route, fix) confidence below
    `gate` goes to the fast LLM instead, with no fix applied."""
    if c.route != "stay_local":
        return c
    weakest = min(c.conf.get("route", 0.0), c.conf.get("in_plan_fix", 0.0))
    if weakest >= gate:
        return c
    return replace(c, route="fast_llm", fix="none", reason=f"weakest confidence {weakest:.2f} below the {gate} gate")


@dataclass
class ReplanGuard:
    """Counts new plans sent back to the LLM since the user last spoke."""
    limit: int = 3
    rejections: int = 0
    trusting: bool = False          # a replan loop was found: run new plans without deciding on them

    def user_spoke(self) -> None:
        """A new request from the user: judge new plans again."""
        self.rejections, self.trusting = 0, False

    def rejected(self) -> bool:
        """A plan_arrived decision sent its plan back. -> True when this starts a replan loop."""
        self.rejections += 1
        if self.rejections >= self.limit and not self.trusting:
            self.trusting = True
            return True
        return False


__all__ = ["LoopDetector", "RIGHT_NOW_ACTION", "ReplanGuard", "STAY_LOCAL_GATE", "enforce_gate", "is_stop"]
