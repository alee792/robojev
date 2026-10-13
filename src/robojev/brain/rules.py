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
  A routine step_done (no fix, no plan-check problem) is the one exception, with core/combine.py:
  keeping local there only starts the next planned step, so the choice stands at any confidence.
- Replan loop: a new plan sent back to the LLM by its plan_arrived decision is a rejection; at
  `limit` (3) rejections since the user last spoke, later plans run as they come, undecided, until
  the user says something new (ReplanGuard).
- A loop of steps (the same two alternating, one failing 3 times) escalates to the capable LLM
  (`LoopDetector`, from e12v2's harness).
- Transient commands ("wait", "hang on", "go on", ...) are events, never part of the task
  (docs/v2.md "The decision"). `is_transient` says whether a typed line is one, so it is kept out of
  the user's task-level words shown to the decider (core/decision.View.user_words); `transient_mentions`
  says whether a plan's restated task repeats one, so the loop keeps the previous reading instead
  (docs/live-results.md H5: a wait that reached the planner came back inside the restatement, and
  nothing cleared it).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .core.combine import routine
from .core.data import Combined

STAY_LOCAL_GATE = 0.8
STOP_RE = re.compile(r"^\s*(stop|stop!|stop now|stop it|stop stop)\W*$", re.I)

# Momentary commands: a pause, a wait, a resume, an acknowledgement. Each is matched as whole words.
# ("a second" is left out: in task language it is an ordinal, "a second tower"; "wait a second" is still
# transient through "wait" and the fillers below.)
TRANSIENT_PHRASES = ("wait", "wait up", "hang on", "hold on", "hold it", "hold up", "one sec", "one second", "a sec",
                     "a moment", "one moment", "just a moment", "pause", "stop", "go on", "go ahead",
                     "carry on", "keep going", "continue", "resume", "proceed", "as you were")
# Words that may surround a transient command without making it about the task.
_FILLERS = {"ok", "okay", "oh", "ah", "um", "hmm", "please", "now", "robot", "a", "an", "the", "one", "sec", "second",
            "seconds", "moment", "minute", "bit", "there", "just", "for", "me", "so", "then", "and", "thanks", "thank",
            "you", "yes", "yeah", "right", "good", "fine", "all", "again", "hey", "it", "that", "is", "s", "can", "we"}
TRANSIENT_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(TRANSIENT_PHRASES, key=len, reverse=True)) + r")\b", re.I)


def is_stop(text: str) -> bool:
    """Typed "stop" is handled in code (docs/v2.md). Only a bare stop: "don't stop" or "stop at the tray" are text."""
    return bool(STOP_RE.match(text))


def is_transient(text: str) -> bool:
    """A typed line that is only a momentary command ("hang on a second", "ok, go on"), not a word
    about the end result. "wait, put the red one on the left" is a correction: words remain once the
    transient phrases and fillers are taken out."""
    if not TRANSIENT_RE.search(text):
        return False
    rest = TRANSIENT_RE.sub(" ", text.lower())
    return not [w for w in re.findall(r"[a-z']+", rest) if w not in _FILLERS]


def transient_mentions(reading: str, said: list) -> list[str]:
    """What a plan's restated task (`reading`) carries of the user's transient lines in `said`: once
    the user has typed one, each such line quoted in the reading, and each transient phrase the
    reading uses as whole words (the planner paraphrases: "hang on a sec" came back as "wait a moment
    as requested" and "pause the handover" in the September logs). With no transient line typed, a
    reading is never flagged ("continue lining the blocks up" is the planner's framing, not a leak).
    Empty = the reading holds the end result only."""
    out: list[str] = []
    low = (reading or "").lower()
    lines = [x for x in said if is_transient(x)]
    if not low or not lines:
        return out
    for line in lines:
        core = line.strip().strip(".!?,;: ").lower()
        if core and core in low and core not in out:
            out.append(core)
    for m in TRANSIENT_RE.finditer(low):
        phrase = m.group(1)
        if phrase not in out and not any(phrase in x for x in out):
            out.append(phrase)
    return out


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


def enforce_gate(c: Combined, gate: float = STAY_LOCAL_GATE, event_kind: str = "") -> Combined:
    """Keep a decision local only on confidence: stay_local with min(route, fix) confidence below
    `gate` goes to the fast LLM instead, with no fix applied. A routine step_done (combine.routine)
    is kept as decided."""
    if c.route != "stay_local" or routine(event_kind, c.fix, c.problems):
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


__all__ = ["LoopDetector", "RIGHT_NOW_ACTION", "ReplanGuard", "STAY_LOCAL_GATE", "TRANSIENT_PHRASES", "enforce_gate",
           "is_stop", "is_transient", "transient_mentions"]
