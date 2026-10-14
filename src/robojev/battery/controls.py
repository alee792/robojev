"""The two controls the fast loop is measured against (the idea from experiments/e12v2/eval/controls.py,
written afresh for the brain's View and the spelling task). Neither reads the script or the true
world; both return the brain's Combined with code-certain confidence, so the gate passes them.

AlwaysLLM     no decision model. Every event except a routine one (a step finishing; a plan arriving,
              which the planner just wrote) goes to the planner, and the arm CARRIES ON until the new
              plan comes. The brain's own rules still apply (STOP, models only pause, the replan-loop
              limit, the reading guard). It shows what the slow loop alone costs: a wait that pauses
              nothing, a hand nobody stops for, a correction answered seconds later.

RulesOnly     a code decider with keyword rules written ONLY from the `train` phrasings in
              phrasings.py (the held-out phrasings and any paraphrase were never looked at). The rules,
              in priority order (typed "stop" and the STOP button are code in the brain, before any of
              this):
  User text
   U1  names a word (an uppercase word of two or more letters, which every train new_word phrasing
       has)                                                                -> hold, planner
   U2  matches REPLY (do what you can, what you can, letters you have, you can manage: an answer to
       the robot's question, which may say "spell" without naming a word)  -> carry on, planner
   U3  matches NEW_WORD (actually, spell, scrap that, instead, change of plan, make it, I meant,
       rather than)                                                        -> hold, planner
   U4  matches WAIT (hang on, wait, hold on, one sec, pause)               -> pause, stay local
   U5  matches GO (go on, carry on, continue, keep going, go ahead)        -> resume, stay local
   U6  matches CHATTER (nice, coffee, doing great, camera, I like) or REMARK (borrowing, take this,
       need this letter, mine now, have it back, putting it back, returned, it's back, another one,
       one more, an extra, adding this, bumped, knocked, slid, elbow, put this one in, do this
       letter, done for you, help with this, reach past, excuse my hand, grabbing my, mind me)
                                                                           -> carry on, stay local
   U7  anything else                                                       -> carry on, planner
  Hands (the brain's bands: very close < 10 cm, near < 25, mid-range < 45)
   H1  very close                                                          -> back off
   H2  coming closer, or in the arm's path                                 -> pause
   H3  paused for a hand, and the hand is gone or neither of those          -> resume
   H4  otherwise                                                           -> carry on
  Cubes moved by someone else
   M1  the current step's cube, not yet grasped                            -> re-target
   M2  a cube that has gone, or one that appeared                          -> planner
   M3  a cube a finished step placed, no longer where it was put           -> re-queue
   M4  a cube now in a position                                            -> planner
   M5  otherwise                                                           -> carry on
  Steps
   S1  finished                                                            -> next step
   S2  failed with a missed grasp, a drop or a stall, first two times      -> retry
   S3  could not start, the place occupied, and another step can run       -> run it first
   S4  any other failure                                                   -> planner
  Other
   P1  a new plan                                                          -> carry on
   P2  paused with nothing happening: as H3                                -> resume or stay
   P3  every step ran, done conditions false                               -> planner
"""
from __future__ import annotations

import re

from robojev.brain.core.data import Combined
from robojev.brain.core.decision import View, object_step
from robojev.brain.core.plan import step_goal

from .spell import SLOT_RE, word_in

# Keyword tables, from phrasings.py's `train` halves only (see the module doc).
NEW_WORD = re.compile(r"\b(actually|spell|scrap that|instead|change of plan|make it|i meant|rather than)\b", re.I)
WAIT = re.compile(r"\b(hang on|wait|hold on|one sec|pause)\b", re.I)
GO = re.compile(r"\b(go on|carry on|continue|keep going|go ahead)\b", re.I)
CHATTER = re.compile(r"\b(nice|coffee|doing great|camera|i like)\b", re.I)
REMARK = re.compile(r"\b(borrowing|take this|need this letter|mine now|have it back|putting it back|returned|it's back|"
                    r"another one|one more|an extra|adding this|bumped|knocked|slid|elbow|put this one in|do this letter|"
                    r"done for you|help with this|reach past|excuse my hand|grabbing my|mind me)\b", re.I)
REPLY = re.compile(r"\b(do what you can|what you can|letters you have|you can manage)\b", re.I)

CERTAIN = {"right_now": 1.0, "in_plan_fix": 1.0, "route": 1.0}


def C(rn: str = "carry_on", fix: str = "none", route: str = "stay_local", reason: str = "") -> Combined:
    return Combined(rn, fix, route, reason, right_now_raw=rn, route_raw=route, fix_raw=fix, conf=dict(CERTAIN))


def _out(c: Combined) -> dict:
    return {"combined": c, "latency_ms": 1.0, "request": None, "answers": None, "in_tok": 0, "jev": False}


class AlwaysLLM:
    name = "always_llm"

    def decide(self, ev, v: View) -> dict:
        if ev.kind in ("step_done", "plan_arrived"):
            return _out(C(reason="routine"))
        c = C(route="fast_llm", reason="always LLM")
        c.react = True
        return _out(c)


class RulesOnly:
    name = "rules"

    def decide(self, ev, v: View) -> dict:
        return _out(self._rule(ev, v))

    # -- hands --------------------------------------------------------------------------------------
    @staticmethod
    def _hand_rn(v: View) -> str:
        h = v.hand
        hand_pause = v.mode == "paused" and (v.pause_reason or "").startswith(("a person", "paused after"))
        if h is not None and h["band"] == "very close":
            return "back_off"                                                                   # H1
        if h is not None and (h["approaching"] or h["in_path"]):
            return "pause"                                                                      # H2
        if hand_pause:
            return "resume"                                                                     # H3
        return "carry_on"                                                                       # H4

    # -- the rules ----------------------------------------------------------------------------------
    def _rule(self, ev, v: View) -> Combined:
        k = ev.kind
        if k == "user_text":
            t = ev.data.get("text", "")
            if word_in(t):
                return C("hold", route="fast_llm", reason="U1")
            if REPLY.search(t):
                return C(route="fast_llm", reason="U2")
            if NEW_WORD.search(t):
                return C("hold", route="fast_llm", reason="U3")
            if WAIT.search(t):
                return C("pause", reason="U4")
            if GO.search(t):
                return C("resume", reason="U5")
            if CHATTER.search(t) or REMARK.search(t):
                return C(reason="U6")
            return C(route="fast_llm", reason="U7")
        if k == "scene_change":
            ch = ev.data.get("change", {})
            if ch.get("what") == "hand":
                return C(self._hand_rn(v), reason="H")
            o = ev.object or ch.get("object")
            st, step = object_step(v, o) if o else ("no step", None)
            obj = v.state.obj(o) if o else None
            if v.current is not None and v.current.get("object") == o and not v.grasped and ch.get("what") == "object_moved":
                return C("re_target", reason="M1")
            if ch.get("what") in ("object_gone", "object_appeared"):
                return C(route="fast_llm", reason="M2")
            if step is not None and st == "done" and obj is not None and obj.where != step_goal(step)[1]:
                return C(fix="re_queue", reason="M3")
            if obj is not None and SLOT_RE.match(obj.where or ""):
                return C(route="fast_llm", reason="M4")
            return C(reason="M5")
        if k == "step_done":
            return C(reason="S1")
        if k == "step_failed":
            r = ev.data.get("reason", "")
            if any(x in r for x in ("grasp_failed", "dropped", "stalled")) and v.tries.get(ev.step_id, 0) <= 2:
                return C(fix="retry", reason="S2")
            if "occupied" in r and v.runnable_other:
                return C(fix="another_step_first", reason="S3")
            return C(route="fast_llm", reason="S4")
        if k == "plan_arrived":
            return C(reason="P1")
        if k == "paused_idle":
            rn = self._hand_rn(v)
            return C("resume" if rn == "resume" else "carry_on", reason="P2")
        if k == "plan_done_unmet":
            return C(route="fast_llm", reason="P3")
        return C(route="fast_llm", reason="other")


RULES = __doc__
