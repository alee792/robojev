"""The controls: the same harness with Jev replaced. Neither may import the oracle (tests check).

RulesDecider  hand-written reactions, written against the core scenarios the way an engineer would
              write them, with fixed numbers where Jev gets bands. Rules, in priority order
              (typed "stop" and the STOP button are code in every arm, before any of this):
  User text
   U1  matches WAIT (wait, hold on, hang on, one sec/second/moment, just a sec, pause, freeze)  -> pause, stay local
   U2  matches GO (go on, carry on, continue, go ahead, resume, keep going, ok go, proceed)     -> resume, stay local
   U3  matches CHANGE (actually, instead, rather, change, swap, switch, reverse, other way,
       opposite, wrong, no, not, don't, highest, lowest, biggest, smallest, left, right, first,
       last, order)                                                                             -> hold, fast LLM
   U4  matches PRAISE (nice, good, great, thanks, cool, lovely, perfect, well done, awesome)     -> carry on, stay local
   U5  anything else                                                                            -> carry on, fast LLM
  Hands (distances from the gripper, cm)
   H1  hand held out still while the current step is a hand_over                                -> carry on (resume if paused for a hand)
   H2  hand closer than 10                                                                      -> back off
   H3  hand closer than 30 and coming closer, or in the arm's path (8 cm corridor)             -> pause
   H4  no hand, or farther than 30 and not coming closer, while paused for a hand              -> resume
   H5  otherwise                                                                                -> carry on
  Objects moved by someone else
   M1  the current step's object, not yet grasped                                               -> re-target
   M2  a dont_touch object now within 6 cm of an object a pending step moves                   -> hold, fast LLM
   M3  a keep_out_of object now in its forbidden place: re-queue its step if it has one,       else fast LLM
   M4  an object whose step already ran and that is no longer where it put it                  -> re-queue
   M5  an object with a pending step                                                            -> carry on
   M6  anything else                                                                            -> carry on, fast LLM
  Steps
   S1  step finished                                                                            -> next step
   S2  failed: nothing grasped / a hand in the way / handover timed out, first two failures     -> retry
   S3  could not start: place occupied or object covered, and another step can run             -> run it first
   S4  failed: blocked by a constraint                                                          -> hold, fast LLM
   S5  any other failure                                                                        -> fast LLM
  Other
   P1  a new plan: an object named in the task text (by its name, e.g. "block 4", or colour for
       an unnumbered block) that the plan never mentions                                        -> fast LLM, else carry on
   P2  paused with nothing happening: as H4                                                     -> resume or stay
   P3  every step ran but done conditions are false: re-queue the object's step if it has one,  else fast LLM

AlwaysLLMDecider  no Jev: every event except a routine step-finished goes to the fast LLM, which
              picks the reaction (right now) and replans; the arm carries on until it answers. Code
              still enforces safety (stop distance, constraints, STOP).
"""

from __future__ import annotations

import math
import re

from ..core import plan as P
from ..core.data import Combined, dxy, seg_dist
from ..core.decision import object_step

WAIT = re.compile(r"\b(wait|hold on|hang on|one (sec|second|moment)|just a (sec|second|moment)|a sec|pause|freeze)\b", re.I)
GO = re.compile(r"\b(go on|carry on|continue|go ahead|resume|keep going|ok(ay)?,? go|proceed)\b", re.I)
CHANGE = re.compile(r"\b(actually|instead|rather|change|swap|switch|reverse|other way|opposite|wrong|no|not|don'?t|do not|"
                    r"highest|lowest|biggest|smallest|left|right|first|last|order)\b", re.I)
PRAISE = re.compile(r"\b(nice|good|great|thanks|thank you|cool|lovely|perfect|well done|awesome)\b", re.I)

RULES = __doc__


def C(rn="carry_on", fix="none", route="stay_local", reason=""):
    return Combined(rn, fix, route, reason, right_now_raw=rn, route_raw=route, fix_raw=fix)


class RulesDecider:
    name = "rules"

    def decide(self, ev, v) -> dict:
        c = self._rule(ev, v)
        return {"combined": c, "latency_ms": 1.0, "request": None, "answers": None, "in_tok": 0, "jev": False}

    def _hand(self, v):
        st = v.state
        h, a = st.hand, st.arm
        if h is None:
            return None
        d = math.dist((h.x, h.y, h.z), (a.x, a.y, a.z))
        nd = math.dist((h.x + h.vx, h.y + h.vy, h.z), (a.x, a.y, a.z))
        path = v.path_to is not None and seg_dist((h.x, h.y), (a.x, a.y), v.path_to) < 8 and dxy((a.x, a.y), v.path_to) > 1
        return {"d": d, "closer": nd < d - 0.5, "path": path, "out": h.held_out}

    def _hand_rn(self, v) -> str:
        hf = self._hand(v)
        hand_pause = v.mode == "paused" and (v.pause_reason or "").startswith(("a person", "backed off"))
        handing = v.current is not None and v.current["skill"] == "hand_over"
        if hf is not None and hf["out"] and handing:
            return "resume" if hand_pause else "carry_on"                                   # H1
        if hf is not None and hf["d"] < 10:
            return "back_off"                                                               # H2
        if hf is not None and ((hf["d"] < 30 and hf["closer"]) or hf["path"]):
            return "pause"                                                                  # H3
        if hand_pause and (hf is None or (hf["d"] > 30 and not hf["closer"])):
            return "resume"                                                                 # H4
        return "carry_on"                                                                   # H5

    def _rule(self, ev, v) -> Combined:
        k = ev.kind
        if k == "user_text":
            t = ev.data.get("text", "")
            if WAIT.search(t):
                return C("pause", reason="U1")
            if GO.search(t):
                return C("resume", reason="U2")
            if CHANGE.search(t):
                return C("hold", route="fast_llm", reason="U3")
            if PRAISE.search(t):
                return C(reason="U4")
            return C(route="fast_llm", reason="U5")
        if k == "scene_change":
            ch = ev.data.get("change", {})
            if ch.get("what") == "hand":
                return C(self._hand_rn(v), reason="H")
            o = ev.object
            st, step = object_step(v, o) if o else ("no step", None)
            obj = v.state.obj(o) if o else None
            forb = P.forbidden(v.plan) if v.plan else set()
            kout = dict(P.keep_out(v.plan)) if v.plan else {}
            if v.current and v.current.get("object") == o and not v.grasped:
                return C("re_target", reason="M1")
            if o in forb and obj is not None:
                for sid in v.queue:
                    s = P.steps_by_id(v.plan)[sid]
                    so = v.state.obj(s["object"]) if s["object"] != P.NONE else None
                    if so is not None and so.id != o and so.where == "table" and dxy((so.x, so.y), (obj.x, obj.y)) < 6:
                        return C("hold", route="fast_llm", reason="M2")
            if o in kout and obj is not None and v.state.in_place(o, kout[o]):
                if st in ("done", "failed", "skipped"):
                    return C(fix="re_queue", reason="M3")
                return C(route="fast_llm", reason="M3")
            if st == "done" and obj is not None and obj.where != P.step_goal(step)[1]:
                return C(fix="re_queue", reason="M4")
            if st in ("pending", "running"):
                return C(reason="M5")
            return C(route="fast_llm", reason="M6")
        if k == "step_done":
            return C(reason="S1")
        if k == "step_failed":
            r = ev.data.get("reason", "")
            if ("nothing grasped" in r or "hand" in r or "take it" in r or "held out" in r) and v.tries.get(ev.step_id, 0) <= 2:
                return C(fix="retry", reason="S2")
            if ("occupied by" in r or "on top of it" in r) and v.runnable_other:
                return C(fix="another_step_first", reason="S3")
            if "must not be touched" in r or "constraint" in r:
                return C("hold", route="fast_llm", reason="S4")
            return C(route="fast_llm", reason="S5")
        if k == "plan_arrived":
            m = P.mentioned(v.plan) if v.plan else set()
            task = (v.task + " " + " ".join(v.user_messages)).lower()
            for o in v.state.objects.values():
                named = o.name.lower().replace("the ", "") in task or (o.number is None and o.colour in task)
                if named and o.id not in m:
                    return C(route="fast_llm", reason="P1")
            return C(reason="P1")
        if k == "paused_idle":
            rn = self._hand_rn(v)
            return C("resume" if rn == "resume" else "carry_on", reason="P2")
        if k == "plan_done_unmet":
            st, _ = object_step(v, ev.object) if ev.object else ("no step", None)
            if st in ("done", "failed", "skipped"):
                return C(fix="re_queue", reason="P3")
            return C(route="fast_llm", reason="P3")
        return C(route="fast_llm", reason="other")


class AlwaysLLMDecider:
    name = "always_llm"

    def decide(self, ev, v) -> dict:
        if ev.routine:
            c = C(reason="routine step finished")
        else:
            c = C(route="fast_llm", reason="always LLM")
            c.react = True
        return {"combined": c, "latency_ms": 1.0, "request": None, "answers": None, "in_tok": 0, "jev": False}
