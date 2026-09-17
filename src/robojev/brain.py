"""Compose Jev answers into persistent arm commands. Hysteresis, commitment, overrides, the
sequencer's commitment to a running primitive, and the silence ladder all live here, in code.
Jev only ever picks; this turns picks into a goal pose and a gripper width.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from robojev import skills
from robojev.config import Config
from robojev.world import World

NONE = "none_of_these"
CONSERVATIVE_MOTION = {"hold", "back_off", "rise_away"}
SAFETY_PRIMS = set(skills.SAFETY)


@dataclass
class Judgment:
    key: str
    chosen: str | None
    p: float                      # p_max (choice), P(yes) (noul), or score (score)
    confidence: float | None
    probabilities: dict
    gated: bool
    applied: str                  # applied | gated | pending_confirmation | override | busy
    tag: int
    age_ms: float


class Brain:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.target: str | None = None
        self.motion = "hold"
        self.hover_position = "directly_above"
        self.hover_height = "high"
        self.speed_level = 1
        self.override = False
        self.avoid: str | None = None
        # sequencer
        self.prim: str | None = None
        self.prim_subject: str | None = None
        self.prim_started_t: float | None = None
        self.prim_status: str | None = None     # running | done | failed
        self.last_result: str | None = None
        self.place: str | None = None
        self.origin_xy = None                    # where the held object was picked up
        self.held: str | None = None             # label of the object we closed on
        self.done = False
        self.last_applied_t: float | None = None
        self.last_applied_tag = -1
        self.recent: list[str] = []
        self.judgments: dict[str, Judgment] = {}
        self._streak: dict[str, tuple[str, int]] = {}
        self.applied_count = 0
        self.offered_keys: list[str] = []

    # -- helpers -----------------------------------------------------------------------------
    def _streak_ok(self, key: str, candidate: str, needed: int) -> bool:
        c, n = self._streak.get(key, (None, 0))
        n = n + 1 if c == candidate else 1
        self._streak[key] = (candidate, n)
        return n >= needed

    def _note(self, text: str):
        if not self.recent or self.recent[-1] != text:
            self.recent.append(text)
            self.recent = self.recent[-8:]

    def reset_task(self):
        self.done = False
        self.place = None
        self._streak.pop("task_done", None)
        self._note("new request")

    def state(self) -> dict:
        now = time.time()
        age = None if self.last_applied_t is None else now - self.last_applied_t
        ladder = "fresh"
        if age is None or age > self.cfg.safety.silence_rise_s:
            ladder = "rise"
        elif age > self.cfg.safety.silence_hold_s:
            ladder = "hold"
        return {"target": self.target, "motion": self.motion, "hover_position": self.hover_position,
                "hover_height": self.hover_height, "speed_name": self.cfg.motion.speed_names[self.speed_level],
                "speed_level": self.speed_level, "override": self.override, "avoid": self.avoid, "ladder": ladder,
                "answer_age_s": age, "recent": list(self.recent),
                "prim": self.prim, "prim_subject": self.prim_subject, "prim_status": self.prim_status,
                "prim_age": (now - self.prim_started_t) if self.prim_started_t else None,
                "last_result": self.last_result, "place": self.place, "held": self.held, "done": self.done}

    def _start(self, key: str, world: World):
        name, _, subj = key.partition(":")
        self.prim, self.prim_subject = name, (subj or None)
        self.prim_started_t, self.prim_status = time.time(), "running"
        if name == "descend_to_grasp":
            e = world.entity(subj)
            self.origin_xy = tuple(e.xyz[:2]) if e else None
        if name == "close_gripper":
            self.held = world.above_label
        self._note(f"{name}" + (f" {subj}" if subj else ""))

    # -- apply one fresh answer set ---------------------------------------------------------------
    def apply(self, answers: dict, world: World, tag: int, age_ms: float, now: float | None = None) -> None:
        now = now or time.time()
        th = self.cfg.thresholds
        N = th.aggressive_consecutive
        labels = {e.label for e in world.entities}
        J = {}

        a = answers.get("target")
        if a and a.get("type") == "choice":
            probs, ch = a["probabilities"], a["choice"]
            pmax = max(probs.values()) if probs else 0.0
            status = "gated"
            if self.target and self.target not in labels:
                self.target = None; self._note("target lost from the scene")
            if pmax >= th.target_p_max:
                cand = None if ch == NONE else ch
                if world.holding_label and cand != world.holding_label and cand is not None:
                    status = "busy"          # while holding, the held object stays the target
                elif cand == self.target:
                    status = "applied"; self._streak["target"] = (cand, 99)
                elif cand is None:
                    status = "applied" if self._streak_ok("target", "none", N) else "pending_confirmation"
                    if status == "applied" and self.target is not None:
                        self._note(f"dropped target {self.target}"); self.target = None
                elif self._streak_ok("target", cand, th.target_switch_consecutive if self.target else 1):
                    status = "applied"; self._note(f"target set to {cand}"); self.target = cand
                else:
                    status = "pending_confirmation"
            J["target"] = Judgment("target", ch, pmax, a.get("confidence"), probs, pmax >= th.target_p_max, status, tag, age_ms)

        a = answers.get("place")
        if a and a.get("type") == "choice":
            probs, ch = a["probabilities"], a["choice"]
            pmax = max(probs.values()) if probs else 0.0
            status = "gated"
            if pmax >= th.place_p_max:
                if self.prim in ("move_to_place", "lower_to_place") and self.prim_status == "running":
                    status = "busy"
                elif ch == self.place or self._streak_ok("place", ch, 2):
                    if ch != self.place:
                        self._note(f"place: {ch}")
                    self.place = ch; status = "applied"
                else:
                    status = "pending_confirmation"
            J["place"] = Judgment("place", ch, pmax, a.get("confidence"), probs, pmax >= th.place_p_max, status, tag, age_ms)

        a = answers.get("next")
        if a and a.get("type") == "choice":
            probs, ch = a["probabilities"], a["choice"]
            pmax = max(probs.values()) if probs else 0.0
            status = "gated"
            if pmax >= th.next_p_max:
                if ch not in self.offered_keys:
                    status = "gated"
                elif ch in SAFETY_PRIMS:
                    running = self.prim_status == "running" and self.prim not in SAFETY_PRIMS
                    if running and pmax < th.next_interrupt_p:
                        status = "busy"       # a lukewarm "hold" must not stutter a primitive in progress
                    else:
                        status = "applied"
                        if self.prim != ch:
                            self._start(ch, world)
                elif self.prim_status == "running" and self.prim not in SAFETY_PRIMS:
                    status = "busy"
                elif self._streak_ok("next", ch, th.next_consecutive):
                    status = "applied"; self._start(ch, world)
                else:
                    status = "pending_confirmation"
            J["next"] = Judgment("next", ch, pmax, a.get("confidence"), probs, pmax >= th.next_p_max, status, tag, age_ms)

        a = answers.get("motion")
        if a and a.get("type") == "choice":
            ch, conf, probs = a["choice"], a.get("confidence", 0.0), a["probabilities"]
            status = "gated"
            if conf >= th.motion_conf:
                if ch in CONSERVATIVE_MOTION or ch == self.motion:
                    status = "applied"; self._streak["motion"] = (ch, 99)
                elif self._streak_ok("motion", ch, N):
                    status = "applied"
                else:
                    status = "pending_confirmation"
                if status == "applied" and ch != self.motion:
                    self._note(f"motion: {ch}"); self.motion = ch
            J["motion"] = Judgment("motion", ch, max(probs.values()), conf, probs, conf >= th.motion_conf, status, tag, age_ms)

        a = answers.get("hover_position")
        if a and a.get("type") == "choice":
            ch, conf, probs = a["choice"], a.get("confidence", 0.0), a["probabilities"]
            status = "gated"
            if conf >= th.hover_position_conf:
                if ch == self.hover_position or ch != "directly_above" or self._streak_ok("hover_position", ch, N):
                    status = "applied"
                    if ch != self.hover_position:
                        self._note(f"hover position: {ch}")
                    self.hover_position = ch
                else:
                    status = "pending_confirmation"
            J["hover_position"] = Judgment("hover_position", ch, max(probs.values()), conf, probs, conf >= th.hover_position_conf, status, tag, age_ms)

        a = answers.get("hover_height")
        if a and a.get("type") == "choice":
            ch, conf, probs = a["choice"], a.get("confidence", 0.0), a["probabilities"]
            status = "gated"
            if conf >= th.hover_height_conf:
                if ch == self.hover_height or ch == "high" or self._streak_ok("hover_height", ch, N):
                    status = "applied"
                    if ch != self.hover_height:
                        self._note(f"hover height: {ch}")
                    self.hover_height = ch
                else:
                    status = "pending_confirmation"
            J["hover_height"] = Judgment("hover_height", ch, max(probs.values()), conf, probs, conf >= th.hover_height_conf, status, tag, age_ms)

        a = answers.get("speed")
        if a and a.get("type") == "score":
            s = float(a["score"]); lvl = max(0, min(len(self.cfg.motion.speed_levels) - 1, int(round(s))))
            status = "applied"
            if lvl <= self.speed_level or self._streak_ok("speed", str(lvl), N):
                if lvl != self.speed_level:
                    self._note(f"speed: {self.cfg.motion.speed_names[lvl]}")
                self.speed_level = lvl
            else:
                status = "pending_confirmation"
            J["speed"] = Judgment("speed", self.cfg.motion.speed_names[lvl], s, a.get("confidence"), a.get("probabilities", {}), True, status, tag, age_ms)

        a = answers.get("avoid")
        if a and a.get("type") == "choice":
            probs, ch = a["probabilities"], a["choice"]
            pmax = max(probs.values()) if probs else 0.0
            status = "gated"
            if pmax >= th.avoid_p_max:
                if ch.startswith("move_away_from:"):
                    label = ch.split(":", 1)[1]
                    if label in labels:
                        status = "override"
                        if self.avoid != label:
                            self._note(f"avoiding {label}")
                        self.avoid = label
                        self._streak["avoid"] = ("clear", 0)
                elif self.avoid is not None:
                    if self._streak_ok("avoid", "clear", th.avoid_clear_consecutive):
                        self._note(f"clear of {self.avoid}"); self.avoid = None; status = "applied"
                    else:
                        status = "pending_confirmation"
                else:
                    status = "applied"
            if self.avoid and self.avoid not in labels:
                self.avoid = None
            J["avoid"] = Judgment("avoid", ch, pmax, a.get("confidence"), probs, pmax >= th.avoid_p_max, status, tag, age_ms)

        a = answers.get("orders_violated")
        if a and a.get("type") == "noul":
            p = float(a["noul"])
            self.override = p >= th.orders_violated_p
            if self.override:
                self._note("override: standing order at risk, holding")
            J["orders_violated"] = Judgment("orders_violated", "yes" if self.override else "no", p, None, {"yes": p, "no": 1 - p},
                                            self.override, "override" if self.override else "applied", tag, age_ms)

        a = answers.get("task_done")
        if a and a.get("type") == "noul":
            p = float(a["noul"])
            status = "applied"
            if p >= th.task_done_p:
                if self._streak_ok("task_done", "yes", th.task_done_consecutive) and not self.done:
                    self.done = True; self._note("task complete")
                    if self.prim not in SAFETY_PRIMS:
                        self._start("hold", world)
                elif not self.done:
                    status = "pending_confirmation"
            else:
                self._streak["task_done"] = ("no", 0)
            J["task_done"] = Judgment("task_done", "yes" if p >= th.task_done_p else "no", p, None, {"yes": p, "no": 1 - p},
                                      p >= th.task_done_p, status, tag, age_ms)

        self.judgments.update(J)
        self.last_applied_t, self.last_applied_tag = now, tag
        self.applied_count += 1

    # -- compose the persistent command -------------------------------------------------------------
    def compose(self, world: World, now: float | None = None):
        """Returns (goal xyz, speed cap m/s, gripper or None, reason)."""
        now = now or time.time()
        m, ws = self.cfg.motion, self.cfg.workspace
        ee, sp, tz = world.arm.ee, world.arm.setpoint, world.table_z
        st = self.state()
        slowest = m.speed_levels[0]
        self.offered_keys = [p.key for p in skills.offered(self.cfg, world, self)]
        if st["ladder"] == "rise":
            return (sp[0], sp[1], min(ws.z[1], tz + m.safe_height)), slowest, None, "no fresh answers: rising to safe height"
        if st["ladder"] == "hold":
            return sp, slowest, None, "no fresh answers: holding"
        cap = m.speed_levels[self.speed_level]
        av = world.entity(self.avoid) if self.avoid else None
        if av is not None:
            dx, dy = ee[0] - av.xyz[0], ee[1] - av.xyz[1]
            L = math.hypot(dx, dy); ux, uy = (dx / L, dy / L) if L > 1e-6 else (-1.0, 0.0)
            need = max(0.0, m.avoid_distance - L)
            goal = ws.clamp((ee[0] + ux * need, ee[1] + uy * need, max(sp[2], skills.hover_z(self.cfg, world))))
            return goal, cap, None, f"avoid {self.avoid}: {L*100:.0f} cm away, want {m.avoid_distance*100:.0f}"
        if self.override:
            return sp, cap, None, "orders_violated: holding"
        if self.prim is None:
            return (sp[0], sp[1], max(sp[2], skills.hover_z(self.cfg, world))), cap, None, "no primitive yet: holding at hover height"
        goal, gripper, done, fail, reason = skills.goal_for(self.cfg, world, self, now)
        if self.prim_status == "running":
            age = now - (self.prim_started_t or now)
            if fail:
                self.prim_status, self.last_result = "failed", f"{self.prim} failed: {fail}"; self._note(self.last_result)
            elif done:
                self.prim_status = "done"
                self.last_result = f"{self.prim}" + (f" {self.prim_subject}" if self.prim_subject else "") + " done"
                if self.prim == "close_gripper" and world.holding_label:
                    self.last_result = f"grasped {world.holding_label}"
                if self.prim == "open_gripper":
                    self.held = None
                self._note(self.last_result)
            elif age > m.primitive_timeout_s and self.prim not in SAFETY_PRIMS:
                self.prim_status, self.last_result = "failed", f"{self.prim} failed: timed out after {age:.0f} s"; self._note(self.last_result)
        if self.prim_status in ("done", "failed") and self.prim in ("move_above", "descend_to_grasp", "move_to_place", "lower_to_place", "set_down_here", "lift", "retreat", "rise_away", "back_off"):
            goal = sp   # finished primitives hold position until the next pick
        return goal, cap, gripper, reason
