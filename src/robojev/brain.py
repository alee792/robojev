"""Compose Jev answers into persistent arm commands. Hysteresis, commitment, overrides and the
silence ladder all live here, in code. Jev only ever picks; this turns picks into a goal pose.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from robojev.config import Config
from robojev.questions.v0 import NONE
from robojev.world import World

CONSERVATIVE_MOTION = {"hold", "back_off", "rise_away"}


@dataclass
class Judgment:
    key: str
    chosen: str | None
    p: float                      # p_max (choice), P(yes) (noul), or score (score)
    confidence: float | None
    probabilities: dict
    gated: bool                   # passed its threshold
    applied: str                  # "applied" | "gated" | "pending_confirmation" | "override"
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
        self.last_applied_t: float | None = None
        self.last_applied_tag = -1
        self.recent: list[str] = []
        self.judgments: dict[str, Judgment] = {}
        self._streak: dict[str, tuple[str, int]] = {}   # key -> (candidate, consecutive count)
        self.applied_count = 0

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

    def state(self) -> dict:
        age = None if self.last_applied_t is None else time.time() - self.last_applied_t
        ladder = "fresh"
        if age is None or age > self.cfg.safety.silence_rise_s:
            ladder = "rise"
        elif age > self.cfg.safety.silence_hold_s:
            ladder = "hold"
        return {"target": self.target, "motion": self.motion, "hover_position": self.hover_position,
                "hover_height": self.hover_height, "speed_name": self.cfg.motion.speed_names[self.speed_level],
                "speed_level": self.speed_level, "override": self.override, "ladder": ladder,
                "answer_age_s": age, "recent": list(self.recent)}

    # -- apply one fresh answer set ---------------------------------------------------------------
    def apply(self, answers: dict, world: World, tag: int, age_ms: float, now: float | None = None) -> None:
        now = now or time.time()
        th = self.cfg.thresholds
        N = th.aggressive_consecutive
        labels = {e.label for e in world.entities}
        J = {}

        # target: dynamic option count -> gate on p_max
        a = answers.get("target")
        if a and a.get("type") == "choice":
            probs, ch = a["probabilities"], a["choice"]
            pmax = max(probs.values()) if probs else 0.0
            status = "gated"
            if self.target and self.target not in labels:
                self.target = None
                self._note("target lost from the scene")
            if pmax >= th.target_p_max:
                cand = None if ch == NONE else ch
                if cand == self.target:
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

        # motion: conservative latches, approach needs N consecutive
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
                    self._note(f"motion: {ch}")
                    self.motion = ch
            J["motion"] = Judgment("motion", ch, max(probs.values()), conf, probs, conf >= th.motion_conf, status, tag, age_ms)

        a = answers.get("hover_position")
        if a and a.get("type") == "choice":
            ch, conf, probs = a["choice"], a.get("confidence", 0.0), a["probabilities"]
            status = "gated"
            if conf >= th.hover_position_conf:
                aggressive = ch == "directly_above"
                if ch == self.hover_position or not aggressive or self._streak_ok("hover_position", ch, N):
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
            s = float(a["score"]); lvl = int(round(s)); lvl = max(0, min(len(self.cfg.motion.speed_levels) - 1, lvl))
            status = "applied"
            if lvl <= self.speed_level or self._streak_ok("speed", str(lvl), N):
                if lvl != self.speed_level:
                    self._note(f"speed: {self.cfg.motion.speed_names[lvl]}")
                self.speed_level = lvl
            else:
                status = "pending_confirmation"
            J["speed"] = Judgment("speed", self.cfg.motion.speed_names[lvl], s, a.get("confidence"), a.get("probabilities", {}), True, status, tag, age_ms)

        a = answers.get("orders_violated")
        if a and a.get("type") == "noul":
            p = float(a["noul"])
            self.override = p >= th.orders_violated_p
            if self.override:
                self._note("override: standing order at risk, holding")
            J["orders_violated"] = Judgment("orders_violated", "yes" if self.override else "no", p, None, {"yes": p, "no": 1 - p},
                                            self.override, "override" if self.override else "applied", tag, age_ms)

        self.judgments.update(J)
        self.last_applied_t, self.last_applied_tag = now, tag
        self.applied_count += 1

    # -- compose the persistent command -------------------------------------------------------------
    def compose(self, world: World, now: float | None = None) -> tuple[tuple[float, float, float], float, str]:
        """Returns (goal xyz, speed cap m/s, reason)."""
        now = now or time.time()
        m, ws = self.cfg.motion, self.cfg.workspace
        ee = world.arm.ee
        sp = world.arm.setpoint
        tz = world.table_z
        st = self.state()
        slowest = m.speed_levels[0]
        if st["ladder"] == "rise":
            return (sp[0], sp[1], tz + m.safe_height), slowest, "no fresh answers: rising to safe height"
        if st["ladder"] == "hold":
            return sp, slowest, "no fresh answers: holding"
        cap = m.speed_levels[self.speed_level]
        if self.override:
            return sp, cap, "orders_violated: holding"
        tgt = world.entity(self.target) if self.target else None
        z = tz + m.hover_heights[self.hover_height]
        if self.motion == "rise_away":
            return (sp[0], sp[1], tz + m.safe_height), cap, "rise_away"
        if tgt is None:
            return (sp[0], sp[1], max(sp[2], z)), cap, "no target: holding at hover height"
        if self.motion == "hold":
            return sp, cap, "hold"
        if self.motion == "back_off":
            dx, dy = ee[0] - tgt.xyz[0], ee[1] - tgt.xyz[1]
            L = math.hypot(dx, dy)
            ux, uy = (dx / L, dy / L) if L > 1e-6 else (-1.0, 0.0)
            return (ee[0] + ux * m.back_off_distance, ee[1] + uy * m.back_off_distance, z), cap, "back_off"
        # approach
        gx, gy = tgt.xyz[0], tgt.xyz[1]
        if self.hover_position == "offset_toward_robot":
            L = math.hypot(gx, gy) or 1.0
            gx, gy = gx - gx / L * m.standoff, gy - gy / L * m.standoff
        elif self.hover_position == "offset_left":
            gy += m.standoff
        elif self.hover_position == "offset_right":
            gy -= m.standoff
        goal = ws.clamp((gx, gy, z))
        return goal, cap, f"approach {self.target} ({self.hover_position}, {self.hover_height})"
