"""The brain's only hands on the arm: every protocol tool call the loop makes goes through here.

Keeping them in one place is what makes two rules checkable: models reach the arm only through
`set_motion` (hold / pause / resume) and `retarget`, and `stop` is called only by the STOP path.
After STOP nothing else is sent (the server refuses a start anyway; this avoids asking).

`skill_id` is the skill this brain started and still follows. `foreign` marks a skill found on the
arm at connect (a reconnect: the protocol cannot name it, see loop.py), which the brain lets run.
"""
from __future__ import annotations

import threading

from ..protocol import RobotServer, SkillStatus

MOTION_TOOL = {"holding": "hold", "paused": "pause", "running": "resume"}


class ArmControl:
    def __init__(self, server: RobotServer, arm: str):
        self.server, self.arm = server, arm
        self.skill_id: str | None = None
        self.foreign = False
        self.stopped = threading.Event()        # set by STOP from any thread

    @property
    def busy(self) -> bool:
        """A skill is on the arm: ours, or one found there at connect."""
        return self.skill_id is not None or self.foreign

    def precondition(self, skill: str, args: dict) -> str | None:
        if self.stopped.is_set():
            return "precondition: STOP was pressed"
        return self.server.precondition(self.arm, skill, args)

    def start(self, skill: str, args: dict) -> str:
        """-> the new skill's id. Raises ValueError with the server's literal reason if it refuses."""
        if self.stopped.is_set():
            raise ValueError("precondition: STOP was pressed")
        self.skill_id = self.server.start(self.arm, skill, args)
        return self.skill_id

    def status(self) -> SkillStatus | None:
        return self.server.status(self.skill_id) if self.skill_id is not None else None

    def set_motion(self, mode: str) -> None:
        """Make the arm match the brain's mode: holding -> hold, paused -> pause, running -> resume.
        Sent only while a skill is on the arm (an idle arm has nothing to hold); idempotent server-side."""
        if self.busy and not self.stopped.is_set():
            getattr(self.server, MOTION_TOOL[mode])(self.arm)

    def retarget(self) -> None:
        if self.skill_id is not None and not self.stopped.is_set():
            self.server.retarget(self.skill_id)

    def ended(self) -> None:
        """Our skill finished or failed (its end event arrived)."""
        self.skill_id = None

    def release(self) -> None:
        """Stop following our skill: hold it at a safe point, grip kept, until the next start() cancels
        it (a start on a busy arm cancels its skill: state "cancelled", no event)."""
        if self.skill_id is not None and not self.stopped.is_set():
            self.server.hold(self.arm)
        self.skill_id = None

    def stop(self) -> None:
        """STOP. Any thread; called only by the STOP path (the button, a typed "stop")."""
        self.stopped.set()
        self.server.stop()
