"""JSONL recorder: one line per policy tick of every skill, plus every event (H6's first version).

Written by the server at the policy rate, so a frame holds exactly what the policy saw (EE pose,
gripper, object poses) and what it commanded (goal, speed, yaw, gripper width). Every tuning run is
training data; the LeRobot-style conversion reads these files later and is not this file's job.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .protocol import RobotEvent


class JsonlRecorder:
    """Implements protocol.Recorder. Lines are {"type": "frame"|"event"|"note", ...}; floats are
    rounded to 0.1 mm / 0.1 ms so a 20-trial run stays a few MB. `tag` is merged into every line
    (the trials put their trial index there) so one file can hold many episodes."""

    def __init__(self, path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a" if append else "w")
        self.frames = 0
        self.events = 0
        self.tag: dict = {}

    def _write(self, rec: dict) -> None:
        if self.tag:
            rec = {**self.tag, **rec}
        self._f.write(json.dumps(_round(rec), separators=(",", ":")) + "\n")

    def frame(self, t: float, arm: str, skill_id: str | None, phase: str, observation: dict, action: dict) -> None:
        self._write({"type": "frame", "t": t, "arm": arm, "skill_id": skill_id, "phase": phase,
                     "observation": observation, "action": action})
        self.frames += 1

    def event(self, ev: RobotEvent) -> None:
        self._write({"type": "event", **asdict(ev)})
        self.events += 1
        self._f.flush()          # events are rare and are what a reader greps for after a crash

    def note(self, **fields) -> None:
        """Anything outside the protocol worth keeping next to the frames (trial setup, verdicts)."""
        self._write({"type": "note", **fields})
        self._f.flush()

    def close(self) -> dict[str, Any]:
        if not self._f.closed:
            self._f.close()
        return {"path": str(self.path), "frames": self.frames, "events": self.events}


def _round(x, nd: int = 4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x
