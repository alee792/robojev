"""What arrives on the brain's one queue. Everything that can happen to an episode is one of these,
so the loop handles one thing at a time on one thread, in arrival order."""
from __future__ import annotations

from dataclasses import dataclass

from .core.data import Event
from .core.planner import PlanResult

from ..protocol import RobotEvent


@dataclass(frozen=True)
class FromRobot:
    """A server notification, bridged from the server's thread."""
    event: RobotEvent


@dataclass(frozen=True)
class UserSaid:
    """Typed text other than STOP (STOP never queues: it acts at once)."""
    text: str


@dataclass(frozen=True)
class Tick:
    """Every heartbeat: look at the world (hands move between notifications) and run timers."""


@dataclass(frozen=True)
class Wake:
    """STOP happened on another thread: wake the loop so it ends now."""


@dataclass(frozen=True)
class Decided:
    """A decision came back for `event`; `out` is the decider's dict (e12v2 Decider shape)."""
    event: Event
    out: dict


@dataclass(frozen=True)
class Planned:
    """A planner request came back. `seq` orders requests: only the newest one is applied."""
    seq: int
    t_request: float
    event: Event | None
    result: PlanResult


@dataclass(frozen=True)
class Crashed:
    """A decider or planner raised: a bug, surfaced in the loop rather than lost in a thread."""
    error: BaseException


Message = FromRobot | UserSaid | Tick | Wake | Decided | Planned | Crashed
