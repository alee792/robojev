"""Changes the robot didn't cause, settled before they become events.

e12v2's ChangeDetector (reused unchanged) reports a change of place at once, as certain: in its toy
world a `where` never flickers. On physics it does: placing a block in the slot next to another
jostles the neighbour, which reads "table" for a tick and its slot again the next (seen on K2's
MuJoCo server), and camera perception will flicker more. Each flicker would be a "moved by someone
else" event, and a decision that re-queues a block already where it belongs.

So an object change is held back until the object has stayed changed for `settle_s`; a change that
undoes the one held back cancels both; the robot's own doing (`touched`) drops it. What comes out is
the net change since the object last rested. Hands are not held back: a hand near the arm is a
safety question and is decided at once.
"""
from __future__ import annotations

from e12v2.core.changes import MOVE_CM, ChangeDetector
from e12v2.core.data import WorldState, dxy


class SettledChanges:
    def __init__(self, settle_s: float, detector: ChangeDetector | None = None):
        self.settle_s = settle_s
        self.detector = detector or ChangeDetector()
        self.pending: dict[str, tuple[dict, float]] = {}    # object id -> (net change, when it last changed)

    def update(self, state: WorldState, t: float, path_to: tuple | None = None,
               touched: set[str] | None = None) -> list[dict]:
        """-> changes to report now: hand changes at once, object changes once settled."""
        touched = touched or set()
        out = []
        for ch in self.detector.update(state, path_to, touched):
            if ch["what"] == "hand":
                out.append(ch)
                continue
            oid = ch["object"]
            held = self.pending.get(oid)
            net = ch if held is None else self._net(held[0], ch)
            if net is None:
                del self.pending[oid]
            else:
                self.pending[oid] = (net, t)
        for oid in touched & set(self.pending):
            del self.pending[oid]
        for oid, (ch, since) in list(self.pending.items()):
            if t - since >= self.settle_s:
                out.append(ch)
                del self.pending[oid]
        return out

    @staticmethod
    def _net(first: dict, then: dict) -> dict | None:
        """`first` followed by `then`, as one change since the object last rested; None if they cancel."""
        if first["what"] == then["what"] == "object_moved":
            a, b = first["from_xy"], then["to_xy"]
            moved = dxy(a, b)
            if first["from"] == then["to"] and moved <= MOVE_CM:
                return None
            return {**then, "from": first["from"], "from_xy": a, "moved_cm": round(moved)}
        if first["what"] == "object_gone" and then["what"] == "object_appeared" and then["to"] == first["from"]:
            return None
        return then
