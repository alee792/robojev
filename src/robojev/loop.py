"""The Doom-style loop: perception thread -> world -> state -> battery -> Jev -> brain -> arm.

Timing model (08 §E2: at 10 Hz most answers arrive after the next tick starts):
  * event-driven (Loop.event_driven, the default): a request is fired only when something material
    changed since the last request that was actually sent, or after Loop.max_silence_s of silence;
    otherwise the tick is logged "skipped_no_change" and nothing is sent. `--clocked` restores one
    request per tick.
  * a request is fired only if fewer than max_in_flight are outstanding, else the tick is skipped
  * each request carries a tag (tick number) and the wall time of its state snapshot
  * an answer is applied only if its tag is newer than the last applied and its snapshot is not
    older than stale_answer_s; otherwise it is logged as dropped
  * compose() runs every tick regardless, so the silence ladder acts without any answer
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from robojev.brain import Brain
from robojev.config import Config
from robojev.events import ChangeDetector
from robojev.jev import JevClient
from robojev.log import RunLog
from robojev.perception.memory import Tracker
from robojev.questions import load as load_questions
from robojev.state import STATE_VERSION, render
from robojev.world import build_world


class Perception(threading.Thread):
    """Runs the detector + tracker at perception_hz. `camera=None` means the virtual scene."""

    def __init__(self, cfg: Config, arm, camera=None, virtual=None, table_z: float | None = None, log=None,
                 cameras: list | None = None, namer=None, display: list | None = None):
        """`cameras`: list of (name, cam, extrinsic_fn or None, finger_mask). None extrinsic = wrist chain.
        `camera=` is shorthand for a single wrist camera.
        `display`: list of (name, cam) shown on the dashboard only -- polled slowly in their own
        thread, never detected on, so an extra camera can never cost the loop a detection."""
        super().__init__(daemon=True, name="perception")
        self.cfg, self.arm, self.virtual, self.log = cfg, arm, virtual, log
        if cameras is None and camera is not None:
            cameras = [("wrist", camera, None, True)]
        self.cameras = cameras or []
        self.display = list(display or [])
        self.display_info: dict = {}         # name -> {"error": ...}, merged into the perception panel
        self.cam = self.cameras[0][1] if self.cameras else None
        self.tracker = Tracker(ttl_s=cfg.loop.remembered_ttl_s, out_of_view_s=cfg.loop.out_of_view_s)
        self.table_z = table_z if table_z is not None else cfg.table_z
        self._table_samples = []
        self._table_locked = False
        self.lock = threading.Lock()
        self.frame_jpeg: bytes | None = None
        self.info: dict = {}
        self.stop_evt = threading.Event()
        self.detectors = {}
        self.fps = 0.0
        self.frames = {}   # name -> latest jpeg
        self.held_label: str | None = None   # set by the loop from the brain; the track follows the EE
        self.raw: dict = {}                  # name -> last raw detections (for the tick log)
        self.colors: dict = {}               # name -> last colour frame (for VLM crops)
        self.vlm = None
        if namer is not None:
            from robojev.perception.vlm import NamingWorker
            self.vlm = NamingWorker(namer); self.vlm.start()
        if self.cameras:
            from robojev.perception.detect import Detector
            from robojev.perception.geometry import Intrinsics
            for name, cam, ext, fmask in self.cameras:
                self.detectors[name] = Detector(Intrinsics(cam.info), extrinsic=ext, finger_mask=fmask)

    def camera_names(self) -> list[str]:
        """Every camera this run can show, wrist first: the detection cameras in order, the sim
        third-person view if this run renders one, then the display-only cameras."""
        names = [c[0] for c in self.cameras]
        if getattr(self.cam, "want_third", False):
            names.append("third")
        return names + [n for n, _ in self.display]

    def _display_loop(self, hz: float = 2.5):
        """Display-only cameras, in their own thread: fetch, re-encode, store. Each camera is wrapped
        on its own, so one that hangs or dies never stops the others (or the detection loop)."""
        period = 1 / hz
        while not self.stop_evt.is_set():
            t = time.time()
            for name, cam in self.display:
                try:
                    f = cam.frame()
                    if f is None:
                        continue
                    ok, jpg = cv2.imencode(".jpg", f.color, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        self.frames[name] = jpg.tobytes()
                    self.display_info.pop(name, None)
                except Exception as e:
                    self.display_info[name] = {"error": repr(e)[:120]}
            self.stop_evt.wait(max(0.0, period - (time.time() - t)))

    def run(self):
        if self.display:
            threading.Thread(target=self._display_loop, daemon=True, name="perception-display").start()
        period = 1 / self.cfg.loop.perception_hz
        n, t0 = 0, time.time()
        while not self.stop_evt.is_set():
            t = time.time()
            try:
                if not self.cameras:
                    dets = self.virtual.detections(t)
                    info = {"plane_z_at_origin": self.virtual.table_z}
                    frame = None
                else:
                    snap = self.arm.snapshot()
                    have_pose = snap.rot is not None and snap.status in ("live", "frozen", "baselining")
                    dets, info, frame = [], {}, None
                    for name, cam, ext, fmask in self.cameras:
                        try:
                            f = cam.frame()
                        except Exception as e:      # one camera's hiccup must not blind the other
                            info[name] = {"error": repr(e)[:120]}
                            continue
                        if ext is None and not have_pose:
                            info[name] = {"waiting": f"arm {snap.status}; no pose yet"}
                            d = []
                        else:
                            pose6 = self._pose6(snap) if have_pose else [0, 0, 0, 0, 0, 0]
                            d, inf = self.detectors[name].run(f.color, f.depth_m, pose6, holding=snap.holding)
                            info[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in inf.items()}
                            if "plane_z_at_origin" in inf and "plane_z_at_origin" not in info:
                                info["plane_z_at_origin"], info["plane_tilt_deg"] = inf["plane_z_at_origin"], inf.get("plane_tilt_deg", 0)
                        for det in d:
                            det.camera = name
                        self.colors[name] = f.color
                        self.raw[name] = [(round(det.base_xyz[0], 3), round(det.base_xyz[1], 3), round(det.height, 3), round(det.width, 3)) for det in d]
                        img = f.color.copy()
                        for det in d:
                            if getattr(det, "flat", False):
                                for (pu, pv) in getattr(det, "pixels", []):
                                    cv2.circle(img, (int(pu), int(pv)), 1, (255, 120, 0), -1)
                            cv2.circle(img, det.pixel, 8, (0, 0, 255), 2)
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            self.frames[name] = jpg.tobytes()
                        dets.extend(d)
                # The table does not move, so its height is measured once from a good vantage point
                # and then locked. Left free-running it sank 2 cm while the arm was low and carrying
                # (the camera then sees only close, steeply angled table), which drove the cup into
                # the table and tripped the lag guard mid-place (demo7).
                z = info.get("plane_z_at_origin")
                if z is not None and info.get("plane_tilt_deg", 99) < 6 and not self._table_locked:
                    if snap is None or snap.ee[2] > 0.12:      # only from up high, where the view is wide
                        self._table_samples.append(z)
                        self._table_samples = self._table_samples[-20:]
                        self.table_z = float(np.median(self._table_samples))
                        spread = (max(self._table_samples) - min(self._table_samples)) if self._table_samples else 1.0
                        if len(self._table_samples) >= 20 and spread < 0.01:
                            self._table_locked = True
                            if self.log:
                                self.log.write("events", kind="perception",
                                               text=f"table height locked at {self.table_z:+.3f} m (spread {spread*1000:.0f} mm)")
                with self.lock:
                    closing = snap.gripper_goal is not None and snap.gripper_goal < 0.02
                    self.tracker.update(dets, t, carried_xy=(snap.ee[:2] if (self.cameras and (snap.holding or closing)) else None),
                                        ee_xy=(snap.ee[:2] if self.cameras else None))
                    if self.cameras and self.held_label and snap.holding:
                        self.tracker.pin(self.held_label, snap.ee[:2], t)
                    if self.cameras and have_pose and not snap.holding:
                        self._absence_check(t, snap)
                    if self.vlm is not None:
                        self._feed_vlm(t)
                    self.info = info | dict(self.display_info) | {"table_z": self.table_z, "n_dets": len(dets)}
                    self.frame_jpeg = self.frames.get(self.cameras[0][0]) if self.cameras else None
                n += 1
                self.fps = n / max(1e-6, time.time() - t0)
            except Exception as e:
                with self.lock:
                    self.info = {"error": repr(e)}
                if self.log:
                    self.log.write("events", kind="perception_error", text=repr(e))
            dt = time.time() - t
            if dt < period:
                time.sleep(period - dt)

    def _feed_vlm(self, now):
        """Submit crops for confirmed, unnamed tracks; apply names that came back (under self.lock)."""
        from robojev.perception.vlm import crop_around
        for e in self.tracker.stable():
            if e.name is None and e.vlm_kind is None and e.last_pixel and e.last_pixel[0] in self.colors:
                cam, u, v = e.last_pixel
                self.vlm.submit(e.id, crop_around(self.colors[cam], (u, v)), e.height, e.width, e.kind(), e.color)
        for eid, res in self.vlm.take().items():
            e = self.tracker.entities.get(eid)
            if e is None:
                continue
            if not res.is_object and res.confidence >= 0.6:
                e.phantom = True
                if self.log:
                    self.log.write("events", kind="vlm", text=f"{e.label()} dropped as not an object: {res.reason}")
                continue
            e.vlm_kind = res.kind
            if res.confidence >= 0.5 and res.name:
                self.tracker.set_name(eid, res.name)
            if self.log:
                self.log.write("events", kind="vlm", text=f"{eid}: {res.name} ({res.kind}, {res.confidence:.2f}) {res.reason}")

    def _absence_check(self, t, snap):
        """Object permanence with negative evidence: if a confirmed track's spot is well inside the
        wrist camera's image at a usable range and no detection matched it this frame, count it as
        absent; ~0.8 s of that and the track is dropped (the cup moved by hand while the camera
        was elsewhere left the arm grasping air three times: real run 12)."""
        name, cam, ext, fmask = self.cameras[0]
        det = self.detectors[name]
        R, tt = det.extrinsic(self._pose6(snap))
        intr = det.intr
        for e in self.tracker.stable():
            if e.last_seen >= t - 0.05:
                e.absent = 0
                continue
            if np.hypot(e.xyz[0] - snap.ee[0], e.xyz[1] - snap.ee[1]) < 0.20:
                continue   # right next to the gripper the view is masked/partial: no verdict (a just-released cup was 'gone': real run 13)
            p = np.asarray([e.xyz[0], e.xyz[1], e.xyz[2] + e.height / 2], float)
            c = R.T @ (p - tt)
            if not (0.15 < c[2] < 0.75):
                continue
            u, v = intr.fx * c[0] / c[2] + intr.ppx, intr.fy * c[1] / c[2] + intr.ppy
            if not (60 < u < intr.w - 60 and 60 < v < intr.h - 100):
                continue   # near the border or the fingers: no verdict
            e.absent = getattr(e, "absent", 0) + 1
            if e.absent >= 8:
                self.tracker.forget(e.id)
                self.log.write("events", kind="perception", text=f"{e.label()} is gone: looked where it was and nothing is there")

    def _pose6(self, snap):
        """EE pose as the driver reports it: [x, y, z, angle-axis]."""
        rot = snap.rot if snap.rot is not None else (0.0, 0.0, 0.0)
        return list(snap.ee) + list(rot)

    def entities(self):
        with self.lock:
            return list(self.tracker.stable()), self.tracker.in_view


@dataclass
class Stats:
    sent: int = 0
    ok: int = 0
    errors: int = 0
    timeouts: int = 0
    skipped: int = 0
    skipped_no_change: int = 0
    dropped_stale: int = 0
    dropped_order: int = 0
    latencies: list = field(default_factory=list)
    ages: list = field(default_factory=list)
    tokens: int = 0

    def pct(self, xs, p):
        if not xs:
            return None
        xs = sorted(xs); k = int((len(xs) - 1) * p / 100)
        return xs[k]

    def view(self):
        L = self.latencies[-200:]; A = self.ages[-200:]
        return {"sent": self.sent, "ok": self.ok, "errors": self.errors, "timeouts": self.timeouts, "skipped": self.skipped,
                "skipped_no_change": self.skipped_no_change,
                "dropped_stale": self.dropped_stale, "dropped_order": self.dropped_order,
                "latency_p50": self.pct(L, 50), "latency_p95": self.pct(L, 95), "latency_max": max(L) if L else None,
                "age_p50": self.pct(A, 50), "age_p95": self.pct(A, 95), "tokens": self.tokens,
                "sparkline": L[-60:]}


class Loop:
    def __init__(self, cfg: Config, arm, perception: Perception, log: RunLog, use_jev: bool = True,
                 task: str = "", orders: list[str] | None = None):
        self.cfg, self.arm, self.per, self.log = cfg, arm, perception, log
        self.qmod = load_questions(cfg.loop.question_set)
        self.brain = Brain(cfg)
        self.use_jev = use_jev
        self.jev = JevClient(cfg.loop.model, timeout_s=cfg.safety.request_timeout_s) if use_jev else None
        self.task = task
        self.orders = list(orders or [])
        self.stats = Stats()
        self.detector = ChangeDetector(move_m=cfg.loop.entity_move_m, max_silence_s=cfg.loop.max_silence_s)
        self.tick = 0
        self.in_flight: dict[int, float] = {}
        self.last_state = None
        self.last_questions = None
        self.last_reason = ""
        self.events: list[dict] = []
        self.stop_requested = False
        self.started_t = time.time()
        self.live_t = None      # when the arm first went live: the 'just appeared' grace counts from here, not from process start
        self.paused = False
        self._world = None

    def event(self, kind: str, **kw):
        rec = {"t": time.time(), "kind": kind, **kw}
        self.events.append(rec); self.events = self.events[-100:]
        self.log.write("events", **rec)

    # -- operator inputs (dashboard / CLI) --------------------------------------------------------
    def set_orders(self, orders: list[str]):
        self.orders = [o.strip() for o in orders if o.strip()]
        self.event("orders", orders=self.orders)

    def set_task(self, text: str):
        self.task = text.strip()
        self.brain.reset_task()
        self.event("task", text=self.task)

    def stop(self, reason="operator STOP"):
        self.arm.freeze(reason); self.event("stop", reason=reason)

    def resume(self):
        self.arm.resume(); self.event("resume")

    # -- one tick ----------------------------------------------------------------------------------
    async def run(self, duration_s: float | None = None):
        self.log.write_json("config.json", {"config": self.cfg.as_dict(), "question_set": self.qmod.VERSION_ID,
                                            "state_version": STATE_VERSION, "model": self.cfg.loop.model})
        if self.jev:
            await self.jev.warm()
        period = 1 / self.cfg.loop.tick_hz
        t_end = time.time() + duration_s if duration_s else None
        next_t = time.perf_counter()
        try:
            while not self.stop_requested and (t_end is None or time.time() < t_end):
                self._tick()
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_t = time.perf_counter()
        finally:
            if self.jev:
                await self.jev.close()

    def _tick(self):
        self.tick += 1
        now = time.time()
        snap = self.arm.snapshot()
        entities, in_view = self.per.entities()
        world = build_world(self.cfg, snap, entities, in_view, self.per.table_z, self.task, self.orders, self.brain.state(), now,
                            loop_start=self.live_t or self.started_t)
        self._world = world
        state = render(self.cfg, world)
        questions = (self.qmod.build(self.cfg, world, self.brain) if self.qmod.VERSION != "v0" else self.qmod.build(self.cfg, world))
        self.last_state, self.last_questions = state, questions
        self.log.write("ticks", tick=self.tick, obs={"ee": snap.ee, "gripper": snap.gripper, "ext_force": snap.ext_force,
                                                      "status": snap.status, "frozen": snap.frozen,
                                                      "entities": [{"id": e.id, "label": e.label(), "xyz": e.xyz.tolist(), "h": e.height, "w": e.width,
                                                                    "color": e.color, "last_seen": e.last_seen} for e in entities],
                                                      "table_z": self.per.table_z, "raw": dict(self.per.raw), "holding": snap.holding,
                                                      "truth": ({n: self.arm.object_xy(n) for n in self.arm.mocap}
                                                                if hasattr(self.arm, "object_xy") else None)},
                       state=state, questions=questions, brain=self.brain.state(), in_flight=len(self.in_flight))
        if self.jev and not self.paused:
            # the keys actually offered in *this* request, not last tick's
            offered_now = list((questions.get("next") or {}).get("criteria", {}))
            ask, why = self.detector.should_ask(world, offered_now, now)
            if not self.cfg.loop.event_driven:
                ask, why = True, "clocked"
            if not ask:
                # an intentional skip is not silence: the situation the last answer described still holds
                self.stats.skipped_no_change += 1
                fresh = self.brain.note_skipped(now)
                self.log.write("answers", tick=self.tick, outcome="skipped_no_change", reason=why, refreshed=fresh)
            elif len(self.in_flight) < self.cfg.safety.max_in_flight:
                self.detector.mark_sent(now)
                self.brain.note_sent(self.tick)
                self.in_flight[self.tick] = now
                self.stats.sent += 1
                asyncio.get_event_loop().create_task(self._ask(self.tick, state, questions, world))
            else:
                # wanted to ask and could not: the change stays pending and the ladder keeps walking
                self.stats.skipped += 1
                self.log.write("answers", tick=self.tick, outcome="skipped_in_flight", reason=why)
        # compose every tick, whether or not anything new arrived
        goal, cap, gripper, reason = self.brain.compose(world, now)
        self.per.held_label = self.brain.held if snap.holding else None
        self.last_reason = reason
        if snap.status == "live" and self.live_t is None:
            self.live_t = time.time()
        if snap.status in ("live", "frozen", "baselining"):
            self.arm.command(goal, cap, gripper, self.brain.pitch)
        self.log.write("commands", tick=self.tick, goal=goal, speed_cap=cap, gripper=gripper, pitch=self.brain.pitch, reason=reason,
                       ladder=self.brain.state()["ladder"], prim=self.brain.prim, prim_status=self.brain.prim_status,
                       offered=self.brain.offered_keys)

    async def _ask(self, tag: int, state, questions, world):
        res = await self.jev.ask(tag, state, questions)
        sent_at = self.in_flight.pop(tag, None)
        now = time.time()
        age_ms = (now - sent_at) * 1000 if sent_at else None
        rec = {"tick": tag, "latency_ms": res.latency_ms, "status": res.status, "request_id": res.request_id,
               "upstream_ms": res.upstream_ms, "input_tokens": res.input_tokens, "age_ms": age_ms}
        if not res.ok:
            self.stats.errors += 1
            if res.error and res.error.startswith("timeout"):
                self.stats.timeouts += 1
            self._retry_if_newest(tag)
            self.log.write("answers", outcome="error", error=res.error, **rec)
            return
        self.stats.ok += 1
        self.stats.latencies.append(res.latency_ms)
        self.stats.tokens += res.input_tokens or 0
        if tag <= self.brain.last_applied_tag:
            self.stats.dropped_order += 1
            self.log.write("answers", outcome="dropped_out_of_order", answers=res.answers, **rec)
            return
        if age_ms is not None and age_ms > self.cfg.safety.stale_answer_s * 1000:
            self.stats.dropped_stale += 1
            self._retry_if_newest(tag)
            self.log.write("answers", outcome="dropped_stale", answers=res.answers, **rec)
            return
        self.stats.ages.append(age_ms or 0)
        self.brain.apply(res.answers, world, tag, age_ms or 0.0, now)
        self.log.write("answers", outcome="applied", answers=res.answers, brain=self.brain.state(), **rec)

    def _retry_if_newest(self, tag: int) -> None:
        """A request that produced no usable answer leaves this situation unanswered. Re-ask at once
        unless a newer request is already out (dropped_out_of_order does not qualify: a newer answer
        already superseded this one)."""
        if tag == self.brain.last_sent_tag:
            self.detector.retry = True

    # -- dashboard view ----------------------------------------------------------------------------
    def view(self) -> dict:
        snap = self.arm.snapshot()
        w = self._world
        return {
            "t": time.time(), "tick": self.tick,
            "arm": {"status": snap.status, "pitch": snap.pitch, "temps": [round(v) for v in snap.temps], "ee": [round(v, 3) for v in snap.ee], "setpoint": [round(v, 3) for v in snap.setpoint],
                    "goal": [round(v, 3) for v in snap.goal], "speed_cap": snap.speed_cap, "frozen": snap.frozen,
                    "ext_force": [round(v, 1) for v in snap.ext_force], "gripper": snap.gripper, "error": snap.error},
            "brain": self.brain.state(), "reason": self.last_reason,
            "judgments": {k: j.__dict__ for k, j in self.brain.judgments.items()},
            "entities": [e.__dict__ | {"xyz": list(e.xyz)} for e in (w.entities if w else [])],
            "offered": self.brain.offered_keys,
            "perception": self.per.info | {"fps": round(self.per.fps, 1), "table_z": self.per.table_z},
            "stats": self.stats.view(), "in_flight": len(self.in_flight),
            "orders": self.orders, "task": self.task, "events": self.events[-15:],
            "question_set": self.qmod.VERSION_ID, "model": self.cfg.loop.model, "paused": self.paused,
            "state": self.last_state,
        }
