"""robojev run: wire arm backend + perception + loop + dashboard.

  robojev run --arm fake --perception virtual --task "hover over the white object"
  robojev run --arm real-ro --perception camera            # real pose + real camera, no motion
  robojev run --arm real --perception camera --i-am-at-the-estop
  robojev replay runs/<dir> --questions v1
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import replace

from aiohttp import web

from robojev.config import DEFAULT, Config
from robojev.log import RunLog
from robojev.loop import Loop, Perception


RESERVED_CAMS = {"wrist", "overhead", "third"}


def display_cameras(args):
    """`--camera NAME=URL`, repeatable: extra cameras the dashboard shows and nothing detects on.
    The name becomes a URL path segment (/cam/<name>.jpg), so it is kept to a plain word."""
    import re
    from robojev.perception.camclient import CamClient
    specs, seen = [], set()
    for spec in (args.camera or []):            # every name checked before anything is dialled
        name, eq, url = spec.partition("=")
        name, url = name.strip(), url.strip()
        if not eq or not name or not url:
            sys.exit(f"--camera wants NAME=URL (e.g. boom=http://127.0.0.1:8766), got {spec!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            sys.exit(f"--camera name {name!r} must be letters, digits, '-' or '_'")
        if name in RESERVED_CAMS:
            sys.exit(f"--camera {name}: reserved name (use --camserver for wrist, --overhead for overhead)")
        if name in seen:
            sys.exit(f"--camera {name}: given twice, names must be unique")
        seen.add(name)
        specs.append((name, url))
    out = []
    for name, url in specs:
        try:
            out.append((name, CamClient(url)))
        except Exception as e:
            sys.exit(f"--camera {name}: cannot reach {url} ({type(e).__name__}). "
                     f"Start it with scripts/camserver.sh <port>, or drop the flag.")
    return out


def build(args, cfg: Config):
    log = RunLog(name=args.run_name)
    if args.arm == "fake":
        from robojev.arm.fake import FakeArm
        arm = FakeArm(cfg)
    elif args.arm == "real-ro":
        from robojev.arm.real import RealArmReadOnly
        arm = RealArmReadOnly(cfg, log=log)
    elif args.arm == "sim":
        from robojev.arm.sim import SimArm
        arm = SimArm(cfg, log=log)
    elif args.arm == "real":
        if not args.i_am_at_the_estop:
            sys.exit("refusing to command the real arm without --i-am-at-the-estop")
        from robojev.arm.real import RealArm
        arm = RealArm(cfg, log=log)
    else:
        sys.exit(f"unknown arm {args.arm}")
    namer = None
    if args.vlm == "claude":
        from robojev.perception.vlm import ClaudeNamer
        namer = ClaudeNamer()
    elif args.vlm == "stub":
        from robojev.perception.vlm import StubNamer
        namer = StubNamer()
    disp = display_cameras(args)
    if args.perception == "virtual":
        from robojev.perception.virtual import VirtualScene
        per = Perception(cfg, arm, virtual=VirtualScene(), log=log, display=disp)
    elif args.perception == "simcam":
        from robojev.arm.sim import SimCamera
        wrist = SimCamera(arm, "cam", third=True)
        over = SimCamera(arm, "overhead", width=640, height=480)
        per = Perception(cfg, arm, cameras=[("wrist", wrist, None, True), ("overhead", over, over.extrinsic_fixed(), False)], log=log, namer=namer, display=disp, segment=args.segment)
    else:
        from robojev.perception.camclient import CamClient
        cams = [("wrist", CamClient(args.camserver), None, True)]
        if args.overhead:
            from robojev.perception.calibrate import load as load_calib
            if not args.overhead_calib:
                sys.exit("--overhead needs --overhead-calib <json> (run `robojev calibrate` first)")
            cams.append(("overhead", CamClient(args.overhead), load_calib(args.overhead_calib), False))
        per = Perception(cfg, arm, cameras=cams, log=log, namer=namer, display=disp, segment=args.segment)
    loop = Loop(cfg, arm, per, log, use_jev=not args.no_jev, task=args.task, orders=args.orders or [])
    if args.arm == "sim" and args.scenario != "static":
        from robojev.arm.sim import Scenario
        loop.scenario = Scenario(arm, args.scenario, start_s=args.scenario_start)
        loop.scenario.start()
    return arm, per, loop, log


async def serve(args, cfg):
    arm, per, loop, log = build(args, cfg)
    print(f"run log: {log.dir}")
    per.start()
    from robojev.dashboard import make_app
    runner = web.AppRunner(make_app(loop))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", args.port)
    await site.start()
    print(f"dashboard: http://127.0.0.1:{args.port}/")
    stop_evt = asyncio.Event()
    def on_sig():
        loop.stop_requested = True
        stop_evt.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, on_sig)
        except NotImplementedError:
            pass
    try:
        if args.perception in ("camera", "simcam") and args.arm != "sim":
            # let the plane fit settle so the arm's hover-start height uses the measured table
            await asyncio.sleep(1.5)
            if per._table_samples:
                object.__setattr__(cfg, "table_z", per.table_z)
                print(f"table_z from camera: {per.table_z:.3f}")
        await asyncio.to_thread(arm.start)
        await loop.run(duration_s=args.duration)
    finally:
        print("stopping: parking arm, closing log")
        try:
            arm.stop()
        finally:
            per.stop_evt.set()
            await runner.cleanup()
            log.close()
            print(f"run log: {log.dir}")


def calibrate_cmd(args, cfg):
    """Arm read-only (no motion), wrist camera detections accumulated for a few seconds, one overhead
    frame, then the solve. Place two objects where both cameras see them, arm parked or aside."""
    import time
    import numpy as np
    from robojev.arm.real import RealArmReadOnly
    from robojev.perception.camclient import CamClient
    from robojev.perception.detect import Detector
    from robojev.perception.geometry import Intrinsics
    from robojev.perception.memory import Tracker
    from robojev.perception import calibrate
    wrist, over = CamClient(args.camserver), CamClient(args.overhead)
    if args.survey:
        # move to the survey pose (camera looking across the table) and hold there: from the parked
        # pose the wrist camera sees little of the table
        from robojev.arm.real import RealArm
        arm = RealArm(cfg)
        arm.start()
        t_wait = time.time()
        while arm.snapshot().status != "live" and time.time() - t_wait < 30:
            time.sleep(0.2)
        print("arm at the survey pose:", arm.snapshot().status)
    else:
        arm = RealArmReadOnly(cfg)
        arm.start()
    try:
        time.sleep(0.5)
        det = Detector(Intrinsics(wrist.info))
        tr = Tracker()
        table = []
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            f = wrist.frame(); s = arm.snapshot()
            dets, info = det.run(f.color, f.depth_m, list(s.ee) + list(s.rot))
            if info.get("plane_z_at_origin") is not None and info.get("plane_tilt_deg", 99) < 6:
                table.append(info["plane_z_at_origin"])
            tr.update(dets)
            time.sleep(0.1)
        table_z = float(np.median(table)) if table else cfg.table_z
        ents = [(float(e.xyz[0]), float(e.xyz[1]), e.height) for e in tr.stable(min_seen=5)]
        print(f"table_z {table_z:.3f}; wrist sees {[(round(x,3), round(y,3), round(h,3)) for x, y, h in ents]}")
        fo = None
        for _ in range(10):                      # the boom server may still be warming up
            fo = over.frame()
            if fo is not None:
                break
            time.sleep(0.5)
        if fo is None:
            sys.exit("no frame from the overhead camera")
        R, t, rep = calibrate.calibrate_from_frames(fo.color, fo.depth_m, Intrinsics(over.info), table_z, ents)
        print("report:", {k: v for k, v in rep.items() if k in ("residual_m", "pairs", "yaw_deg", "n_pairs", "warning", "error", "fixed_dets")})
        if R is None:
            sys.exit("calibration failed")
        if rep.get("residual_m", 1.0) > 0.03:
            sys.exit(f"calibration not saved: residual {rep.get('residual_m')} m means the objects were paired wrongly (the two cameras must see the same two objects, and nothing else of the same height nearby)")
        if rep.get("n_pairs", 0) < 2:
            sys.exit("calibration not saved: only one object seen by both cameras, so the yaw is unknown (the fit flips between runs). Add a second object 8 cm+ tall and rerun.")
        calibrate.save(args.out, R, t, rep | {"table_z": table_z})
        print(f"saved {args.out}: camera at {np.round(t, 3).tolist()} in base frame")
    finally:
        arm.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robojev")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["fake", "sim", "real-ro", "real"], default="fake")
    r.add_argument("--perception", choices=["virtual", "simcam", "camera"], default="virtual")
    r.add_argument("--scenario", choices=["static", "drift", "intruder", "mat"], default="static")
    r.add_argument("--vlm", choices=["off", "stub", "claude"], default="off", help="slow-tier track naming (claude needs ANTHROPIC_API_KEY)")
    r.add_argument("--scenario-start", type=float, default=10.0)
    r.add_argument("--camserver", default="http://127.0.0.1:8765")
    r.add_argument("--overhead", default=None, help="second camserver URL (boom D455), e.g. http://127.0.0.1:8766")
    r.add_argument("--overhead-calib", default=None, help="calibration json from `robojev calibrate`")
    r.add_argument("--camera", action="append", default=[], metavar="NAME=URL",
                   help="extra display-only camserver, repeatable: --camera boom=http://127.0.0.1:8766")
    r.add_argument("--task", default="")
    r.add_argument("--orders", action="append")
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--no-jev", action="store_true")
    r.add_argument("--no-evade", action="store_true", help="pickup only: no default standing orders, evade/orders_violated never act")
    r.add_argument("--segment", action="store_true", help="FastSAM instance masks on the wrist camera (splits touching objects; needs the seg extra)")
    r.add_argument("--clocked", action="store_true",
                   help="one request per tick (pre-event-driven behaviour) instead of asking only on change")
    r.add_argument("--run-name", default=None)
    r.add_argument("--i-am-at-the-estop", action="store_true")
    c = sub.add_parser("calibrate", help="solve the overhead camera's pose from the table plane + objects both cameras see")
    c.add_argument("--camserver", default="http://127.0.0.1:8765")
    c.add_argument("--overhead", default="http://127.0.0.1:8766")
    c.add_argument("--out", default="overhead_calib.json")
    c.add_argument("--seconds", type=float, default=3.0, help="how long to accumulate wrist detections")

    c.add_argument("--survey", action="store_true", help="drive the arm to the survey pose first (it moves!), park after")
    p = sub.add_parser("replay")
    p.add_argument("run_dir")
    p.add_argument("--questions", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        cfg = DEFAULT
        if args.clocked:
            cfg = replace(cfg, loop=replace(cfg.loop, event_driven=False))
        if args.no_evade:
            cfg = replace(cfg, loop=replace(cfg.loop, evade_enabled=False), orders=replace(cfg.orders, default=()))
        asyncio.run(serve(args, cfg))
    elif args.cmd == "calibrate":
        calibrate_cmd(args, DEFAULT)
    else:
        from robojev.replay import replay
        asyncio.run(replay(args.run_dir, args.questions, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
