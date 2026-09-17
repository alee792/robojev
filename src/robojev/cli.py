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

from aiohttp import web

from robojev.config import DEFAULT, Config
from robojev.log import RunLog
from robojev.loop import Loop, Perception


def build(args, cfg: Config):
    log = RunLog(name=args.run_name)
    if args.arm == "fake":
        from robojev.arm.fake import FakeArm
        arm = FakeArm(cfg)
    elif args.arm == "real-ro":
        from robojev.arm.real import RealArmReadOnly
        arm = RealArmReadOnly(cfg, log=log)
    elif args.arm == "real":
        if not args.i_am_at_the_estop:
            sys.exit("refusing to command the real arm without --i-am-at-the-estop")
        from robojev.arm.real import RealArm
        arm = RealArm(cfg, log=log)
    else:
        sys.exit(f"unknown arm {args.arm}")
    if args.perception == "virtual":
        from robojev.perception.virtual import VirtualScene
        per = Perception(cfg, arm, virtual=VirtualScene(), log=log)
    else:
        from robojev.perception.camclient import CamClient
        per = Perception(cfg, arm, camera=CamClient(args.camserver), log=log)
    loop = Loop(cfg, arm, per, log, use_jev=not args.no_jev, task=args.task, orders=args.orders or [])
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
        if args.perception == "camera":
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robojev")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["fake", "real-ro", "real"], default="fake")
    r.add_argument("--perception", choices=["virtual", "camera"], default="virtual")
    r.add_argument("--camserver", default="http://127.0.0.1:8765")
    r.add_argument("--task", default="")
    r.add_argument("--orders", action="append")
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--no-jev", action="store_true")
    r.add_argument("--run-name", default=None)
    r.add_argument("--i-am-at-the-estop", action="store_true")
    p = sub.add_parser("replay")
    p.add_argument("run_dir")
    p.add_argument("--questions", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        asyncio.run(serve(args, DEFAULT))
    else:
        from robojev.replay import replay
        asyncio.run(replay(args.run_dir, args.questions, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
