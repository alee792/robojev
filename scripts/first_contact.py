"""First real-arm run, no Jev: exercises the exact streaming path the loop uses.

Sequence: RealArm.start() (sleep -> STAGED -> gripper open -> hover start, pointing down), wait for
the force baseline, then walk a 6 cm square at 3 cm/s via mover goals, then stop() parks.
Records pose vs setpoint at 20 Hz to runs/first_contact/trace.jsonl and prints lag / smoothness.

    .venv/bin/python scripts/first_contact.py --go      (without --go it only prints the plan)
"""
import argparse, json, math, sys, time
from pathlib import Path

from robojev.config import DEFAULT
from robojev.log import RunLog

ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); ap.add_argument("--side", type=float, default=0.06)
ap.add_argument("--speed", type=float, default=0.03)
args = ap.parse_args()
cfg = DEFAULT
z = cfg.table_z + cfg.motion.safe_height
x0, y0 = cfg.motion.hover_start
corners = [(x0 + args.side, y0, z), (x0 + args.side, y0 + args.side, z), (x0, y0 + args.side, z), (x0, y0, z)]
print("plan: stage -> hover start", (x0, y0, round(z, 3)), "-> square", [tuple(round(v, 3) for v in c) for c in corners], f"at {args.speed*100:.0f} cm/s -> park")
print("workspace", cfg.workspace, "\nwatchdog trip", cfg.safety.effort_trip_n, "N")
if not args.go:
    sys.exit(0)

from robojev.arm.real import RealArm
log = RunLog(name="first_contact")
arm = RealArm(cfg, log=log)
trace = []
def sample(label):
    s = arm.snapshot()
    rec = {"t": time.time(), "label": label, "ee": s.ee, "sp": s.setpoint, "goal": s.goal, "F": s.ext_force, "status": s.status, "frozen": s.frozen}
    trace.append(rec); log.write("trace", **rec)
try:
    arm.start()
    print("staged + hover start; baselining force ...")
    t0 = time.time()
    while arm.snapshot().status == "baselining" and time.time() - t0 < 5:
        sample("baseline"); time.sleep(0.05)
    print("baseline", arm.watchdog.baseline, "status", arm.snapshot().status)
    for i, c in enumerate(corners):
        arm.command(c, args.speed)
        print(f"corner {i}: goal {tuple(round(v,3) for v in c)}")
        t1 = time.time()
        while time.time() - t1 < args.side / args.speed + 1.5:
            sample(f"leg{i}"); time.sleep(0.05)
            s = arm.snapshot()
            if s.frozen or s.status == "error":
                print("FROZEN/ERROR", s.status, s.error); break
        s = arm.snapshot()
        print(f"   arrived ee {tuple(round(v,3) for v in s.ee)} err {1000*math.dist(s.ee, c):.1f} mm  F {tuple(round(v,1) for v in s.ext_force)}")
        if s.frozen or s.status == "error":
            break
finally:
    print("parking")
    arm.stop()
    log.close()
lags = [math.dist(r["ee"], r["sp"]) for r in trace if r["label"].startswith("leg")]
if lags:
    lags.sort(); print(f"ee-vs-setpoint lag: p50 {1000*lags[len(lags)//2]:.1f} mm  p95 {1000*lags[int(len(lags)*0.95)]:.1f} mm  max {1000*lags[-1]:.1f} mm")
print("trace:", log.dir / "trace.jsonl")
