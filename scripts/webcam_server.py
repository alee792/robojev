"""Plain USB webcam frame server (Logitech Brio and friends): colour only, no depth.

Speaks exactly the camserver protocol, so CamClient and `robojev run --camera` cannot tell the
difference apart from the missing depth image:

    .venv/bin/python scripts/webcam_server.py --index 0 --port 8767
    .venv/bin/python scripts/webcam_server.py --list          # which indices actually hand over frames

    GET /info   -> JSON: width/height/name (+ an *estimated* fx/fy/ppx/ppy, see below)
    GET /frame  -> JSON header line + JPEG colour + a zero-length depth image ("png": 0)

No sudo: this is a UVC camera read through OpenCV, not librealsense, so none of the D405 root
dance applies. The handler itself is imported from robojev.perception.camserver so the framing
can never drift from the RealSense server's.

The intrinsics in /info are a guess from --hfov, not a calibration: they exist so Intrinsics()
does not blow up, and they are fine for a display-only camera. Do not detect or calibrate off
them -- without depth there is nothing to deproject anyway.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time

import cv2

from robojev.perception.camserver import make_handler
from http.server import ThreadingHTTPServer

PERMISSION_HELP = """
No frames from the camera (it would not open, or opened and stayed silent). On macOS that is
almost always the privacy gate -- OpenCV prints "not authorized to capture video" just above:

  System Settings > Privacy & Security > Camera

and switch on the app this is running under (Terminal, iTerm, or your editor) -- the permission
belongs to the terminal app, not to python. macOS only asks once per app, so if it was refused
earlier there is no second prompt: you have to flip it by hand, then restart that terminal app
(a running process does not pick up the new grant).

Otherwise: another program may already hold the camera (Zoom, Photo Booth, FaceTime), or
--index {index} may simply be the wrong camera -- run with --list to see what answers.
"""


def open_capture(index: int, width: int, height: int, fps: int):
    """VideoCapture on the platform's native backend, sized before the first read."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # prefer the newest frame; not honoured everywhere
    except Exception:
        pass
    return cap


def make_info(index: int, w: int, h: int, hfov_deg: float) -> dict:
    """The /info a colour-only camera reports. fx/fy are a guess from the FOV, flagged as such:
    Intrinsics() needs them to exist, but there is no depth here to deproject with them."""
    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"width": w, "height": h, "fx": fx, "fy": fx, "ppx": w / 2, "ppy": h / 2,
            "model": "none", "coeffs": [0.0] * 5, "depth_scale": 0.0, "has_depth": False,
            "serial": f"index{index}", "name": f"USB webcam (index {index})",
            "intrinsics": f"estimated from hfov {hfov_deg} deg, not calibrated"}


def probe(max_index: int = 6, width: int = 1280, height: int = 720):
    """Tiny enumeration: which indices open *and* actually deliver a frame."""
    print("probing camera indices (a macOS permission prompt may appear)...", flush=True)
    usable = 0
    for i in range(max_index):
        cap = open_capture(i, width, height, 30)
        if cap is None:
            print(f"  index {i}: does not open")
            continue
        ok, frame = False, None
        t0 = time.time()
        while time.time() - t0 < 2.0 and not ok:
            ok, frame = cap.read()
        if ok and frame is not None:
            usable += 1
            print(f"  index {i}: {frame.shape[1]}x{frame.shape[0]}  <- usable")
        else:
            print(f"  index {i}: opens but no frames (permission, or in use by another app)")
        cap.release()
    if not usable:      # "not authorized to capture video" from OpenCV lands here
        print(PERMISSION_HELP.format(index="<n>"))


class Capture:
    """Same shape as camserver.Capture: .info, .lock, .latest = (t, jpg, png), .n."""

    def __init__(self, index: int, width: int, height: int, fps: int, hfov_deg: float, quality: int):
        self.cap = open_capture(index, width, height, fps)
        if self.cap is None:
            sys.exit(f"cannot open camera index {index}." + PERMISSION_HELP.format(index=index))
        self.quality = quality
        self.lock = threading.Lock()
        self.latest = None
        self.n = 0
        ok, frame = False, None
        t0 = time.time()                      # the grant, if it is coming, arrives within a second
        while time.time() - t0 < 6.0 and not ok:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
        if not ok or frame is None:
            self.cap.release()
            sys.exit(PERMISSION_HELP.format(index=index))
        h, w = frame.shape[:2]
        self.info = make_info(index, w, h, hfov_deg)
        self._store(frame)
        threading.Thread(target=self._run, daemon=True).start()

    def _store(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.lock:
                self.latest = (time.time(), jpg.tobytes(), b"")   # b"" -> header says "png": 0
                self.n += 1

    def _run(self):
        misses = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                misses += 1
                if misses in (30, 300):       # unplugged mid-run: say so once, keep serving the last frame
                    print(f"no frame for {misses} reads (camera unplugged or taken over?)", flush=True)
                time.sleep(0.05)
                continue
            misses = 0
            self._store(frame)


def main():
    ap = argparse.ArgumentParser(description="colour-only webcam server speaking the camserver protocol")
    ap.add_argument("--index", type=int, default=0, help="OpenCV camera index (see --list)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--width", type=int, default=1280, help="the Brio can do 4K; 720p keeps the JPEGs small")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--hfov", type=float, default=78.0, help="horizontal FOV used for the estimated intrinsics")
    ap.add_argument("--list", action="store_true", help="probe camera indices and exit")
    ap.add_argument("--max-index", type=int, default=6, help="how many indices --list walks")
    args = ap.parse_args()
    if args.list:
        probe(max_index=args.max_index, width=args.width, height=args.height)
        return
    cap = Capture(args.index, args.width, args.height, args.fps, args.hfov, args.quality)
    print("camera:", cap.info, flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cap))
    print(f"serving http://127.0.0.1:{args.port}/frame  (colour only, ctrl-c to stop)", flush=True)
    t0 = time.time()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(5)
            print(f"{cap.n} frames, {cap.n / max(1e-9, time.time() - t0):.1f} fps", flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
