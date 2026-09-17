"""Standalone D405 frame server. Run this (with sudo on macOS, where librealsense needs root to
claim the UVC interface) and the loop reads frames over localhost; nothing else runs as root.

    sudo .venv/bin/python -m robojev.perception.camserver [--port 8765] [--serial N]

Serves:
  GET /frame   -> JSON header line + JPEG colour + 16-bit PNG depth (aligned to colour), see client
  GET /info    -> intrinsics, depth scale, resolution
Frames are captured in a thread; each request returns the latest one (never blocks on the camera).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

W, H, FPS = 640, 480, 30


class Capture:
    def __init__(self, serial: str | None):
        import pyrealsense2 as rs
        self.rs = rs
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        dev = prof.get_device()
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.info = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                     "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs),
                     "depth_scale": self.depth_scale, "serial": dev.get_info(rs.camera_info.serial_number),
                     "name": dev.get_info(rs.camera_info.name)}
        self.lock = threading.Lock()
        self.latest = None  # (t, jpeg bytes, png bytes)
        self.n = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                frames = self.align.process(self.pipe.wait_for_frames(5000))
            except Exception as e:
                print("capture error", e, flush=True)
                time.sleep(0.2)
                continue
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue
            color = np.asanyarray(c.get_data())
            depth = np.asanyarray(d.get_data())
            ok1, jpg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok2, png = cv2.imencode(".png", depth)
            if ok1 and ok2:
                with self.lock:
                    self.latest = (time.time(), jpg.tobytes(), png.tobytes())
                    self.n += 1


def make_handler(cap: Capture):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/info"):
                body = json.dumps(cap.info).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            if self.path.startswith("/frame"):
                with cap.lock:
                    latest, n = cap.latest, cap.n
                if latest is None:
                    self.send_response(503); self.end_headers(); return
                t, jpg, png = latest
                header = json.dumps({"t": t, "n": n, "jpg": len(jpg), "png": len(png)}).encode() + b"\n"
                body = header + jpg + png
                self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            self.send_response(404); self.end_headers()
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()
    cap = Capture(args.serial)
    print("camera:", cap.info, flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cap))
    print(f"serving http://127.0.0.1:{args.port}/frame  (ctrl-c to stop)", flush=True)
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
