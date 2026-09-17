"""Client for camserver: fetch the latest colour + aligned depth frame over localhost."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
import httpx


@dataclass
class Frame:
    t: float
    n: int
    color: np.ndarray            # HxWx3 BGR
    depth_m: np.ndarray | None   # HxW float32 metres (0 = no reading); None on a colour-only camera


class CamClient:
    def __init__(self, base: str = "http://127.0.0.1:8765", timeout: float = 1.0):
        self.base = base
        self.http = httpx.Client(timeout=timeout)
        self.info = self.http.get(base + "/info").json()
        # a plain webcam (scripts/webcam_server.py) has no depth at all: no scale, no depth image
        self.scale = float(self.info.get("depth_scale") or 0.0)
        self.has_depth = bool(self.info.get("depth_scale"))

    def frame(self) -> Frame | None:
        r = self.http.get(self.base + "/frame")
        if r.status_code != 200:
            return None
        raw = r.content
        nl = raw.index(b"\n")
        hdr = json.loads(raw[:nl])
        jpg = raw[nl + 1: nl + 1 + hdr["jpg"]]
        png = raw[nl + 1 + hdr["jpg"]: nl + 1 + hdr["jpg"] + hdr["png"]]
        color = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        depth = None
        if hdr["png"]:
            d = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
            if d is not None:
                depth = d.astype(np.float32) * self.scale
        return Frame(hdr["t"], hdr["n"], color, depth)


if __name__ == "__main__":
    c = CamClient()
    print(c.info)
    f = c.frame()
    centre = (float(f.depth_m[f.depth_m.shape[0] // 2, f.depth_m.shape[1] // 2])
              if f.depth_m is not None else "no depth (colour-only camera)")
    print("frame", f.n, f.color.shape, "depth centre m", centre)
    cv2.imwrite("cam_color.jpg", f.color)
