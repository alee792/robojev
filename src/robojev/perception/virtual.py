"""Virtual perception: scripted entities in base frame, for dry runs against fake/sim arms."""
from __future__ import annotations

import math
import time

from robojev.perception.detect import Detection


class VirtualScene:
    def __init__(self, table_z: float = -0.025):
        self.table_z = table_z
        self.t0 = time.time()
        self.objects = [
            {"xy": (0.36, 0.05), "h": 0.11, "w": 0.08, "color": "white", "drift": (0.0, 0.0)},
            {"xy": (0.30, -0.12), "h": 0.02, "w": 0.15, "color": "black", "drift": (0.0, 0.0)},
        ]

    def detections(self, now: float | None = None) -> list[Detection]:
        now = now or time.time()
        out = []
        for o in self.objects:
            x = o["xy"][0] + o["drift"][0] * math.sin((now - self.t0) / 6)
            y = o["xy"][1] + o["drift"][1] * math.sin((now - self.t0) / 6)
            out.append(Detection(xyz=(x, y, self.table_z + o["h"] / 2), base_xyz=(x, y, self.table_z), height=o["h"],
                                 width=o["w"], color_bgr=(200, 200, 200), color_name=o["color"], n_points=200, pixel=(0, 0)))
        return out
