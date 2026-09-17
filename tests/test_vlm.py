import time
import numpy as np
from robojev.perception.vlm import StubNamer, NamingWorker, crop_around


def test_stub_worker_names_once():
    w = NamingWorker(StubNamer()); w.start()
    img = np.zeros((480, 640, 3), np.uint8)
    w.submit("A", crop_around(img, (320, 240)), 0.11, 0.08, "cup-like object", "white")
    w.submit("A", crop_around(img, (320, 240)), 0.11, 0.08, "cup-like object", "white")  # duplicate ignored
    t0 = time.time()
    while not w.results and time.time() - t0 < 2:
        time.sleep(0.05)
    res = w.take()
    assert list(res) == ["A"] and res["A"].kind == "cup" and "cup" in res["A"].name
    w.stop_evt.set()
