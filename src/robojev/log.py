"""Run logs: append-only JSONL, one directory per run, everything needed for offline replay."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, root: Path | str = "runs", name: str | None = None):
        stamp = name or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {}

    def path(self, stream: str) -> Path:
        return self.dir / f"{stream}.jsonl"

    def write(self, stream: str, **rec) -> None:
        f = self._files.get(stream)
        if f is None:
            f = self._files[stream] = self.path(stream).open("a")
        rec.setdefault("t", time.time())
        f.write(json.dumps(rec, default=_default) + "\n")
        f.flush()

    def write_json(self, name: str, obj) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=_default))

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:
        pass
    return str(o)


def read_jsonl(path: Path | str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
