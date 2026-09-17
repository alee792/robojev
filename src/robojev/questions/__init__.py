"""Versioned question batteries. Each module exposes VERSION and build(cfg, world) -> dict."""
import hashlib
import importlib
import inspect


def load(name: str):
    mod = importlib.import_module(f"robojev.questions.{name}")
    # a stable label on a changing battery defeats replay comparisons: tag the version with a
    # content hash of the module source
    digest = hashlib.sha1(inspect.getsource(mod).encode()).hexdigest()[:8]
    mod.VERSION_ID = f"{mod.VERSION}@{digest}"
    return mod
