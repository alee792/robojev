"""Versioned question batteries. Each module exposes VERSION and build(cfg, world) -> dict."""
import importlib


def load(name: str):
    return importlib.import_module(f"robojev.questions.{name}")
