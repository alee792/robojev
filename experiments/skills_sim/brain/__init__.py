"""H1 + M1: the brain as a client of the robot protocol, on a real clock with async events.

loop.py    Brain: an asyncio event loop around e12v2's pure pieces (decision request, combiner, plan
           format and validator, change detector); everything slow runs in threads.
schema.py  the planner's step schema and robot description, generated from the server's manifest.
adapt.py   protocol.WorldState (m, many arms/hands) -> e12v2's core WorldState (cm, one arm/hand).
Imports nothing hardware-specific: only protocol.py and e12v2.core.
"""
from .loop import Brain, BrainConfig, EpisodeResult
from .schema import ManifestPlanner, describe_robot

__all__ = ["Brain", "BrainConfig", "EpisodeResult", "ManifestPlanner", "describe_robot"]
