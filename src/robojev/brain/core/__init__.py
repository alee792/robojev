"""e12v2's design logic (experiments/e12v2/core), copied: events and world state (data), the decision
request (decision), the combiner and its gates (combine), the plan schema, validator and diffs (plan),
the planner request (planner), the change detector (changes), the safety filter (safety) and the seams
(interfaces). Imports nothing outside this package. e12v2's tick harness is not here: the brain's
asyncio loop (robojev.brain.loop) replaces it, and the rules it kept (is_stop, LoopDetector) are in
robojev.brain.rules."""
