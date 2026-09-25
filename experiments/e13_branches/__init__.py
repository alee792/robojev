"""E13: LLM-prepared branches, Jev picks on a correction, low confidence escalates. See cli.py.

System under test: plan.py (schema, validator), planner.py (LLM steps 1 and 3), router.py (Jev step 2).
Evaluation side: world.py (inputs + labels), oracle.py (correct arrangements), mocks.py, score.py.
"""
