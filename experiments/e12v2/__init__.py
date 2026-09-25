"""E12 update (e12v2): a text-only, closed-loop episode simulator for the v2 design (docs/v2.md).

  core/  design logic only (events, decision request, combiner, harness loop, plan schema + validator,
         planner and decision-backend interfaces, safety filter, change detector). Imports nothing
         from sim/ or eval/; this is the part meant to be promoted into the robot harness.
  sim/   a text world implementing core's World / UserChannel / Skill interfaces.
  eval/  scenarios, the evaluation oracle, mocks, controls (rules, always-LLM), metrics, CLI.
  backends.py  live backends (Jev over HTTP, OpenAI Responses API), reusing common.py and E13.
"""
