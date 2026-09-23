"""E13 system under test, step 1 and step 3: the LLM Planner.

Builds the plan request (task + scene) and the escalation request (task + scene + current plan +
correction), sends them through a backend, validates the answer with the generic `plan.validate`,
and retries once with the validation errors. The OpenAI backend uses the Responses API with a strict
JSON-schema structured output, `max_retries=0` and a short timeout. Must not import the oracle.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from . import plan as P

PLAN_INSTRUCTIONS = f"""You plan for a robot arm that moves blocks. The scene lists the blocks, a tray whose slots each hold one block, two bins that hold any number, and the table. Given the user's task, say where every block must end up.

Return:
- task_reading: one sentence, the final arrangement the task asks for.
- parameters: 0 to {P.MAX_PARAMETERS} settings of this task that the user might plausibly change while the robot is working. Name each value by what it means (e.g. "left_bin", not "option_2") and give it a one-line meaning that a user's spoken correction can be matched against. The default is the value the task as given asks for. Only expose a setting if every value gives a complete arrangement.
- branches: 1 to {P.MAX_BRANCHES}. Each branch sets every parameter to one value and gives the full goal for that combination. Include the default combination (every parameter at its default), then the combinations the user is most likely to ask for.
- goal: one entry per block, every block exactly once. The destination is a tray slot, a bin, or stays_on_table. At most one block per tray slot.

Work the goals out carefully; the robot follows them exactly and nothing checks them against the task."""

ESCALATION_INSTRUCTIONS = PLAN_INSTRUCTIONS + """

The robot was already carrying out `current_plan` (its settings are in `current_settings`) when the user said `correction`. Return a new plan whose DEFAULT branch is the final arrangement the user now wants: the task as corrected. If the message changes nothing, return the current arrangement as the default. Blocks already moved can be moved again. Parameters and extra branches are optional; prepare them for likely further corrections."""


@dataclass
class LLMRequest:
    kind: str            # "plan" | "escalation"
    name: str            # schema name
    instructions: str
    input: str
    schema: dict


@dataclass
class LLMResult:
    raw: str | None
    latency_ms: float
    in_tok: int = 0
    out_tok: int = 0
    error: str | None = None
    model: str | None = None


@dataclass
class PlanOutcome:
    plan: dict | None
    valid_first: bool
    valid_after_retry: bool
    attempts: list[dict] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return sum(a["latency_ms"] for a in self.attempts)

    @property
    def in_tok(self) -> int:
        return sum(a["in_tok"] for a in self.attempts)

    @property
    def out_tok(self) -> int:
        return sum(a["out_tok"] for a in self.attempts)


def plan_request(task_text: str, scene_view: dict) -> LLMRequest:
    body = {"task": task_text, "scene": scene_view}
    return LLMRequest("plan", "plan", PLAN_INSTRUCTIONS, json.dumps(body), P.plan_schema(scene_view))


def escalation_request(task_text: str, scene_view: dict, plan: dict, current: dict, correction: str) -> LLMRequest:
    cur = P.combo_key(plan, current)
    shown = {**P.lite(plan), "current_goal": next(b["goal"] for b in plan["branches"] if P.combo_key(plan, P.settings_of(b)) == cur)}
    body = {"task": task_text, "scene": scene_view, "current_plan": shown, "current_settings": current, "correction": correction}
    return LLMRequest("escalation", "plan", ESCALATION_INSTRUCTIONS, json.dumps(body), P.plan_schema(scene_view))


def with_errors(req: LLMRequest, raw: str | None, errors: list[str]) -> LLMRequest:
    extra = ("\n\nYour previous answer was:\n" + (raw or "(no output)") +
             "\n\nIt was rejected for these reasons; fix them and answer again:\n- " + "\n- ".join(errors))
    return LLMRequest(req.kind, req.name, req.instructions, req.input + extra, req.schema)


def _parse(raw: str | None):
    if not raw:
        return None, ["empty output"]
    try:
        return json.loads(raw), []
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON: {e}"]


def get_plan(backend, req: LLMRequest, scene_view: dict, ref=None, retry: bool = True) -> PlanOutcome:
    """Call, validate, and on failure retry once with the errors. `ref` is passed through for mocks."""
    out = PlanOutcome(None, False, False)
    for attempt in range(2 if retry else 1):
        r: LLMResult = backend.call(req, ref=(ref, attempt))
        plan, errs = _parse(r.raw) if r.error is None else (None, [f"call failed: {r.error}"])
        if plan is not None:
            errs = P.validate(plan, scene_view)
        out.attempts.append({"latency_ms": r.latency_ms, "in_tok": r.in_tok, "out_tok": r.out_tok, "errors": errs,
                             "error": r.error, "raw": r.raw, "model": r.model})
        if not errs:
            out.plan = plan
            out.valid_first = attempt == 0
            out.valid_after_retry = True
            return out
        if r.error is not None:      # a failed call (timeout, HTTP error) is not retried: no retries in this harness
            return out
        req = with_errors(req, r.raw, errs)
    return out


# ---------------------------------------------------------------- OpenAI backend

def _openai():
    try:
        import openai  # optional: `uv run --extra llm ...`
    except ImportError:
        raise SystemExit("--llm openai needs the `openai` package: cd experiments && uv run --extra llm python e13_branches.py ...")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("--llm openai needs OPENAI_API_KEY in the environment")
    return openai


def list_models(substring: str = "terra") -> list[str]:
    openai = _openai()
    client = openai.OpenAI(max_retries=0, timeout=15.0)
    return sorted(m.id for m in client.models.list() if substring in m.id)


def openai_kwargs(req: LLMRequest, model: str, max_output_tokens: int | None = None, effort: str | None = None) -> dict:
    kw = {
        "model": model,
        "instructions": req.instructions,
        "input": req.input,
        "text": {"format": {"type": "json_schema", "name": req.name, "schema": req.schema, "strict": True}},
        "store": False,
    }
    if max_output_tokens:
        kw["max_output_tokens"] = max_output_tokens
    if effort:
        kw["reasoning"] = {"effort": effort}
    return kw


class OpenAIBackend:
    def __init__(self, model: str, timeout_s: float = 30.0, max_output_tokens: int | None = None, effort: str | None = None):
        openai = _openai()
        self.model, self.max_output_tokens, self.effort = model, max_output_tokens, effort
        self.client = openai.OpenAI(max_retries=0, timeout=timeout_s)

    def call(self, req: LLMRequest, ref=None) -> LLMResult:
        t0 = time.perf_counter()
        try:
            resp = self.client.responses.create(**openai_kwargs(req, self.model, self.max_output_tokens, self.effort))
        except Exception as e:  # no retries: a failed call is recorded as a failure
            return LLMResult(None, (time.perf_counter() - t0) * 1000, error=repr(e)[:300])
        ms = (time.perf_counter() - t0) * 1000
        u = resp.usage
        err = None
        if getattr(resp, "status", "completed") != "completed":
            err = f"status {resp.status}: {getattr(resp, 'incomplete_details', None)}"
        raw = resp.output_text or None
        if raw is None and err is None:
            err = "no output text (refusal?)"
        return LLMResult(raw, ms, u.input_tokens if u else 0, u.output_tokens if u else 0, err, getattr(resp, "model", None))
