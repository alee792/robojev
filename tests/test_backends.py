"""The live backends (robojev.brain.backends) against mocked HTTP: what goes on the wire, and that the
answers come back in the shape combine() reads. No network, no keys."""
from __future__ import annotations

import json

import httpx
import pytest

from robojev.brain.backends import BudgetExceeded
from robojev.brain.backends.decisions_api import DecisionsAPI, choice_confidence, from_answers, to_questions
from robojev.brain.backends.jev import JevHTTP
from robojev.brain.backends.openai_planner import OpenAIPlanner
from robojev.brain.core.combine import Gates, combine
from robojev.brain.core.data import Event
from robojev.brain.core.planner import LLMRequest

STATE = {"event": {"kind": "user text", "what_happened": "the user typed", "user_just_said": "actually, highest on the left"},
         "plan_position": {"arm_in_step": "carrying block 1"}}
QUESTIONS = {
    "right_now": {"type": "choice",
                  "instructions": {"question": "Given `event`, what should the arm do right now?", "context": "literal facts"},
                  "criteria": {"carry_on": {"what": "No change.", "not_for": "a hand"}, "hold": {"what": "Stop at a safe point."},
                               "pause": {"what": "Stop until told."}}},
    "route": {"type": "choice", "instructions": {"question": "Who should handle `event`?", "ask_about": "whether the end result must change"},
              "criteria": {"stay_local": {"what": "Plan holds."}, "fast_llm": {"what": "Plan must change."}}},
    "check.block_2": {"type": "noul", "instructions": "Is block 2 something the task asks about?",
                      "criteria": {"true": "The task covers it.", "false": "It does not."}},
}


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def decisions_reply(answers, status=200):
    def handler(req: httpx.Request):
        handler.sent = json.loads(req.content)
        return httpx.Response(status, json={"model": "gpt-6-luna", "answers": answers, "usage": {"input_tokens": 312}},
                              headers={"x-request-id": "req_1", "openai-processing-ms": "69"})
    return handler


ANSWERS = [
    {"type": "choice", "name": "right_now", "choice": "hold", "confidence": 0.85,
     "probabilities": [{"value": "carry_on", "probability": 0.04}, {"value": "hold", "probability": 0.9},
                       {"value": "pause", "probability": 0.06}]},
    {"type": "choice", "name": "route", "choice": "fast_llm", "confidence": 0.9,
     "probabilities": [{"value": "stay_local", "probability": 0.05}, {"value": "fast_llm", "probability": 0.95}]},
    {"type": "predicate", "name": "check.block_2", "probability": 0.1},
]


# ---------------------------------------------------------------- Decisions API


def test_decisions_sends_every_group_in_one_request():
    h = decisions_reply(ANSWERS)
    DecisionsAPI(client=client(h)).answer(STATE, QUESTIONS)
    sent = h.sent
    assert sent["model"] == "gpt-6-luna"
    assert [q["name"] for q in sent["questions"]] == ["right_now", "route", "check.block_2"]
    rn = sent["questions"][0]
    assert rn["type"] == "choice" and [c["value"] for c in rn["choices"]] == ["carry_on", "hold", "pause"]
    assert rn["choices"][0]["description"] == "No change. Not for: a hand."
    assert "Context: literal facts" in rn["instructions"] and "`event`" in rn["instructions"]
    assert "whether the end result must change" in sent["questions"][1]["instructions"]
    pred = sent["questions"][2]
    assert pred["type"] == "predicate" and "True means: The task covers it." in pred["instructions"]
    part = sent["input"][0]["content"][0]
    assert part["type"] == "input_text" and json.loads(part["text"].split("\n", 1)[1]) == STATE


def test_decisions_answers_come_back_in_jevs_shape_with_both_confidences():
    r = DecisionsAPI(client=client(decisions_reply(ANSWERS))).answer(STATE, QUESTIONS)
    a = r["answers"]
    assert a["right_now"]["choice"] == "hold" and a["right_now"]["probabilities"] == {"carry_on": 0.04, "hold": 0.9, "pause": 0.06}
    assert a["right_now"]["confidence_api"] == 0.85
    assert a["right_now"]["confidence_formula"] == pytest.approx((0.9 - 1 / 3) / (2 / 3))
    assert a["right_now"]["confidence"] == a["right_now"]["confidence_formula"]          # default: the formula the gates were set on
    assert a["check.block_2"] == {"type": "noul", "noul": 0.1}
    assert r["in_tok"] == 312 and r["meta"]["request_id"] == "req_1" and r["meta"]["upstream_ms"] == 69.0
    api = DecisionsAPI(client=client(decisions_reply(ANSWERS)), confidence="api").answer(STATE, QUESTIONS)
    assert api["answers"]["route"]["confidence"] == 0.9


def test_decisions_answers_feed_combine():
    r = DecisionsAPI(client=client(decisions_reply(ANSWERS))).answer(STATE, QUESTIONS)

    class V:
        hand = None

        def name(self, oid):
            return oid
    c = combine(Event(1, 0, "user_text", "typed"), r["answers"], V(), Gates())
    assert (c.right_now, c.route) == ("hold", "fast_llm") and c.problems == []


def test_a_refused_question_is_left_out_and_reads_as_no_confidence():
    answers = [ANSWERS[0] | {}, {"type": "refusal", "name": "route", "refusal": "no"}]
    r = DecisionsAPI(client=client(decisions_reply(answers))).answer(STATE, QUESTIONS)
    assert "route" not in r["answers"] and r["meta"]["refusals"] == ["route"]


def test_decisions_errors_come_back_as_no_answers():
    r = DecisionsAPI(client=client(decisions_reply([], status=500))).answer(STATE, QUESTIONS)
    assert r["answers"] is None and r["meta"]["status"] == 500

    def boom(req):
        raise httpx.ConnectTimeout("slow")
    be = DecisionsAPI(client=client(boom))
    r = be.answer(STATE, QUESTIONS)
    assert r["answers"] is None and "ConnectTimeout" in r["meta"]["error"] and be.errors == 1


def test_decisions_attaches_images_inline():
    h = decisions_reply(ANSWERS)
    DecisionsAPI(client=client(h)).answer(STATE, QUESTIONS, images=[b"\x89PNG"])
    img = h.sent["input"][0]["content"][1]
    assert img == {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="}
    with pytest.raises(ValueError):
        DecisionsAPI(client=client(h)).answer(STATE, QUESTIONS, images=["https://example.com/a.png"])


def test_decisions_budget():
    be = DecisionsAPI(client=client(decisions_reply(ANSWERS)), max_requests=1)
    be.answer(STATE, QUESTIONS)
    with pytest.raises(BudgetExceeded):
        be.answer(STATE, QUESTIONS)


def test_translation_helpers():
    assert choice_confidence({"a": 0.5, "b": 0.5}) == 0.0 and choice_confidence({"a": 1.0}) == 0.0
    with pytest.raises(ValueError):
        to_questions({"s": {"type": "score", "instructions": "x"}})
    out, refused = from_answers([{"type": "choice", "name": "x", "choice": "a", "probabilities": [{"value": "a", "probability": 1.0},
                                                                                                 {"value": "b", "probability": 0.0}]}])
    assert out["x"]["confidence"] == 1.0 and out["x"]["confidence_api"] is None and refused == []


# ---------------------------------------------------------------- Jev


def test_jev_posts_the_request_as_is_and_returns_its_answers():
    def handler(req):
        handler.sent = json.loads(req.content)
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"right_now": {"type": "choice", "choice": "hold",
                                                                                          "confidence": 0.59}},
                                         "usage": {"input_tokens": 401}}, headers={"x-typesafe-request-id": "j1"})
    r = JevHTTP(client=client(handler)).answer(STATE, QUESTIONS)
    assert handler.sent == {"state": STATE, "model": "jev-1.13.0", "questions": QUESTIONS}
    assert r["answers"]["right_now"]["choice"] == "hold" and r["in_tok"] == 401 and r["meta"]["request_id"] == "j1"


def test_jev_errors_come_back_as_no_answers():
    r = JevHTTP(client=client(lambda req: httpx.Response(529, json={"error": "overloaded"}))).answer(STATE, QUESTIONS)
    assert r["answers"] is None and r["meta"]["status"] == 529

    def boom(req):
        raise httpx.ReadTimeout("slow")
    r = JevHTTP(client=client(boom)).answer(STATE, QUESTIONS)
    assert r["answers"] is None and "ReadTimeout" in r["meta"]["error"]


# ---------------------------------------------------------------- the planner

REQ = LLMRequest("plan", "plan", "You plan.", '{"request": "plan"}', {"type": "object", "properties": {}, "required": [],
                                                                       "additionalProperties": False})


def responses_reply(body, status=200):
    def handler(req):
        handler.sent = json.loads(req.content)
        return httpx.Response(status, json=body)
    return handler


def test_planner_sends_strict_schema_and_cache_key_and_reads_usage():
    h = responses_reply({"status": "completed", "model": "gpt-6-luna",
                         "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": '{"steps": []}'}]}],
                         "usage": {"input_tokens": 2300, "output_tokens": 440, "input_tokens_details": {"cached_tokens": 2000},
                                   "output_tokens_details": {"reasoning_tokens": 20}}})
    r = OpenAIPlanner(client=client(h), effort="low").call(REQ)
    s = h.sent
    assert s["text"]["format"] == {"type": "json_schema", "name": "plan", "schema": REQ.schema, "strict": True}
    assert s["prompt_cache_key"] == "robojev-planner" and s["reasoning"] == {"effort": "low"} and s["store"] is False
    assert s["instructions"] == "You plan." and s["input"] == REQ.input
    assert (r.raw, r.in_tok, r.out_tok, r.cached_tok, r.reasoning_tok, r.error) == ('{"steps": []}', 2300, 440, 2000, 20, None)


def test_planner_failures_are_errors_not_answers():
    r = OpenAIPlanner(client=client(responses_reply({"error": {"message": "bad"}}, 400))).call(REQ)
    assert r.raw is None and r.error.startswith("HTTP 400")
    r = OpenAIPlanner(client=client(responses_reply({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                                                     "output": []}))).call(REQ)
    assert r.error.startswith("status incomplete")
    r = OpenAIPlanner(client=client(responses_reply({"status": "completed", "output": [
        {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}))).call(REQ)
    assert r.error == "refusal: no"
    p = OpenAIPlanner(client=client(responses_reply({})), max_calls=0)
    with pytest.raises(BudgetExceeded):
        p.call(REQ)
