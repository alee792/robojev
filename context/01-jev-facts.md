# Jev facts that bind a control loop

All **[verified]** by reading the mirrored docs on 2026-09-17 unless marked. Paths relative to
`sources/typesafe-docs/`. Model at time of writing: `jev-1.13.0` (docs "last reviewed 2026-09-16").

## Contract
- `POST https://api.typesafe.ai/v1/systemone` with `{state, model, questions}` → `{model, answers, usage}`. (`api.md`)
- `state`: string | object | array. **Text only** — no images/audio/video "yet". (`concepts/system-one.md`, `concepts/state.md`)
- Question types: `noul` → `{noul: P(yes)}` (**no confidence**); `choice` → `{choice, probabilities, confidence}`; `score` → `{score (0..n-1, can be between levels), legend, probabilities, confidence}`. (`api.md`)
- `instructions` and `criteria` accept structured JSON (objects/arrays), e.g. `{what, not_for, examples}` per option. (`primitives/advanced.md`, `concepts/how-to-build-with-system-one.md`)
- Question **keys are not sent to the model** — put meaning in `instructions`. (`api.md`)
- Choice: **up to 255 options**. (`primitives/choice.md`)
- Questions in one request see the same state and are **evaluated independently and in parallel**; adding questions "typically doesn't add any latency". One answer never becomes context for another. (`patterns/fan-out.md`, `concepts/state.md`)
- Reference nested state with backticked paths in instructions: `` `support.tickets[0].message` ``. (`concepts/how-to-build-with-system-one.md`)
- ⚠️ SDK default retry policy (2 retries, backoff up to 5 s, 10 s per-attempt timeout, 30 s budget) is unsafe for a control loop — see `09-sim-and-sdk-notes.md`.
- Missing key returns **403** in practice (docs say 401). Measured latency/rate behavior: `08-experiment-results.md`.
- SDK: `pip install typesafe-sdk` (sync `TypeSafeClient`, async `AsyncTypeSafeClient`), `result.nouls/choices/scores[...]`, `RetryPolicy(max_retries, backoff_max, timeout)`, default HTTP timeout 10 s, `extra_body`/raw dict passthrough, `raw_http_response`. (`sdk/python-usage.md`, `sdk/python-constants.md`)

## Limits
- **Context:** 64k tokens for all `state`+`questions`; 32k for `state` + longest question. (`jaggedness-jev-1.13.md`)
- **Rate:** 250,000 tok/s and **1,200 req/min (= 20 req/s)**; "adjusting dynamically… can change without notice". 429 → backoff, honors `retry-after`. 529 = overloaded. (`models.md`, `api.md`)
- **Latency:** "Most queries complete in about 100 ms" (`concepts/how-to-build-with-system-one.md`); blog: 70–500 ms end-to-end, measured from US West Coast where the service runs. Demo header: 92–117 ms.
- **Price:** $0.042 / M input tokens; output free. (`models.md`)

## Model behavior ("jaggedness", `jaggedness-jev-1.13.md`)
- Literal reading — write the exact condition; put boundary cases in criteria.
- **Bad at math, counting, numeric closeness, date comparison.** Keep arithmetic in code; pass named buckets. Don't interpolate between score levels to recover magnitudes. *[measured nuance: simple integer-cm comparisons among ≤10 objects were ~96–100% accurate, and bands-only was worse because it's lossy — see `08` §E6.]*
- Weak at indirection / multi-hop.
- Accuracy drops with irrelevant state ("context rot") → filter state per question set.
- Adversarial state can steer it (relevant if NL user text is placed in state).
- Instruction/criteria contradictions hurt (e.g. inverted noul criteria).
- Not a generator.

## Closing the open questions from `sources/jev-widowx-sources.md` §7
1. **Confidence** — **empirically confirmed for Choice: `(p_max − 1/n)/(1 − 1/n)`** (see `08-experiment-results.md` §E4; probabilities are rounded to 2 dp). Docs: a statistic computed from the returned distribution (flatter → lower); formula not published ("separate cookbook… later"). Calibration is claimed "across groups of predictions; it does not guarantee that an individual answer is correct." Docs suggest 3 bands (act / caution / don't act) with thresholds scaled to stakes, and pinning a model version once thresholds are tuned. ⇒ **Usable as a gate for *which option to act on*, not as a safety mechanism.** Safety must be in code. (`confidence.md`, `models.md`, `concepts/system-one.md`)
2. **State shape** — prefer a JSON object with descriptive names; group related facts; send only what the questions need; nested + backticked paths are fine. (`concepts/state.md`)
3. **Models** — only `jev-1.13.0` (`jev-latest` = `jev-preview` = 1.13.0). No smaller/faster tier. Aliases move silently; response `model` reports the actual ID. (`models.md`)
4. Trossen mode model — **deferred** (hardware out of scope this session).
5. **Pricing/rate limits** — see Limits. At ~6k tok/call: 10 Hz ≈ $9/hr, ≈ half of the 20 rps ceiling. Two 10 Hz loops saturate it.

## Useful tools & repos [verified exist]
- `github.com/typesafe-ai/system-one-adapter-python` — drop-in `system_one` API backed by OpenAI/Anthropic LLMs. Use for offline dev, baselines, or an LLM "judge" slot. (`sources/system-one-adapter-README.md`)
- TypeSafe agent skill: `claude plugin marketplace add typesafe-ai/skills` + `claude plugin install typesafe@typesafe-ai`. Copy at `sources/typesafe-skill/SKILL.md`.
- Full doc corpus: `sources/typesafe-docs/llms-full.txt` (~800 KB); index `llms.txt`.
- Relevant cookbooks mirrored: `cookbooks/function_calling.md` (NL → typed function args), `consistency_*` (self-consistency), `classification_using_confidence.md`, `parallel_questions.md`, `hierarchical_classification.md` (beam search over >255 options).
