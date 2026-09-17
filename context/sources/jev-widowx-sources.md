# Primary sources: Jev + WidowX robotics harness

Collated for a clean session. Goal: build a harness that uses TypeSafe's Jev (a System One
decision model) to drive a Trossen WidowX AI arm.

Verification key:
- **[verified]** — fetched and read this session
- **[confirmed-url]** — URL confirmed to exist via search/index, contents not read
- **[unverified]** — believed correct from general knowledge, check it first

---

## 0. Start here: the two doc entry points

| What | URL | Note |
|---|---|---|
| Trossen Arm docs **MCP server** | `https://mcp.docs.trossenrobotics.com/mcp` | **[verified]** Hosted, no install. Serves the whole Trossen doc set + C++/Python API reference + demo scripts directly to Claude Code. Add this first. |
| TypeSafe docs index (`llms.txt`) | https://docs.typesafe.ai/llms.txt | **[confirmed-url]** The complete doc index in LLM-readable form. Fetch before exploring further. |

Adding the Trossen MCP server means you do not have to scrape their docs site at all.

---

## 1. Jev / TypeSafe AI

The model: you send a `state` (string/object/array) plus a map of named, typed `questions`,
and get one typed answer per question with a calibrated probability distribution. Three
question types: `noul` (P(true)), `choice` (option + distribution + confidence), `score`
(probability-weighted float across ordered levels + legend + distribution + confidence).
No text generation, no action space.

### Documentation
| Page | URL | Status |
|---|---|---|
| Doc index (llms.txt) | https://docs.typesafe.ai/llms.txt | [confirmed-url] |
| Introduction | https://docs.typesafe.ai/introduction | [confirmed-url] |
| **HTTP API reference** | https://docs.typesafe.ai/api | **[verified]** (pasted into session) |
| Primitives | https://docs.typesafe.ai/primitives | [confirmed-url] |
| State — formats & best practices | https://docs.typesafe.ai/concepts/state | [confirmed-url] |
| Confidence — how it's derived | https://docs.typesafe.ai/confidence | [confirmed-url] |
| Models & aliases | https://docs.typesafe.ai/models | [confirmed-url] |
| Python SDK docs | https://docs.typesafe.ai/sdk/python/ | [confirmed-url] |
| Python SDK changelog | https://docs.typesafe.ai/sdk/python/changelog/ | [confirmed-url] |
| API keys console | https://console.typesafe.ai/settings/keys | [confirmed-url] |

**Read `/confidence` and `/concepts/state` early.** Confidence gating and state serialization
are both load-bearing for a control harness and neither was reachable this session.

### Code
| What | URL | Status |
|---|---|---|
| Python SDK (PyPI) | `pip install typesafe-sdk` — v0.6.0, requires-python >=3.10 | **[verified]** |
| Python SDK source | https://github.com/typesafe-ai/typesafe-sdk-python | [confirmed-url] |

### The wire contract (verified)
```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
Content-Type: application/json
```
```json
{"state": <string|object|array>,
 "model": "jev-latest",
 "questions": {"<id>": {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}}
```
Response: `{"model": ..., "answers": {"<id>": {...}}, "usage": {input_tokens, output_tokens}}`

Facts that matter for a control loop:
- Every question in one request sees the same state and is answered **independently** — batch
  generously, ~12 questions costs about what 1 costs.
- `score` lands in **`[0, n-1]`**, zero-indexed. The answer returns a `legend` mapping level
  index → description, so derive the midpoint from the response, not from what you sent.
- `score` `criteria` needs **at least two levels**.
- `noul` carries **no** `confidence` field; `choice` and `score` do.
- `instructions` accepts `string | object | array` — structured context is allowed.
- Errors: `401`, `422`, `429` (rate limit), `529` (overloaded). 429/529 want backoff.

---

## 2. Trossen Arm driver (the layer that actually moves the arm)

This is the WidowX AI's native driver — Python wrapper over the `libtrossen_arm` C++ library,
talking to the arm controller over wired ethernet.

| What | URL | Status |
|---|---|---|
| **Docs MCP server** | `https://mcp.docs.trossenrobotics.com/mcp` | **[verified]** |
| Docs root | https://docs.trossenrobotics.com/trossen_arm/main/index.html | [confirmed-url] |
| Getting started | https://docs.trossenrobotics.com/trossen_arm/main/getting_started.html | [confirmed-url] |
| Software setup | https://docs.trossenrobotics.com/trossen_arm/main/getting_started/software_setup.html | [confirmed-url] |
| Configuration (incl. networking) | https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html | [confirmed-url] |
| Demo scripts | https://docs.trossenrobotics.com/trossen_arm/main/getting_started/demo_scripts.html | [confirmed-url] |
| **Programming guide → concepts** | https://docs.trossenrobotics.com/trossen_arm/main/programming_guide/concepts.html | [confirmed-url] |
| Programming guide → writing a script | https://docs.trossenrobotics.com/trossen_arm/main/programming_guide/writing_a_script.html | [confirmed-url] |
| **C++/Python API reference** | under https://docs.trossenrobotics.com/trossen_arm/main/programming_guide.html | [confirmed-url] |
| CLI tool | https://docs.trossenrobotics.com/trossen_arm/main/software_tools/cli.html | [confirmed-url] |
| MCP server setup | https://docs.trossenrobotics.com/trossen_arm/main/software_tools/mcp_server.html | [confirmed-url] |
| Specifications | https://docs.trossenrobotics.com/trossen_arm/main/specifications.html | [confirmed-url] |
| Troubleshooting | https://docs.trossenrobotics.com/trossen_arm/main/troubleshooting.html | [confirmed-url] |
| Changelog | https://docs.trossenrobotics.com/trossen_arm/main/changelog.html | [confirmed-url] |
| Source | https://github.com/TrossenRobotics/trossen_arm | **[verified]** |
| PyPI | `trossen-arm` — latest **1.11.0** | **[verified]** |
| WidowX AI product page | https://www.trossenrobotics.com/widowx-ai | [confirmed-url] |

**Version pinning matters:** the driver and the controller firmware must share major.minor.
Docs are versioned in the URL (`/main/`, `/v1.6/`, …) — use the path matching your pinned
driver, not `main`, if they differ.

### Driver API surface (verified from the SDK headers)
Write side: `set_all_modes`, `set_all_positions`, `set_cartesian_positions`,
`set_gripper_position`, plus per-joint/arm variants.

Read side — far richer than most wrappers expose, and the interesting half for a supervisor:
```
get_all_positions          get_all_velocities        get_all_accelerations
get_all_efforts            get_all_external_efforts   get_all_compensation_efforts
get_cartesian_positions    get_cartesian_velocities   get_cartesian_external_efforts
get_all_driver_temperatures  get_all_rotor_temperatures
get_joint_limits           get_joint_characteristics   get_error_information
get_driver_version         get_controller_version
```
`get_all_external_efforts` / `get_cartesian_external_efforts` are the contact signal — the
state where "normal contact vs collision" is a genuine judgment call.

---

## 3. LeRobot + Trossen integration

Needed only if the harness touches teleop, dataset recording, training, or policy rollout.

| What | URL | Status |
|---|---|---|
| LeRobot docs | https://huggingface.co/docs/lerobot/index | **[verified]** (from PyPI metadata) |
| LeRobot source | https://github.com/huggingface/lerobot | **[verified]** |
| LeRobot on PyPI | `lerobot` — latest **0.6.1** | **[verified]** |
| Trossen ↔ LeRobot integration source | https://github.com/TrossenRobotics/lerobot_trossen | **[verified]** |
| Trossen AI / LeRobot plugin config | https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin/configuration.html | [confirmed-url] |
| Trossen LeRobot tutorial | https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot.rst → `.html` | [confirmed-url] |
| Trossen OpenPI tutorial | https://docs.trossenrobotics.com/trossen_arm/main/tutorials/openpi.html | [confirmed-url] |
| Trossen MuJoCo tutorial | https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_arm_mujoco.html | [confirmed-url] |
| Trossen ROS 2 tutorial | https://docs.trossenrobotics.com/trossen_arm/main/tutorials/ros2.html | [confirmed-url] |

Packages in `lerobot_trossen`: `lerobot-robot-trossen`, `lerobot-teleoperator-trossen`
(plus `trossen-arm`, `trossen-slate`). LeRobot auto-discovers any installed package prefixed
`lerobot_robot_`, `lerobot_teleoperator_`, or `lerobot_camera_`.

Useful detail **[verified via search]**: WidowX AI followers observe only joint positions by
default, but can optionally record joint velocity, motor effort, and external applied effort.

Two known-relevant issues/PRs:
- https://github.com/TrossenRobotics/lerobot_trossen/pull/13 — lazy `trossen-slate` import so
  WidowX AI works on macOS (`trossen-slate` ships Linux wheels only)
- https://github.com/huggingface/lerobot/issues/2228 — WidowX AI model, depth cameras, tests

---

## 4. Cameras

The WidowX AI bay typically runs a **D405 at the wrist** and a **D455(F) on the scene** — both
Intel RealSense, both depth-capable. Wrist-mounted depth needs no hand-eye calibration to be
useful for approach, since the camera moves with the gripper.

| What | URL | Status |
|---|---|---|
| librealsense source | https://github.com/IntelRealSense/librealsense | [unverified] |
| pyrealsense2 API reference | https://intelrealsense.github.io/librealsense/python_docs/_rs.html | [unverified] |
| RealSense get-started docs | https://dev.intelrealsense.com/docs/docs-get-started | [unverified] |

Not checked this session — verify before relying on them.

---

## 5. Background reading (VLA / action chunking)

Only needed if the harness supervises or blends a learned policy's output.

| What | URL | Status |
|---|---|---|
| ACT / action chunking overview | https://www.emergentmind.com/topics/action-chunking-with-transformer-act | **[verified]** |
| VLA Knows Its Limits: Adaptive Execution Horizons | https://arxiv.org/pdf/2602.21445 | **[verified]** |
| TIDAL: Temporally Interleaved Diffusion and Action Loop | https://arxiv.org/pdf/2601.14945 | **[verified]** |

Terms worth knowing: **action chunking** (predict H future actions per observation),
**temporal ensembling** (blend overlapping chunk predictions, `w_i = exp(-m·i)`, where `m`
controls how fast new observations are incorporated), **execution/receding horizon** (execute
k of H then replan), **adaptive execution horizon** (choose k per-chunk rather than fixed).

---

## 6. What was blocked this session

These returned `EGRESS_BLOCKED` / 403 from the egress proxy and are the reason this list
exists. A session with open egress should hit them directly:

- `docs.typesafe.ai` (all pages)
- `typesafe.ai`
- `docs.trossenrobotics.com` (all pages)
- `vercel.com`

Reachable and used instead: `github.com`, `raw.githubusercontent.com`, `pypi.org`, `arxiv.org`.

---

## 7. Open questions to resolve first in the clean session

1. **`/confidence`** — how is confidence derived from the distribution? Decides whether a
   confidence gate is a safety mechanism or just a smoothness knob.
2. **`/concepts/state`** — preferred state shape. Flat JSON, nested, or prose? Affects how arm
   telemetry should be serialized.
3. **`/models`** — is there anything below `jev-latest` with lower latency, and what are the
   real rate limits?
4. **Trossen `programming_guide/concepts`** — the driver's mode model (position / velocity /
   effort / external-effort), and whether anything other than position mode is safe to drive
   from a slow outer loop.
5. **Pricing / rate limits** — a servo loop makes a call per tick; what does an hour on the
   bench cost, and what 429s will it hit?
