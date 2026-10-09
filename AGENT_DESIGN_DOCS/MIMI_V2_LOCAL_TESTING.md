# Mimi v2: local simulation and conversation eval

Run from the repository root. The optional UI dependencies are separate from
production requirements; install them in a dedicated local environment:

```powershell
py -3 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r requirements.txt -r requirements_local.txt
.\.venv-local\Scripts\python.exe -m mimi_v2.simulator
```

The UI listens on `http://127.0.0.1:7860`, with `share=False`. A fresh browser
session receives a generated local PSID. History, knowledge and alerts use RAM;
closing the process loses test state. No Facebook, Redis, PostgreSQL or email
connector is imported by the simulator/eval. These tools cannot test platform
webhook signatures or Facebook delivery itself.

The default deterministic offline model tests wiring, faults and history. It is
not a quality benchmark. Use the fault selector for model timeout, storage read
failure or local notification failure. A handoff is pending until the simulated
human clicks “Người hỗ trợ tiếp nhận”; accepted handoffs suppress chatbot replies.
“Cho chatbot tiếp tục” resumes and “Xóa phiên test” resets the test session.
Diagnostics contain assessment, sources, timings and errors, not hidden reasoning.

## Explicit live model mode

The existing key was authorized for this project. The local loader reads only
`OPENAI_API_KEY` and `OPENAI_CHAT_MODEL` from environment/`.env`; it does not load
Facebook, database or email variables. Live requests go to OpenAI and may cost
money; use synthetic conversations rather than production student transcripts.

```powershell
.\.venv-local\Scripts\python.exe -m mimi_v2.simulator --live --api responses --model gpt-5-mini
.\.venv-local\Scripts\python.exe -m mimi_v2.simulator --live --api chat_completions --model gpt-5-mini
```

## Conversation dataset

JSON schema version 1: `cases` is a nonempty list. Each case has a unique safe
`id`, optional `role` (`student`, `parent`, `unknown`), and a nonempty `turns`
list. Role is metadata, not an injected identity claim; put self-identification
in the actual user message when testing parental conversation behavior.

Each turn has `user`, optional `expect` and `review` (human-review criteria).
Supported automated expectations: `contains`, `not_contains` (string lists),
`intent`, `role`, `support_mode`, `handoff` (strings), `max_questions` and
`max_chars` (nonnegative integers). Unsupported fields and duplicate IDs fail
validation rather than silently being ignored. See `eval_datasets/v2_smoke.json`.

```powershell
.\.venv-local\Scripts\python.exe -m mimi_v2.evaluation eval_datasets/v2_smoke.json --compare --output eval_results
.\.venv-local\Scripts\python.exe -m mimi_v2.evaluation eval_datasets/v2_smoke.json --live --compare --api responses --model gpt-5-mini --output eval_results
```

`--compare` runs the actual legacy node/edge/state code with isolated dependency
injection alongside v2. It preserves legacy behavior (including history defects).
It does not run Facebook delivery, durable database writes or OCEAN updates.
Reports include legacy source provenance and comparison limitations.

For multiple API/model candidates, create a JSON file containing a list:

```json
[
  {"engine":"baseline","live":true,"api":"responses","model":"gpt-5-mini"},
  {"engine":"v2","live":true,"api":"responses","model":"gpt-5-mini"},
  {"engine":"v2","live":true,"api":"chat_completions","model":"gpt-5-mini"}
]
```

```powershell
.\.venv-local\Scripts\python.exe -m mimi_v2.evaluation path-to-dataset.json --configs path-to-configs.json --live --output eval_results
```

Each case/config uses fresh storage. The JSON report captures each input/output,
API/model calls, usage when available, latency, checks, configuration/source and
dataset hashes. Cost is unavailable unless the provider supplies it; no price is
invented. Markdown summarizes turn status. Runtime errors and skipped cases are
never counted as pass. Turns without automatic criteria remain pending review.

## Human review and acceptance

The runner writes `JOB.review.json`. Fill `verdict` with `pass` or `fail`,
`reviewer` and `comment`, preserving case/config/turn identifiers:

```powershell
.\.venv-local\Scripts\python.exe -m mimi_v2.evaluation --review-job eval_results/JOB.json --ratings eval_results/JOB.review.json --output reviewed_results
```

Human pass cannot erase an automated failure or runtime error. The included
smoke dataset demonstrates tooling only. Quality acceptance is still pending
the user's dataset and named human review; these reports do not authorize rollout
or select a winning model automatically.

## Verification

```powershell
.\.venv-local\Scripts\python.exe -m pytest -q tests/test_v2_local_eval.py
```

Verified with Gradio 5.49.1: rendered offline conversation, pending → accepted
handoff, resume and model-timeout diagnostic in the browser. The optional install
may resolve Pydantic/Starlette/Pandas versions differently from the production
environment; the dedicated environment above avoids altering a deployed runtime.

## Messenger configuration and rollback (not activated by this PR)

The default `MIMI_ENGINE=legacy` leaves current Facebook routing unchanged.
Only after quality review and explicit rollout approval, the operator can choose:

```text
MIMI_ENGINE=v2
MIMI_V2_API=responses
MIMI_V2_MODEL=gpt-5-mini
MIMI_V2_REASONING_EFFORT=minimal
MIMI_V2_VERBOSITY=low
MIMI_V2_MAX_OUTPUT_TOKENS=2048
MIMI_V2_TIMEOUT=60
```

Use `chat_completions` for the alternative API. Omitted reasoning/verbosity use
the provider defaults; the sample above is a configurable candidate, not a model
quality winner. The production adapter rejects timeout above 60 seconds to keep
the Redis lock lease bounded. Config changes require process restart because the
production engine is cached. Keep the existing authorized OpenAI key; do not copy
keys into versioned configs or artifacts.

Only configured `ADMIN_FACEBOOK_PSIDS` can run `#accept [PSID]` (pending only) or
`#resume [PSID]`. Parent role inferred from text is not an admin identity.
Rollback chooses `MIMI_ENGINE=legacy` and restarts the process without deleting
history, topic, takeover or v2 metadata. Existing accepted takeover remains
silent in legacy through its original flag. Pending v2 review is not an accepted
takeover and must still be followed up by the human operator during rollback.
No actual production environment variables have been changed by these docs.
