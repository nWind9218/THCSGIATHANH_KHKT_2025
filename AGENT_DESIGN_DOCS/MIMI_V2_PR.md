# Mimi v2 — opt-in Messenger core, model adapter and local evaluation

## Scope

- Mature Vietnamese persona, structured context assessment, corrections and listen-only mode.
- Canonical history with per-session locks; legacy history/topic compatibility.
- Safety before retrieval; pending notification distinct from accepted human takeover.
- Responses and Chat Completions adapter with explicit model/reasoning/verbosity/token/timeout configuration.
- Shared isolated core for Gradio and multi-turn conversation eval; real legacy sandbox baseline.
- 1,000 synthetic conversation cases, 20 categories, 4,400 user turns, scenario-level 600/200/200 splits.

## Production boundary

Messenger remains legacy by default. V2 requires `MIMI_ENGINE=v2`; Facebook URLs,
Page credentials and Send API are unchanged. No production rollout, WebSocket
changes, database migration or memory reset. Local tools use RAM and cannot send
Facebook messages/email or connect to production Redis/Postgres.

## Evidence and acceptance

119 offline unit/regression tests passed including the live pilot metering tests. Both model
API paths passed synthetic credential probes. Gradio rendered conversation,
pending/accepted/resume and fault behavior was inspected in the browser.
Full 1,000-case offline comparison exercised 8,800 turns; fixture scores are not
model-quality evidence. Live balanced 20-category pilot is authorized and running
as follow-up; final usage/cost and outcome report will be added before review.

## Remaining gates

Draft only: behavioral review, cost decision for full live evaluation, and explicit
rollout approval are still required. No claim that Responses is faster solely
from adapter migration. Holdout is synthetic and not an independent clinical
benchmark. Source/usage provenance and named review criteria are retained.

## Local commands

See `AGENT_DESIGN_DOCS/MIMI_V2_LOCAL_TESTING.md` and `eval_datasets/README.md`.
Optional Gradio dependencies belong in a dedicated `.venv-local`, not the deployed
runtime. Eval outputs are ignored to prevent accidental conversation disclosure.
The pilot is limited to exactly 20 development cases × baseline/v2; returned API
usage is recorded before response validation, including incomplete/refused output.
Unknown usage is not silently billed as zero. USD estimates use published rates,
not an independently verified account balance or invoice.
