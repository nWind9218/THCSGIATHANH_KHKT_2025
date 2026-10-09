# Durable full 1000-case evaluation

Run from the repository root, using the local environment containing project
dependencies. This job executes exactly the full 1000-case synthetic dataset
against baseline and v2, with isolated local stores and the same Responses API
configuration: `gpt-5-mini`, minimal reasoning, low verbosity, 2048 output tokens,
60-second timeout. Production Facebook, Redis, database and email connections
are not imported. The existing authorized key is loaded through the model-only
environment whitelist.

```powershell
.\.venv\Scripts\python.exe -u -m mimi_v2.batch_eval --live --output eval_results/full_1000 --concurrency 8 --request-cap 3000
```

An explicit `--request-cap` is required for live runs and resumes. It is a total
provider-attempt limit for this job, including retries and previously persisted
attempts, not a number of conversations or user turns. A shared durable
reservation occurs before dispatch, so concurrency cannot overrun the cap.
The example's 3000 is only an example allowance, not authorization to run.

Concurrency can be 1–16. Eight means at most eight isolated conversations run
at once; baseline and v2 execute sequentially within each case. Infrastructure
setup, acknowledge/resume/reset actions, knowledge fixtures and injected faults
from the dataset are actually executed locally. Expected faults remain errors,
never quality passes.

## Durable progress

- `manifest.json`: frozen dataset, configuration, current source and legacy hashes.
- `dataset.json`: exact source dataset used for the job.
- `cases/CASE-ENGINE.json`: atomic per-turn result, next-turn position, stored
  conversation, pending/accepted status, knowledge, alerts and local recovery buffer.
- `attempts/ATTEMPT.json`: atomic record written before each provider request;
  observed usage is saved before validating the provider response.
- `usage.jsonl`: deduplicated billing estimates keyed by unique attempt ID.
- `turn_progress.jsonl`: small progress records; no giant report rewrite per turn.
- `execution_errors/`: case construction/IO errors without destroying prior checkpoints.
- `report.json`, `cost_summary.json`, `human_review.json`: aggregate end-of-run results.

Every turn is saved before the next turn starts. Every completed turn is skipped
on resume. Store state and recovery buffers are restored so subsequent turns
can refer to earlier messages.

```powershell
.\.venv\Scripts\python.exe -u -m mimi_v2.batch_eval --live --resume --output eval_results/full_1000 --concurrency 8 --request-cap 3000
```

An OS-held lock plus PID/creation-time verification rejects a duplicate launch
or resume while the original process is live. Resume rejects changed source,
dataset or model configuration. Do not edit model, core, prompts, contracts,
baseline or runner source during the job. Documentation/test edits do not change
the runtime hash set. Completed jobs can be resumed safely and make no new requests.

If the stored attempt count already reaches or exceeds the explicit cap, resume
makes no new provider calls. A larger total cap requires a newly authorized
allowance. Budget exhaustion is `waiting_budget`, distinct from quota failures.
Future attempt records carry durable sequence and local reservation/dispatch
timestamps; they are not provider receipt timestamps. Old records are not given
invented sequence or timestamps. Frozen source checks remain enforced: this
hardening checkout cannot resume the older job by bypassing its source identity.

## Crashes and quota

If a process dies during a turn, that turn is marked an incomplete error on
resume and is not automatically replayed: the provider might already have billed
it. Its user message is carried forward once, with no invented assistant output.
Remaining turns continue from the persisted state. Atomic attempt records recover
usage even if the JSONL tail was interrupted; an unobserved response remains
unknown billing rather than zero cost.

Insufficient quota opens a shared circuit. No new provider requests start;
partly attempted turns are retained as incomplete errors and untouched turns
stay unprocessed with `waiting_quota`. The command exits 2 with state
`needs_input`. After topping up the same account, use `--resume`. Authentication
failure similarly pauses pending work; the key is not part of the public manifest.
If checkpoint persistence fails, new requests stop to avoid further unsaved spending.

Rate limits, timeouts, network failures and selected server errors have at most
two retries with bounded backoff. Each attempt has its own billing record, including
unknown usage after failures. Invalid model JSON, schema errors and ordinary
conversation failures are not retried automatically. A failed case does not
prevent independent cases from running.

Costs use observed usage and the documented standard rates. Cached input is
discounted; reasoning tokens already included in output are not billed twice.
These are estimates, not account-invoice reconciliation. Human review remains
required for qualitative judgments and rollout.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_v2_batch_eval.py tests/test_v2_pilot.py
```
