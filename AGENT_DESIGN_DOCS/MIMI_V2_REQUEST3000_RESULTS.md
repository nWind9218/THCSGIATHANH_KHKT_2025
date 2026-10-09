# Live evaluation paused — 3,000-request prefix

Job `191d033deed0430d83d45cbceaaa07ca`, stopped on user instruction on
09/10/2026. No automatic resume; the heartbeat automation is PAUSED.
Facebook remains legacy and unchanged. PR #3 remains draft.

## Request count and cost

The job had already exceeded 3,000 provider requests when the new stop instruction
arrived. The actual stop preserves **8,232 attempt records**, **2,285 completed
conversation turns**, and **8 interrupted turns**. Request count is not case or
user-turn count: one turn can make several model calls.

| Scope | Requests | Known-usage USD estimate | Unknown usage |
|---|---:|---:|---:|
| First finalized ledger prefix | 3,000 | 1.05413620 | 0 |
| Entire stopped batch | 8,232 | 2.88690245 | 10 |

Within the prefix: baseline 1,975 requests / $0.57139625, v2 1,025 requests /
$0.48273995. These are estimates from returned usage, not verified account
balance deductions. Earlier pilot ($0.20937555) and assessment preflight
($0.00402225) are separate; all three runs have $3.10030025 in known usage.
Unknown billing is not treated as zero.

The prefix uses preserved JSONL **finalization order**, not exact request-start
chronology: the runner did not record a timestamp or sequence at request start.
This limitation cannot be reconstructed reliably from UUIDs or file mtimes.

## Evaluation evidence

Count only completed turns whose provider attempts all fall within the prefix.
Partial boundary turns and zero-provider-call controls are excluded, not passed.

| Engine | Whole turns | Pending semantic review | Fail status | Runtime errors |
|---|---:|---:|---:|---:|
| Baseline | 418 | 392 | 24 | 2 |
| V2 | 402 | 400 | 0 | 2 |

Failed objective checks: baseline support-mode 19 and question-count 5; v2
question-count 1 on a turn already classified as runtime error. Error precedence
does not erase the failed check. Six partial boundary turns are excluded.

The included turns cover academic pressure (436) and bullying/exclusion (384).
This is neither a balanced 20-category sample nor a fully paired comparison;
different call counts cause different conversation cutoffs. Do not interpret
zero automatic failures as superior counseling quality or clinical acceptance.

In-window read-only findings (turn indices start at zero):

- `vn-v2-0008`, index 2: explicit listen-only classified as closing; canned
  goodbye despite passing question-count and support-mode checks.
- `vn-v2-0051`, index 1: assertions of safety and a close friendship unsupported
  by the student's statements.
- `vn-v2-0032`, index 2, and `vn-v2-0057`, index 4: duplicate known keys rejected
  by validation; schema validity alone does not ensure usable assessment.
- Ordinary openings repeatedly route to pending handoff; calibration needs
  review. Risk labels were not retained, so urgent versus uncertain cannot be
  distinguished from these checkpoints.

Positive examples retain temporal corrections and respond to concrete privacy
questions. These observations are not a formal semantic pass. Parent boundaries,
deep psychological questions, multi-intent coverage and all 20 categories remain
unproven in this prefix. No merge or rollout approval is inferred.

## Checkpoint and resume

Ignored local artifacts under `eval_results/full_1000`:

- `pause_snapshot/`: original ledger append order, attempts, pre-recovery cases,
  manifest and progress. No deletion or truncation of prior results.
- `paused_by_user.json`: per-case/engine next-turn resume points and pause reason.
- `first_3000_requests.report.json`, `.summary.md`: prefix accounting and evidence.
- `cost_summary.paused.json`: full stopped-job usage estimate.

The offline recovery makes no model/provider calls. Interrupted turns are kept
as explicit incomplete errors, carry their user input forward once, and are not
automatically replayed because the provider may already have charged them.
Completed turns are skipped and saved history/state is restored on resume.
The runner cannot resume inside an unfinished provider subcall; it resumes at
the persisted conversation boundary. Original interrupted state is preserved
in the snapshot for inspection.

Only after explicit user authorization:

```powershell
.\.venv\Scripts\python.exe -u -m mimi_v2.batch_eval --live --resume --output eval_results/full_1000 --concurrency 8
```

Do not modify frozen `mimi_v2` runtime source before resuming this job.
`scripts/summarize_eval_checkpoint.py` is offline-only and sits outside that
runtime hash set. Runtime hashes were verified unchanged before recovery.

### Offline checkpoint integrity audit

After recovery: 523 checkpoint files (515 complete, 8 ready), containing 2,293
turn entries = 2,285 completed + 8 explicit incomplete errors. Every file's turn
indices are contiguous, unique and agree with `next_turn`; no violations found.
Runtime and legacy source hashes still match the frozen manifest. No provider
calls were made for this audit, and the stopped process remains absent.
