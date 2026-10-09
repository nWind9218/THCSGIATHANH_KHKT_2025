# Offline hardening without invalidating the stopped benchmark

This checkout is `codex/mimi-v2-offline-hardening`, based on `ce6f7a1`.
It is separate from the original `codex/mimi-v2` checkout and its paused job
`191d033deed0430d83d45cbceaaa07ca`. Do not alter the original runtime, checkpoint,
ledger, automation or API credentials while preparing these changes.

## Scope

- Reproduce listen-only/closing routing conflict with offline model fixtures.
  Preserve genuine stopping, urgent/uncertain handoff and accepted silence.
  Improve prompt instructions without pretending keyword checks prove meaning.
- Collapse only identical duplicate known facts; conflicting duplicates must
  remain validation errors. Strengthen user-fact grounding and corrections.
- Add a future-run provider-attempt budget reserved durably before dispatch,
  including concurrent requests and retries. Require an explicit CLI cap.
  Record request sequence/time for future runs; do not invent old ordering.
- Keep semantic quality unproven until separately authorized live re-evaluation.
  No changes to Facebook delivery, default legacy mode or safety thresholds.

## Important limits

The stopped benchmark has per-turn checkpoints, not model-subcall checkpoints.
Its eight interrupted turns cannot be reconstructed from usage-only records;
they stay explicit incomplete errors, with no automatic paid replay. The new
budget must not claim subcall-level resumption or bypass frozen identity checks.

Offline tests establish routing, normalization, caps and checkpoint invariants,
not counseling intelligence, empathy or clinical safety. Existing semantic
findings and the original results remain evidence, not ground-truth answers.

## Integration gate

Changes are prepared locally in this attached worktree only. Do not cherry-pick
into the original checkout, push over PR #3, run new paid evals or rollout without
resolving the benchmark/version decision with the user. The original job can
still be resumed with its unchanged runtime when explicitly authorized.

## Verification receipts

- Final combined isolated unit/regression suite: **175 passed in 34.14s**. No live
  model, production storage, Facebook or email calls were used.
- Ten behavior regressions reproduced four failures before the core fix, then
  passed. They prove routing/payload and lossless normalization, not model meaning.
- Fourteen batch tests cover total caps, retries, historical accounting, zero
  additional calls on exhausted-cap resume, source-identity rejection and
  timestamp semantics. The yielding-provider test confirms overlapping dispatch
  across cases stops at exactly three reserved requests, with no new calls when
  resumed at the same exhausted cap. This is not a production load benchmark.
- Read-only independent review found no blocking code defect. True closing with
  listen now needs an extra model call; stopping and factual grounding remain
  model-dependent and require authorized semantic re-evaluation.
- The original runtime hash still matches its paused manifest, and original
  request count remains 8,232. No automatic resume or rollout was enabled.

## Portability correction

The first combined run had 172 passing tests and one dataset-provenance failure.
The case contents were identical; Git checkout converted catalog source from LF
to CRLF, changing byte hashes. Catalog provenance now hashes canonical LF bytes,
with regression coverage for LF/CRLF equivalence and real content changes.
No materialized dataset or original-checkout file was regenerated or changed.
