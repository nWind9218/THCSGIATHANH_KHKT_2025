# Q4 design r1 receipt

This revision adds review-only Q4 annotation/contract drafts and records product approval of design structure. Detailed per-dimension anchors and contract refinements remain draft. The original user review is copied byte-for-byte to specs/q4-design-review-source-v0.1.md, with its raw hash recorded in the decision registry. External source labels/favicon links in that review are not an approved safety policy.

Verification: 146 existing offline tests pass; all 79 existing scripted replay expectations match. Executable sources, schemas, observer prompt, tests, both fixture suites and empty human-gold templates retain their previous source hashes. These are regression/integrity results, not Q4 contract validation or semantic calibration. Q4 executable tests, schema, observer extension and scorer are not implemented. Q4 acceptance, semantic accuracy and safety calibration remain NOT_EVALUATED.

Reproduce existing-suite verification from this eval worktree:

```powershell
.\.venv-local\Scripts\python.exe -m pytest -q tests/eval_v02
.\.venv-local\Scripts\python.exe -m mimi_eval.replay --offline --validate-only
```

Before/after manifests and actual pytest receipt are preserved here. No provider, credentials, DB, migration, generator or runtime changes; no phase transition. BLOCKs for full-rubric acceptance and authoritative safety policy remain.

Rollback: revert this dedicated Q4 design documentation commit on the eval branch. Restore documentation/registry/manifest together; do not reset the user's main checkout. Preserve original user approval in task history. Earlier receipts remain historical and immutable.
