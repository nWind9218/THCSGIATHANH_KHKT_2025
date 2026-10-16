# Offline handoff — T01–T07 / P1–P3

Status: IMPLEMENTED AND VERIFIED OFFLINE. Branch `codex/mimi-persona-eval-v02`; PR base `codex/mimi-v2`, commit `48cc1f2994cc1891bd47d1d2d347f99a16d7d204`. Main checkout uncommitted runtime changes were not copied, staged, reset or exercised. The audit's dirty-runtime fingerprints remain separately identified in audit-baseline.json. This PR adds eval files only.

## Completed tasks

| Task | Deliverables | Verification |
|---|---|---|
| T01 | Decision registry with ten dispositions; rules, annotation guide; Q3 sample corrected in derivative Design_Add-ins.md | Scope amendments encoded; full-rubric and safety-owner BLOCKs preserved |
| T02 | Strict input/output schemas, contracts.py, eval-only pinned requirements | Closed/required schema, identity, scope, label/dimension coverage, nullability and negative cross-field tests |
| T03 | Observer prompt with trust boundary, multi-action spans, tri-state and no length bias | Static contract checks only; judge semantics NOT_EVALUATED |
| T04 | evidence.py | Unicode emoji/combining marks, role/source/hash/span negatives and duplicate source IDs |
| T05 | rules.py, validator.py | Four verdicts, independent FAIL precedence, no averaging, exception uncertainty, separate lanes and accepted-silence contract |
| T06 | 20 nuclei × 3 candidates + 4 controls; empty human-gold template | 64 synthetic cases, all UNREVIEWED; family splits keep T14/T20 together |
| T07 | replay.py, reporting.py, source manifest and actual reports | Network denied, replay deterministic, no cross-case state, drift stops, report overwrite refused |

## Actual checks

```powershell
.\.venv-local\Scripts\python.exe -m pytest -q tests/eval_v02 --junitxml=mimi_eval/reports/pytest-final.xml
.\.venv-local\Scripts\python.exe -m mimi_eval.replay --offline --validate-only
.\.venv-local\Scripts\python.exe -m mimi_eval.replay --offline --output mimi_eval/reports/offline-v0.2
.\.venv-local\Scripts\python.exe -m pip check
```

Results: **108 tests passed**, replay **64 cases / 0 scripted expectation mismatches**, pip check reports no broken requirements. Replay counts: **28 PASS / 21 FAIL / 12 NEEDS_REVIEW / 3 INVALID**. The INVALID set contains one simulated runtime technical failure and two invalid identity/evidence controls. Quality failures and need-compliance failures are separate in JSON. These counts are harness observations under scripted assertions, not a measured Mimi pass rate.

First test iteration found a prompt token mismatch (`Accepted-silence` vs enum `accepted_silence`), corrected without changing expected outcomes. A subsequent review tightened unconfirmed violation handling, dependent silence-state proof and internal issue-code handling. Earlier verification receipt (105 passed) is retained as an implementation iteration; pytest-final.xml (108 passed) is the handoff result. No semantic gold was changed to make tests pass.

Machine receipts: ../reports/verification.json, ../reports/pytest-final.xml, ../reports/offline-v0.2/offline-report.json and .md. Source manifest SHA256: `e8e5db107eae99591a4670a7b3ff6b2f544c911b9d9932c4cdd6eb62ae308b88`. All included source-file hashes were rechecked after the run. Source hashes canonicalize artifact file CRLF to LF UTF-8 for Git portability; source message identities/quotes remain exact code points. Generated reports, this handoff and the manifest itself are excluded from its source-file list to avoid recursive hashes.

For a deliberate new source baseline, first review changes, then use the following from repository root; do not run during/resume an existing benchmark:

```python
from pathlib import Path
import json
from mimi_eval.replay import source_hash
path = Path("mimi_eval/specs/source-manifest-v0.2.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
for relative in manifest["files"]:
    manifest["files"][relative] = source_hash(Path(relative))
# Add explicitly reviewed new paths to manifest["files"] if the scope has changed.
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

## Limits and BLOCKs

- No LLM was called, no provider was implemented, no shared database was accessed and no credentials were loaded. The simulated Terra/Responses/standard/medium-high config has no effective model or effort.
- Observer semantic accuracy, action/label/Q agreement, false PASS/FAIL, safety calibration, medium/high comparison and production readiness: **NOT_EVALUATED**. Literal citation integrity does not prove the interpretation.
- No authoritative safety/privacy exception policy and qualified owner have been supplied. `reviewed_fixture` and its synthetic reviewer only exercise proof plumbing. They are not human adjudication or safety approval.
- Full Q1/Q4/Q5/Q6 score anchors remain unavailable; full-rubric PASS is explicitly blocked. A scoped Q2/Q3 PASS is not full-rubric acceptance.
- Pending/notification-sent are not human ACK. Router comparisons are synthetic diagnostic fixtures, not live routing evaluation.
- Frozen replay has no live generator. Runtime changes, transport and store persistence are outside this verification. All synthetic fixtures remain UNREVIEWED.

## Rollback and P4 proposal

Rollback by reverting this dedicated PR's additions: mimi_eval/, tests/eval_v02/ and requirements_eval.txt. No production rollback, schema migration or data restoration is needed. Preserve historical reports; a superseding run uses a new output directory.

P4 remains NOT_STARTED. A separate proposal must approve: (1) exact Promptfoo version and bridge preserving canonical four-state output; (2) credential choice, endpoint and Terra entitlement; (3) hard request/USD ceilings with every physical retry/escalation reserved and logged; (4) Terra adapter mocks before any separately authorized paid smoke. Human gold and authoritative safety ownership are required before P5 semantic calibration. This handoff does not authorize transition, deployment or generator changes.
