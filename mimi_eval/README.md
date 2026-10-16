# Mimi Persona Eval v0.2 — offline P1–P3

This package evaluates **the harness** using frozen synthetic inputs and scripted observer assertions. It does not run Mimi or a judge. There is no provider/DB/email/production import, Terra adapter, Promptfoo integration or live switch. Those require separate P4 approval. No dotenv or production credentials are loaded.

Approved scope: T01–T07, product-owner Scope approval — Mimi Persona Eval v0.2. Contract decisions D0–D6 are implemented; all ten audit corrections have explicit approved/deferred disposition in specs/decision-register-v0.2.json. specs/Design_Add-ins.md preserves the addendum as a derivative and corrects its Q3 length-biased example. The original user attachment is unchanged.

## Run

From repository root, in an isolated local environment (Python 3.13 was used for verification):

```powershell
python -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r requirements_eval.txt
.\.venv-local\Scripts\python.exe -m pytest -q tests/eval_v02
.\.venv-local\Scripts\python.exe -m mimi_eval.replay --offline --validate-only
.\.venv-local\Scripts\python.exe -m mimi_eval.replay --offline --output eval_results/mimi-v02-run-001
```

Dependency installation fetches public packages if absent. Tests and replay deny network, including loopback; tests also reject production imports and credential-file reads. Use a fresh output directory per run: replay refuses report overwrite. Do not run legacy eval tools or instantiate production stores.

Source-manifest-v0.2.json pins package, fixtures, contracts, prompt, tests and requirements SHA256 after LF UTF-8 artifact normalization. Decoded transcript text and quote/source hashes are never normalized. Drift stops replay. To intentionally establish a new baseline after edits, review the changes and regenerate that manifest explicitly; never update a running benchmark baseline. Verification commands/actual results and rollback are in specs/offline-handoff.md; checked-in reports are immutable receipts for this version.

## Contracts and interpretation

contracts/ contains strict JSON Schema; contracts.py enforces identity and cross-field constraints; evidence.py verifies source/role/hash/code-point span; validator.py owns four verdicts. rules.py has only owner-approved behavior/honesty gates and an explicitly fixture-only proof simulation. An observer cannot output final verdict, approve an exception, confirm ACK or declare effective model/effort. One observed span can support reflect+advise. False differs from unknown; listen-only differs from no-questions.

79 frozen candidates: 25 nuclei × 3 candidates, plus 4 accepted-silence/technical controls. Deliberately invalid inputs/quotes exercise INVALID. All human adjudication is UNREVIEWED. Scripted score assertions test gate arithmetic, not their own semantic accuracy. The empty human-gold template must be completed by humans later; no fabricated gold.

Reports keep response_quality, router_policy and system_actions separate. scoped_q2_q3 PASS explicitly has full_rubric_pass=false. Accepted-silence is a separate system-action contract. Technical errors, quality failures, need-compliance failures, REVIEW and INVALID are separate fields. Every semantic metric without gold is NOT_EVALUATED. Offline PASS does not establish production safety.

## Student-purpose refinement (r1)

See specs/student-perspective-review-v0.2.md for the reviewed design intent and concrete Vietnamese contrasts. T21–T24 hold the final “Trời ơi, em phải làm sao đây...” utterance and reusable replies constant while changing the preceding purpose; T23/T24 also hold no-questions constant. T25 tests genuine ambiguity despite a complete prefix. These are scripted hypotheses, all UNREVIEWED, not observed LLM classification results.

Q3 assesses communication quality with no slang quota. It shares tone concerns with persona, but does not establish complete Mimi persona acceptance; the separate persona set provides illustrative cases but no scorer or acceptance gate. Persona acceptance is explicitly NOT_EVALUATED in the report. Reports under reports/offline-v0.2/ and pytest-final.xml are historical baseline receipts; interaction revision receipts are under reports/student-lens-r1/.

## Approved persona rubric; unreviewed fixtures (r1)

See specs/mimi-persona-rubric-v0.2.md and its JSON source: six product-owner-approved facets with 0–3 anchors, scoped evidence, and four context-sensitive tone directions confirmed by the product owner (subtle playful, mischievous playful, gentle warm, energetic encouraging). No slang/emoji quota; no single default temperament is imposed across contexts.

fixtures/mimi-persona-suite-draft-v0.2.json has 17 scenarios × 3 candidates = 51 illustrative candidates, with an independent fixture schema and empty human-gold-persona-template.v0.2.json. All detailed scores remain PROPOSED/UNREVIEWED. Existing Q2/Q3 schemas, validator and 79-case replay are unchanged; persona draft cannot be passed into that replay as if it had been graded. No new model adapter, observer output contract, persona scorer or production prompt change.

Validation receipts under reports/persona-draft-r1/ cover draft structure, source/span integrity and the existing offline suite. They do not measure persona acceptance or judge semantic accuracy. For independent human annotation, hide draft_scores/style_variant/appropriate_modes from the annotation packet rather than treating the illustrations as gold.

## Known limits and next phase

- Deterministic evidence integrity does not prove semantic correctness; wrong but literal interpretations remain possible until human calibration.
- Authoritative safety/privacy policy version and qualified owner are not supplied. Safety exception proof is simulated only. Unapproved policy claims review.
- Detailed Q1/Q4/Q5/Q6 baseline score anchors are unavailable: full-rubric PASS is blocked rather than invented.
- Router comparisons are fixture expectations, not production router calibration. Frozen replay does not exercise the dirty runtime changes audited in the main checkout.
- Model contract is simulated configuration only: gpt-5.6-terra / Responses / standard / medium, high for approved comparison. No effective model/effort is claimed; provider_calls=0.
- P4 proposal: separately approve exact Promptfoo pin/bridge preserving four verdicts; credential/endpoint/entitlement; physical-attempt ledger with shared request/USD ceilings; Terra adapter mocks, followed by separately authorized live smoke. Do not transition phases automatically.

Rollback: revert the dedicated eval PR only (mimi_eval/, tests/eval_v02/, requirements_eval.txt). No runtime, generator prompt, shared schema or database migration is part of this PR.

Product approval recorded on 2026-10-17 in the rubric and decision registry. Approval covers rubric design, not illustrative scores, human gold or persona acceptance. Approval revision receipts: reports/persona-approved-r1/. Current rubric inventory: specs/rubric-inventory-v0.2.md. Historical draft receipts remain unchanged.
