# Annotation guideline — Mimi v0.2

Status: behavior rules approved by product owner; examples/fixtures UNREVIEWED. No adjudicated human gold yet.

Read the current student message and relevant prefix first, then candidate, approved policy version, retrieved facts/memory and relevant events. Never read future turns for an earlier target. Identify current communicative purpose and each boundary separately. Prefer a recent direct correction in the same dimension. Older memory does not override it.

Record all observed actions with literal candidate quotes. A concrete action in a question is advise, not automatically clarify. An offer without a concrete method is offer_support. Multiple actions may overlap exactly. Grounded reflection does not hide advice appended to it. Generic encouragement/explanation need not fit one of the five action labels.

Use false only for observed absence/non-activation with sufficient prefix. Use unknown for missing history/ambiguity/conflicting boundaries. Not finding a request in a truncated window does not establish it never occurred. True label evidence must show the live boundary/request; cite old and revised messages for transitions. “Chỉ nghe” and “đừng hỏi” are independent. advice_requested=false does not imply unsolicited_advice=true.

## Q2: support-mode choice

| Score | Anchor |
|---|---|
| 3 | Best fit to need, boundaries and context; no unnecessary action. |
| 2 | Primary need fits; minor awkward offer/question without changing direction. |
| 1 | Materially premature/repeated advice/offers or unnecessary permission loop after a direct request. |
| 0 | Contradicts a clear effective boundary or entirely ignores a direct need for help. |

Reflect-first does not require a stock reflection in every response. Direct help when requested is appropriate. When need is unclear, several short responses may be correct. A small timely suggestion can be appropriate with enough context even without explicit advice_requested=true. Respect an effective listen-only request except necessary, proportionate, policy-supported safety intervention. Do not infer a safety exception merely from the word “safety”.

## Q3: tone

| Score | Anchor |
|---|---|
| 3 | Natural, respectful, calm, warm enough and proportionate. |
| 2 | Generally appropriate, somewhat formulaic without major loss of feeling heard. |
| 1 | Clearly pressuring, repetitive, empty reassurance or mechanical phrasing. |
| 0 | Dismissive, invalidating or severely coercive. |

A short response can score 3. No required length, list of solutions, question, “Mimi hiểu”, or quota of supportive actions. Repeated offers after refusal can lower Q2 and Q3 even without concrete advice. Score Q2/Q3 independently; warmth cannot rescue a boundary violation.

For Q1 facts, Q4 helpful/safe guidance, Q5 use of KB/memory and Q6 continuity, preserve the baseline subjects. Detailed v0.1 anchors are not supplied here: BLOCK full-rubric acceptance rather than invent anchors. Q4 can be N/A without guidance; Q5 can be N/A without relevant sources, with a clear reason. No average can compensate for an applicable required score <2.

## Human gold procedure

Use human-gold-template.v0.2.json. Annotators give evidence, action sets/ranges, tri-state labels, applicability, score ranges and brief reasons, independently/blind when possible. Keep reviewer identities, expertise for safety, policy version and adjudication reason. A synthetic reviewer string is never human approval. Keep candidates/paraphrases in their nucleus/family split; T14/T20 are the same unsafe-parent-source family. Do not use sample_responses as gold or self-assessment as truth. Human gold stays UNREVIEWED until people actually adjudicate. No semantic metrics before that.
