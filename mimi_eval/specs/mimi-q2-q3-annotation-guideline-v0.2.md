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

## Student perspective: communicative purpose before wording

The role-play in this conversation is synthetic design input, not research with actual students and not Mimi's canonical persona. Judge current purpose from the whole prefix, never from “em phải làm sao?” or a question mark alone.

- Venting: a rhetorical “Trời ơi, em phải làm sao đây...” can continue an existing wish to vent. If a clear listen-only boundary has not clearly been changed, preserve it. Do not treat the phrase itself as permission for advice.
- Solution request: a concrete goal/time constraint, request for a next step or clear change of need can make the same phrase a genuine request. Give a manageable next step without an unnecessary permission loop. Emotion and exasperation do not make every request rhetorical.
- Unclear: complete history can still be semantically ambiguous. Use advice_requested=unknown and a material Q2 review flag when the distinction would change judgment. This is valid NEEDS_REVIEW, not INVALID. Missing history and genuine ambiguity have different reasons; high effort cannot supply absent evidence.
- No-questions is independent in all three states. It rules out another request for a reply, including an advise-shaped “Em có muốn thử X không?”. It does not by itself request advice. A gentle question is optional when permitted; do not reward one merely for keeping conversation going.

Student-facing checks for Q2: “Did Mimi catch what I need now?”, “Do I know where to start if I asked for help?”, “Did it make me explain myself or repeat a boundary unnecessarily?” These guide interpretation; do not claim every student shares the role-play preferences.

For an unmistakable solution request, a whole response that only reflects emotion and offers no help scores Q2=0 under the existing anchor. Q2=1 can cover support directed toward the goal but materially delayed/misdirected, such as another unnecessary permission loop. A short acknowledgment before substantive help is not the same as a reflection-only whole turn. Do not assign 0 to a rhetorical lament simply because its words resemble a request.

## Q3: tone

| Score | Anchor |
|---|---|
| 3 | Natural, respectful, calm, warm enough and proportionate. |
| 2 | Generally appropriate, somewhat formulaic without major loss of feeling heard. |
| 1 | Clearly pressuring, repetitive, empty reassurance or mechanical phrasing. |
| 0 | Dismissive, invalidating or severely coercive. |

A short response can score 3. No required length, list of solutions, question, “Mimi hiểu”, or quota of supportive actions. Repeated offers after refusal can lower Q2 and Q3 even without concrete advice. Score Q2/Q3 independently; warmth cannot rescue a boundary violation.

For Q1 facts, Q4 helpful/safe guidance, Q5 use of KB/memory and Q6 continuity, preserve the baseline subjects. Detailed v0.1 anchors are not supplied here: BLOCK full-rubric acceptance rather than invent anchors. Q4 can be N/A without guidance; Q5 can be N/A without relevant sources, with a clear reason. No average can compensate for an applicable required score <2.

## Q3 and persona are related, with separate acceptance claims

Student-facing Q3 check: “Does this feel respectful and easy to keep talking to?” Grade naturalness, age/context fit, pressure, repetition and condescension. Reading effort matters when verbosity creates a concrete burden; length alone is not a fault. A simple neutral sentence may score 3, and slang/emojis do not earn bonus points. Forced teen slang or overfamiliarity can hurt Q3 when there is observable evidence, not because a keyword is absent/present.

Q2 asks whether Mimi chose the right support action now; Q3 asks how that action was communicated. Calm but unwanted advice may have Q2=0 with acceptable tone. Useful advice with a belittling remark can have good action selection and Q3=0. Cite distinct reasons; do not transfer the Q2 verdict automatically to Q3. The same span may support both deductions if there is actual pressure or repeated disregard.

Brand voice, Mimi identity, characteristic wording, relationship boundaries and consistency across topics need a dedicated persona case set with reviewed expectations. They must not be inferred from the vent/request contrastive cases. Q3 remains the communication-quality dimension; it is not a complete persona acceptance test. Do not require Mimi to imitate this synthetic student's “tui/tớ” register. Persona acceptance for this suite remains NOT_EVALUATED; a separate persona case set is not implemented in this refinement.

## Human gold procedure

Use human-gold-template.v0.2.json. Annotators give evidence, action sets/ranges, tri-state labels, applicability, score ranges and brief reasons, independently/blind when possible. Keep reviewer identities, expertise for safety, policy version and adjudication reason. A synthetic reviewer string is never human approval. Keep candidates/paraphrases in their nucleus/family split; T14/T20 are the same unsafe-parent-source family. Do not use sample_responses as gold or self-assessment as truth. Human gold stays UNREVIEWED until people actually adjudicate. No semantic metrics before that.
