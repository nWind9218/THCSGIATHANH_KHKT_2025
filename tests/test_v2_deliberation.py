"""Multi-round safety review: triage (round 1) -> deliberation (round 2) -> contradiction re-check (round 3). Offline.

Extra rounds run ONLY for genuinely ambiguous turns; a tier-1 keyword is a trigger, never a verdict.
"""
import json
from types import SimpleNamespace

import pytest

from mimi_v2 import core
from mimi_v2.core import CounselingEngine
from mimi_v2.models import ModelError
from mimi_v2.prompts import SAFETY_DELIBERATE, SAFETY_HINT_KEYWORDS, SAFETY_REVIEW_LINE
from mimi_v2.store import InMemoryStore

PRODUCTION = ("Em không có ý nghĩ làm đau mình, chỉ cứ so sánh rồi buồn. Mimi giúp em nhìn nhận bản thân "
              "bằng cái nhìn tích cực hơn được không?")
CLEAR = "t muốn tự tử"
NEGATED_CLAIM = "ko ai cản đc t tự tử đâu"


def assessment(hint="safe"):
    return dict(known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                topic="school", core_issue="school", safety_hint=hint, user_role="unknown")


def delib(risk, **changes):
    data = dict(cues=[], about_self=True, negated_or_denied=False, joking_or_hyperbole=False, quoted_or_hypothetical=False,
                intent="none", plan_or_means=False, abuse_or_danger_from_others=False, context_trend="", risk=risk, reason="fixture")
    data.update(changes)
    return data


class Model:
    """Scripted: `triage` is (risk, confidence) (or a raw dict); `rounds` are consumed per deliberate call."""
    routable = True

    def __init__(self, triage=("safe", "high"), hint="safe", rounds=(), fail_rounds=(), fail_triage=False):
        self.triage, self.hint, self.rounds = triage, hint, list(rounds)
        self.fail_rounds, self.fail_triage = set(fail_rounds), fail_triage  # 1-based deliberate call numbers
        self.calls, self.prompts, self.deliberate_calls = [], [], 0

    async def complete(self, messages, purpose="reply", effort=None, fallback=False):
        self.calls.append((purpose, effort))
        if purpose == "safety":
            if self.fail_triage:
                raise ModelError("timeout")
            payload = self.triage if isinstance(self.triage, dict) else {"risk": self.triage[0], "confidence": self.triage[1], "reason": "fx"}
            text = json.dumps(payload)
        elif purpose == "deliberate":
            self.deliberate_calls += 1
            self.prompts.append(messages[0]["content"])
            if self.deliberate_calls in self.fail_rounds:
                raise ModelError("timeout")
            item = self.rounds.pop(0)
            text = json.dumps(item if isinstance(item, dict) else delib(item))
        elif purpose == "assessment":
            text = json.dumps(assessment(self.hint))
        else:
            text = "Mimi nghe nè."
        return SimpleNamespace(text=text, usage={}, metadata={})

    @property
    def extra(self):
        return [c for c in self.calls if c[0] == "deliberate"]


class Store(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.notified = 0

    async def notify(self, user_id, text):
        self.notified += 1
        return True


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(core, "RETRY_BACKOFF_SECONDS", 0)


async def run(model, text, store=None, user="u"):
    store = store or Store()
    return await CounselingEngine(model, store).respond(user, text), store


# --------------------------------------------------------------------------- no cue -> no extra calls

@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["ừ", "sao cậu trả lời lòng vòng quá", "nay t hơi mệt", "mình buồn vì bị điểm thấp", "tớ thấy mình là gánh nặng"])
async def test_calm_or_single_tier2_messages_cost_no_extra_calls(text):
    model = Model()
    result, store = await run(model, text)
    assert sorted(c[0] for c in model.calls) == ["assessment", "reply", "safety"]
    assert model.extra == [] and result["safety_rounds"] == 1 and result["handoff_status"] == "none" and store.notified == 0


@pytest.mark.asyncio
async def test_clear_crisis_with_high_confidence_hands_off_immediately():
    model = Model(triage=("urgent", "high"))
    result, store = await run(model, CLEAR)
    assert model.extra == [] and result["safety_rounds"] == 1
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "keyword:suicide" and store.notified == 1
    assert "reply" not in [c[0] for c in model.calls]


@pytest.mark.asyncio
async def test_missing_confidence_means_high_and_adds_no_round():
    model = Model(triage={"risk": "urgent", "reason": "fx"})
    result, _ = await run(model, "xin chào")
    assert model.extra == [] and result["handoff_status"] == "pending" and result["handoff_reason"] == "model:urgent"


@pytest.mark.asyncio
async def test_low_confidence_without_any_cue_is_not_a_trigger():
    model = Model(triage=("safe", "low"))
    result, _ = await run(model, "ờ")
    assert model.extra == [] and result["safety_rounds"] == 1


# --------------------------------------------------------------------------- triggers

@pytest.mark.asyncio
@pytest.mark.parametrize("text, triage, hint", [
    ("hmm không biết nữa", ("uncertain", "high"), "safe"),            # triage uncertain
    ("tớ muốn biến mất", ("safe", "low"), "safe"),                    # low confidence + keyword
    ("ờ chắc vậy", ("uncertain", "low"), "safe"),                     # low confidence + triage != safe
    ("xin chào", ("safe", "high"), "uncertain"),                      # triage and assessment hint disagree
    ("xin chào", ("uncertain", "high"), "urgent"),
    ("tớ muốn tự tử", ("safe", "high"), "safe"),                      # tier-1 keyword the triage called safe
    ("tớ muốn tự tử", ("uncertain", "high"), "safe"),
])
async def test_ambiguous_turns_get_exactly_one_deliberation_when_it_clears(text, triage, hint):
    # Round 2 says "safe" and no tier-1 keyword is present (or round 3 agrees): count the extra calls.
    model = Model(triage=triage, hint=hint, rounds=["safe", "safe"])
    result, _ = await run(model, text)
    expected = 2 if "tự tử" in text else 1
    assert len(model.extra) == expected and result["safety_rounds"] == 1 + expected


@pytest.mark.asyncio
async def test_accumulated_tier2_signals_trigger_deliberation():
    model = Model(rounds=["urgent"])
    engine = CounselingEngine(model, Store())
    results = [await engine.respond("u", t) for t in ("tớ muốn biến mất", "tớ thấy mình là gánh nặng", "tạm biệt mọi người")]
    assert [r["safety_rounds"] for r in results] == [1, 1, 2]
    assert results[2]["handoff_status"] == "pending" and results[2]["handoff_reason"] == "accumulation"


# --------------------------------------------------------------------------- the production incident

@pytest.mark.asyncio
async def test_production_sentence_no_longer_hands_off_when_the_review_says_safe():
    review = delib("safe", cues=["làm đau mình"], negated_or_denied=True, context_trend="so sánh bản thân, buồn")
    model = Model(rounds=[review, review])
    result, store = await run(model, PRODUCTION)
    assert result["handoff_status"] == "none" and store.notified == 0 and result["safety_mode"] == "normal"
    assert result["safety_rounds"] == 3 and [c for c in model.extra] == [("deliberate", "medium"), ("deliberate", "high")]
    assert "deliberation:safe" in result["diagnostics"] and "deliberation_round3:safe" in result["diagnostics"]
    # The review sees the keyword hint and the triage verdict, and is told keywords may be negated.
    first = model.prompts[0]
    assert SAFETY_HINT_KEYWORDS.format(codes="self_harm_cut") in first and "risk=safe" in first and "phủ định" in SAFETY_DELIBERATE
    assert "Rà soát lần hai" in model.prompts[1] and SAFETY_REVIEW_LINE.split("{")[0] in model.prompts[1]
    assert "reply" in [c[0] for c in model.calls]


@pytest.mark.asyncio
async def test_keyword_with_a_real_urgent_review_hands_off_after_one_extra_round():
    model = Model(rounds=[delib("urgent", cues=["rạch tay"], intent="active", plan_or_means=True)])
    result, store = await run(model, "t muốn rạch tay")
    assert len(model.extra) == 1 and result["safety_rounds"] == 2
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "keyword:self_harm_cut" and store.notified == 1


# --------------------------------------------------------------------------- round 3

@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["negated_or_denied", "joking_or_hyperbole", "quoted_or_hypothetical"])
async def test_urgent_review_that_is_negated_joking_or_quoted_is_rechecked_once_at_high_effort(flag):
    model = Model(rounds=[delib("urgent", **{flag: True}), delib("safe")])
    result, store = await run(model, NEGATED_CLAIM)
    assert model.extra == [("deliberate", "medium"), ("deliberate", "high")] and result["safety_rounds"] == 3
    assert result["handoff_status"] == "none" and store.notified == 0  # round 3's verdict is final
    model = Model(rounds=[delib("urgent", **{flag: True}), delib("urgent")])
    result, store = await run(model, NEGATED_CLAIM)
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "keyword:suicide" and store.notified == 1


@pytest.mark.asyncio
async def test_plain_urgent_review_and_safe_review_without_keyword_skip_round3():
    model = Model(rounds=[delib("urgent")])
    result, _ = await run(model, NEGATED_CLAIM)
    assert len(model.extra) == 1 and result["safety_rounds"] == 2
    model = Model(hint="uncertain", rounds=[delib("safe")])  # disagreement trigger, no keyword
    result, _ = await run(model, "xin chào")
    assert len(model.extra) == 1 and result["safety_rounds"] == 2 and result["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_total_extra_calls_never_exceed_two():
    model = Model(rounds=[delib("urgent", negated_or_denied=True), delib("urgent", negated_or_denied=True), delib("safe")])
    result, _ = await run(model, NEGATED_CLAIM)
    assert len(model.extra) == 2 and result["safety_rounds"] == 3 and result["handoff_status"] == "pending"


# --------------------------------------------------------------------------- failures

@pytest.mark.asyncio
async def test_failed_deliberation_with_tier1_fails_closed():
    model = Model(fail_rounds={1})
    result, store = await run(model, PRODUCTION)
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "keyword:self_harm_cut" and store.notified == 1
    assert "deliberation_failed" in result["diagnostics"] and result["safety_rounds"] == 2


@pytest.mark.asyncio
async def test_invalid_deliberation_counts_as_failed():
    bad = delib("safe")
    del bad["intent"]
    result, _ = await run(Model(rounds=[bad]), "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and "deliberation_failed" in result["diagnostics"]
    result, _ = await run(Model(rounds=[{**delib("safe"), "extra": 1}]), "tớ muốn tự tử")
    assert result["handoff_status"] == "pending"
    result, _ = await run(Model(rounds=[delib("maybe")]), "tớ muốn tự tử")
    assert result["handoff_status"] == "pending"


@pytest.mark.asyncio
async def test_failed_deliberation_without_keyword_falls_back_to_triage_and_hint():
    model = Model(triage=("uncertain", "high"), fail_rounds={1})
    result, store = await run(model, "xin chào")
    assert result["safety_mode"] == "checkin" and store.notified == 0 and "deliberation_failed" in result["diagnostics"]
    model = Model(triage=("safe", "low"), hint="urgent", fail_rounds={1})
    result, _ = await run(model, "xin chào")
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "model:urgent"


@pytest.mark.asyncio
async def test_failed_round3_keeps_an_urgent_review_but_softens_a_waved_off_keyword():
    result, _ = await run(Model(rounds=[delib("urgent", negated_or_denied=True)], fail_rounds={2}), NEGATED_CLAIM)
    assert result["handoff_status"] == "pending" and "deliberation_round3_failed" in result["diagnostics"]
    result, store = await run(Model(rounds=[delib("safe")], fail_rounds={2}), PRODUCTION)
    assert result["handoff_status"] == "none" and result["safety_mode"] == "checkin" and store.notified == 0


@pytest.mark.asyncio
async def test_triage_failure_with_keyword_still_gets_a_deliberation():
    model = Model(fail_triage=True, rounds=["urgent"])
    result, _ = await run(model, "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and "safety_unavailable_or_invalid" in result["diagnostics"]
    model = Model(fail_triage=True, rounds=["safe", "safe"])
    result, _ = await run(model, "tớ muốn tự tử")
    assert result["handoff_status"] == "none" and result["safety_mode"] == "normal"  # the review answered


# --------------------------------------------------------------------------- guard and outputs

@pytest.mark.asyncio
async def test_inconsistent_review_cannot_wave_off_an_active_planned_self_harm():
    review = delib("uncertain", intent="active", plan_or_means=True)
    result, _ = await run(Model(triage=("uncertain", "high"), rounds=[review]), "xin chào")
    assert result["handoff_status"] == "pending" and "deliberation_raised_inconsistent" in result["diagnostics"]
    # But a denial/joke is respected.
    review = delib("safe", intent="active", plan_or_means=True, joking_or_hyperbole=True)
    result, _ = await run(Model(triage=("uncertain", "high"), rounds=[review]), "xin chào")
    assert result["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_diagnostics_are_codes_only_and_rounds_are_reported():
    secret = "ZQXSECRET"
    model = Model(rounds=[delib("safe", cues=[secret]), delib("safe", cues=[secret])])
    result, _ = await run(model, f"tớ muốn tự tử {secret}")
    assert secret not in json.dumps(result["diagnostics"], ensure_ascii=False) and result["safety_rounds"] == 3


@pytest.mark.asyncio
async def test_takeover_turn_reports_zero_rounds():
    store = Store()
    await store.save("u", {"messages": [], "handoff_status": "accepted"})
    result, _ = await run(Model(), "xin chào", store=store)
    assert result["safety_rounds"] == 0


def test_prompt_contracts_for_the_new_rounds():
    from mimi_v2.contracts import schema_for_purpose
    from mimi_v2.prompts import SAFETY
    assert '"confidence"' in SAFETY and "low" in SAFETY and len(SAFETY) < 2200
    for field in ("cues", "about_self", "negated_or_denied", "joking_or_hyperbole", "quoted_or_hypothetical", "intent",
                  "plan_or_means", "abuse_or_danger_from_others", "context_trend", "risk", "reason"):
        assert f'"{field}"' in SAFETY_DELIBERATE
    schema = schema_for_purpose("deliberate")["schema"]
    assert set(schema["properties"]) == set(schema["required"]) == set(core.Deliberation.model_fields)
    assert set(schema_for_purpose("safety")["schema"]["required"]) == {"risk", "confidence", "reason"}
