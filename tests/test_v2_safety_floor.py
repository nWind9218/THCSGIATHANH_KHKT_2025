"""Keyword / accumulated-signal memory around the LLM safety gate: cues trigger a deliberation, never decide. Offline."""
import json
import logging
from types import SimpleNamespace

import pytest

from mimi_v2.core import CounselingEngine, canonical_session
from mimi_v2.prompts import SAFETY, SAFETY_HINT_ACCUMULATED, SAFETY_HINT_KEYWORDS
from mimi_v2.store import InMemoryStore

SUICIDE = "tớ muốn tự tử"
DISAPPEAR = "tớ muốn biến mất"
BURDEN = "tớ thấy mình là gánh nặng"
GOODBYE = "tạm biệt mọi người"
NEUTRAL = "ok cảm ơn cậu nha"


def assessment():
    return dict(known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                topic="school", core_issue="school", safety_hint="safe", user_role="unknown")


class Model:
    """Always-safe triage unless told otherwise; the deliberation says `deliberate` (default urgent: a cue that
    reaches round 2 is treated as real). Records every safety prompt it receives."""

    def __init__(self, risks=("safe",), routable=False, deliberate="urgent"):
        self.risks, self.routable, self.deliberate = list(risks), routable, deliberate
        self.safety_prompts, self.calls = [], []

    async def complete(self, messages, purpose="reply", effort=None, fallback=False):
        self.calls.append((purpose, effort))
        if purpose == "safety":
            self.safety_prompts.append(messages[0]["content"])
            risk = self.risks.pop(0) if len(self.risks) > 1 else self.risks[0]
            text = json.dumps({"risk": risk, "reason": "fixture"})
        elif purpose == "deliberate":
            text = json.dumps(dict(cues=[], about_self=True, negated_or_denied=False, joking_or_hyperbole=False,
                                   quoted_or_hypothetical=False, intent="none", plan_or_means=False,
                                   abuse_or_danger_from_others=False, context_trend="", reason="fixture",
                                   risk=self.deliberate))
        elif purpose == "assessment":
            text = json.dumps(assessment())
        else:
            text = "Mimi hiểu nè."
        return SimpleNamespace(text=text, usage={}, metadata={})


@pytest.mark.asyncio
async def test_tier1_keyword_triggers_deliberation_and_its_urgent_verdict_hands_off():
    store, model = InMemoryStore(), Model()
    result = await CounselingEngine(model, store).respond("u", SUICIDE)
    assert result["handoff_status"] == "pending" and len(store.alerts) == 1
    assert ("reply", None) not in model.calls
    assert "deliberation:urgent" in result["diagnostics"] and "safety_floor:suicide" not in result["diagnostics"]
    assert result["handoff_reason"] == "keyword:suicide" and result["safety_rounds"] == 2


@pytest.mark.asyncio
async def test_deliberation_runs_at_medium_effort_and_round3_at_high_effort():
    # Routable model: triage "uncertain" -> medium deliberation says "safe" with a keyword present -> high re-check.
    model = Model(risks=("uncertain",), routable=True, deliberate="safe")
    result = await CounselingEngine(model, InMemoryStore()).respond("u", SUICIDE)
    assert [c for c in model.calls if c[0] in {"safety", "deliberate"}] == [("safety", None), ("deliberate", "medium"), ("deliberate", "high")]
    assert result["safety_rounds"] == 3 and result["handoff_status"] == "none"
    assert {"deliberation:safe", "deliberation_round3:safe"} <= set(result["diagnostics"])
    # Control: without a keyword the same deliberation clears the turn after a single extra round.
    model = Model(risks=("uncertain",), routable=True, deliberate="safe")
    assert (await CounselingEngine(model, InMemoryStore()).respond("u", NEUTRAL))["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_three_different_tier2_signals_hand_off_on_the_third_message():
    store, model = InMemoryStore(), Model()
    engine = CounselingEngine(model, store)
    first = await engine.respond("u", DISAPPEAR)
    second = await engine.respond("u", BURDEN)
    assert first["handoff_status"] == second["handoff_status"] == "none"
    assert not any(d.startswith("deliberation") for d in first["diagnostics"] + second["diagnostics"])
    third = await engine.respond("u", GOODBYE)
    assert third["handoff_status"] == "pending" and len(store.alerts) == 1
    assert "deliberation:urgent" in third["diagnostics"] and third["handoff_reason"] == "accumulation"
    assert len([c for c in model.calls if c[0] == "reply"]) == 2


@pytest.mark.asyncio
async def test_single_or_repeated_same_tier2_signal_does_not_force_handoff():
    engine = CounselingEngine(Model(), InMemoryStore())
    for _ in range(4):
        result = await engine.respond("u", DISAPPEAR)
        assert result["handoff_status"] == "none"
        assert not any(d.startswith("deliberation") for d in result["diagnostics"])
    saved = await engine.store.load("u")
    assert saved["turn_count"] == 4 and {e["code"] for e in saved["risk_signals"]} == {"disappear"}


@pytest.mark.asyncio
async def test_floor_does_not_repeat_on_later_neutral_messages():
    store, model = InMemoryStore(), Model()
    engine = CounselingEngine(model, store)
    for text in (DISAPPEAR, BURDEN, GOODBYE):
        await engine.respond("u", text)
    follow = await engine.respond("u", NEUTRAL)
    assert not any(d.startswith("deliberation") for d in follow["diagnostics"])
    assert len(store.alerts) == 1
    assert follow["response_parts"] == ["Mimi hiểu nè."]


@pytest.mark.asyncio
async def test_hints_reach_the_safety_prompt_only_when_relevant():
    model = Model()
    engine = CounselingEngine(model, InMemoryStore())
    await engine.respond("u", NEUTRAL)
    assert model.safety_prompts[-1] == SAFETY
    await engine.respond("u", DISAPPEAR)
    keyword_prefix = SAFETY_HINT_KEYWORDS.split("{")[0]
    accumulated_prefix = SAFETY_HINT_ACCUMULATED.split("{")[0]
    assert SAFETY_HINT_KEYWORDS.format(codes="disappear") in model.safety_prompts[-1]
    assert accumulated_prefix not in model.safety_prompts[-1]
    await engine.respond("u", BURDEN)
    assert keyword_prefix in model.safety_prompts[-1] and accumulated_prefix not in model.safety_prompts[-1]
    await engine.respond("u", NEUTRAL)
    # Two distinct earlier codes: the accumulated hint appears even if the new message is neutral.
    assert SAFETY_HINT_ACCUMULATED.format(codes="burden, disappear") in model.safety_prompts[-1]
    assert keyword_prefix not in model.safety_prompts[-1]


@pytest.mark.asyncio
async def test_diagnostics_and_logs_carry_codes_but_never_user_text(caplog):
    secret = "ZQXSECRET"
    text = f"{SUICIDE} vì {secret}"
    caplog.set_level(logging.DEBUG, logger="mimi_v2.turn")
    result = await CounselingEngine(Model(), InMemoryStore()).respond("u", text)
    assert "deliberation:urgent" in result["diagnostics"]
    blob = json.dumps(result["diagnostics"], ensure_ascii=False) + caplog.text
    assert "suicide" in caplog.text
    assert secret not in blob and "tự tử" not in blob and text not in blob


@pytest.mark.asyncio
async def test_signals_persist_on_safe_handoff_and_discarded_screen_paths():
    # Safe path: assessment returns a deep-copied session; the turn's signals must survive it.
    store = InMemoryStore()
    await CounselingEngine(Model(), store).respond("safe", DISAPPEAR)
    assert (await store.load("safe"))["risk_signals"] == [{"code": "disappear", "turn": 1}]
    # Screen-discarded path: safety said urgent, the assessment never replaces the session.
    store = InMemoryStore()
    result = await CounselingEngine(Model(risks=("urgent",)), store).respond("urgent", DISAPPEAR)
    assert "assessment_discarded_not_safe" in result["diagnostics"]
    assert (await store.load("urgent"))["risk_signals"] == [{"code": "disappear", "turn": 1}]
    # Handoff path (floor from the third code) keeps all three codes.
    store = InMemoryStore()
    engine = CounselingEngine(Model(), store)
    for text in (DISAPPEAR, BURDEN, GOODBYE):
        await engine.respond("floor", text)
    saved = await store.load("floor")
    assert saved["handoff_status"] == "pending" and saved["turn_count"] == 3
    assert sorted((e["code"], e["turn"]) for e in saved["risk_signals"]) == [("burden", 2), ("disappear", 1), ("goodbye", 3)]
    # Only codes are stored, never message text.
    assert "muốn" not in json.dumps(saved["risk_signals"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_signals_expire_after_the_window():
    store = InMemoryStore()
    await store.save("u", {"messages": [], "turn_count": 31,
                           "risk_signals": [{"code": "burden", "turn": 1}, {"code": "goodbye", "turn": 3}]})
    await CounselingEngine(Model(), store).respond("u", NEUTRAL)
    saved = await store.load("u")
    assert saved["turn_count"] == 32 and saved["risk_signals"] == [{"code": "goodbye", "turn": 3}]


@pytest.mark.asyncio
async def test_old_session_without_new_fields_loads_and_gains_them():
    store = InMemoryStore()
    await store.save("u", {"known": {}, "user_role": "unknown", "handoff_status": "none", "notification_sent": False,
                           "messages": [{"role": "user", "content": "cũ"}, {"role": "assistant", "content": "trả lời cũ"}]})
    result = await CounselingEngine(Model(), store).respond("u", NEUTRAL)
    assert result["handoff_status"] == "none"
    saved = await store.load("u")
    assert saved["turn_count"] == 1 and saved["risk_signals"] == [] and saved["greeted"] is True
    # A bare legacy list of messages also loads.
    session = canonical_session([{"role": "user", "content": "x"}])
    assert (session["greeted"], session["turn_count"], session["risk_signals"]) == (False, 0, [])


def test_corrupt_new_fields_are_tolerated_not_fatal():
    session = canonical_session({"messages": [], "greeted": "yes", "turn_count": -3,
                                 "risk_signals": [{"code": "burden", "turn": 2, "extra": 1}, "junk", {"code": 5, "turn": 1},
                                                  {"code": "goodbye", "turn": True}, {"code": "x"}]})
    assert session["greeted"] is False and session["turn_count"] == 0
    assert session["risk_signals"] == [{"code": "burden", "turn": 2}]
    assert canonical_session({"messages": [], "risk_signals": "oops", "turn_count": "7"})["risk_signals"] == []
