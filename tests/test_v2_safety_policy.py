"""Risk-level policy: only "urgent" alerts teachers. Offline, scripted models, in-memory store.

| signal                                              | action                                            |
| urgent (model / hint recheck / tier-1 / accumulation) | handoff + one teacher email                       |
| model "uncertain" (after the medium recheck)         | normal reply + SAFETY_CHECKIN, no email           |
| 2nd model "uncertain" within 5 turns                 | treated as urgent -> handoff                      |
| safety model error/invalid                           | normal reply + 111 note, no email (floor aside)   |
| safety AND assessment failed                         | fixed message with 111, no email (floor aside)    |
| assessment failure alone                             | fallback assessment, routes on safety only        |
"""
import json
import logging
from types import SimpleNamespace

import pytest

from mimi_v2 import core
from mimi_v2.contracts import normalize_assessment
from mimi_v2.core import CounselingEngine, canonical_session, fallback_assessment
from mimi_v2.models import ModelError
from mimi_v2.prompts import (ASSESSMENT, REPLY_FAILURE_MESSAGE, SAFETY, SAFETY_CHECKIN, SAFETY_UNAVAILABLE_NOTE)
from mimi_v2.store import InMemoryStore


def assessment_json(**changes):
    data = dict(known=[], missing_fields=[], next_question_focus="", need="comfort", intent="support",
                topic="school", core_issue="school pressure", safety_hint="safe", user_role="unknown")
    data.update(changes)
    return json.dumps(data, ensure_ascii=False)


class Model:
    """Scripted: `risks` is consumed per dedicated safety call (last one repeats); errors per purpose."""

    def __init__(self, risks=("safe",), hint="safe", assessment_text=None, fail=(), recheck=None, routable=False, **assess):
        self.risks, self.hint, self.fail, self.recheck, self.routable = list(risks), hint, set(fail), recheck, routable
        self.assessment_text = assessment_text
        self.assess = assess
        self.calls, self.reply_prompts = [], []

    async def complete(self, messages, purpose="reply", effort=None, fallback=False):
        self.calls.append(purpose)
        if purpose in self.fail:
            raise ModelError("timeout")
        if purpose == "safety":
            if self.recheck and "Bộ phân tích phụ" in messages[0]["content"]:
                risk = self.recheck
            else:
                risk = self.risks.pop(0) if len(self.risks) > 1 else self.risks[0]
            text = json.dumps({"risk": risk, "reason": "fixture"})
        elif purpose == "assessment":
            text = self.assessment_text or assessment_json(safety_hint=self.hint, **self.assess)
        else:
            self.reply_prompts.append(messages[0]["content"])
            text = "Mimi nghe nè."
        return SimpleNamespace(text=text, usage={}, metadata={})


class Store(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.notified = 0

    async def notify(self, user_id, text):
        self.notified += 1
        return await super().notify(user_id, text)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(core, "RETRY_BACKOFF_SECONDS", 0)


# --------------------------------------------------------------------------- production incident

@pytest.mark.asyncio
async def test_duplicate_known_key_in_assessment_no_longer_causes_a_handoff():
    # The Railway case: safety said "safe", the assessment repeated a `known` key with a different value.
    dup = assessment_json(known=[{"key": "ý_kiến", "value": "lòng vòng"}, {"key": "ý_kiến", "value": "không hiểu nổi"}])
    store, model = Store(), Model(assessment_text=dup)
    result = await CounselingEngine(model, store).respond("u", "bạn hay trả lời lòng vòng nhiều lúc không hiểu nổi bạn á")
    assert result["handoff_status"] == "none" and store.notified == 0 and result["handoff_reason"] is None
    assert result["diagnostics"] == [] and model.calls == ["safety", "assessment", "reply"]
    assert result["assessment"]["known"] == {"ý_kiến": "không hiểu nổi"}  # the newest value wins


def test_normalize_assessment_duplicate_keys_last_wins():
    out = normalize_assessment({"known": [{"key": "a", "value": "1"}, {"key": "b", "value": "x"}, {"key": "a", "value": "2"}]})
    assert out["known"] == {"a": "2", "b": "x"}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["ừ", "ờ", "à", "ok", "uk", "hmm", ".",
                                  "sao cậu trả lời sai hoài", "lòng vòng quá", "chán ghê", "mệt thật sự, kiểm tra hoài"])
async def test_fillers_and_complaints_judged_safe_never_alert(text):
    store, model = Store(), Model()
    result = await CounselingEngine(model, store).respond("u", text)
    assert result["handoff_status"] == "none" and store.notified == 0
    assert result["safety_mode"] == "normal" and result["handoff_reason"] is None
    assert SAFETY_CHECKIN not in model.reply_prompts[-1]


def test_safety_prompt_teaches_short_replies_and_complaints_are_safe():
    for needle in ('"ừ"', '"ờ"', '"ok"', "phàn nàn", "lòng vòng", "chán ghê", "KHÔNG đủ", "TIN MỚI NHẤT", "Khủng hoảng cũ"):
        assert needle in SAFETY
    assert len(SAFETY) < 2200
    assert "Phân vân giữa safe và uncertain thì chọn uncertain" not in SAFETY
    hint = ASSESSMENT[ASSESSMENT.index("safety_hint nâng"):]
    assert '"ừ"' in hint and "phàn nàn" in hint


# --------------------------------------------------------------------------- uncertain -> check-in

@pytest.mark.asyncio
async def test_model_uncertain_is_a_checkin_reply_without_alert():
    store, model = Store(), Model(risks=("uncertain",))
    engine = CounselingEngine(model, store)
    result = await engine.respond("u", "chắc mình mệt quá rồi")
    assert result["handoff_status"] == "none" and store.notified == 0 and result["handoff_reason"] is None
    assert result["safety_mode"] == "checkin" and "safety_checkin" in result["diagnostics"]
    assert "reply" in model.calls and SAFETY_CHECKIN in model.reply_prompts[-1]
    assert "111" in SAFETY_CHECKIN and "Tổng đài 111" in SAFETY_CHECKIN
    saved = await store.load("u")
    assert saved["uncertain_turns"] == [1] and saved["handoff_status"] == "none"
    # Safe turns afterwards do not carry the check-in prompt.
    model.risks = ["safe"]
    await engine.respond("u", "ừ")
    assert SAFETY_CHECKIN not in model.reply_prompts[-1]


@pytest.mark.asyncio
async def test_checkin_overrides_closing_shortcut_and_listen_rules():
    for changes in ({"intent": "closing", "need": "comfort"}, {"intent": "support", "need": "listen"}, {"intent": "closing", "need": "listen"}):
        model = Model(risks=("uncertain",), **changes)
        result = await CounselingEngine(model, Store()).respond("u", "tạm biệt mọi người nha")
        assert "reply" in model.calls and SAFETY_CHECKIN in model.reply_prompts[-1], changes
        assert result["response_parts"][0] == "Mimi nghe nè."  # not the canned goodbye (reply precedes the greeting)


@pytest.mark.asyncio
async def test_second_model_uncertain_within_five_turns_is_urgent():
    store, model = Store(), Model(risks=("uncertain", "safe", "safe", "safe", "uncertain"))
    engine = CounselingEngine(model, store)
    first = await engine.respond("u", "m1")            # turn 1: uncertain -> check-in
    for text in ("m2", "m3", "m4"):                    # turns 2-4: safe
        await engine.respond("u", text)
    fifth = await engine.respond("u", "m5")            # turn 5: uncertain again, 4 turns after the first
    assert first["safety_mode"] == "checkin" and store.notified == 1
    assert fifth["handoff_status"] == "pending" and fifth["handoff_reason"] == "model:uncertain_repeat"
    assert "uncertain_repeat_escalated" in fifth["diagnostics"] and "reply" not in model.calls[-2:]
    assert "Mình đang nhờ thầy cô" in fifth["response_parts"][-1]


@pytest.mark.asyncio
async def test_uncertain_outside_the_window_is_a_fresh_checkin():
    store = Store()
    await store.save("u", {"messages": [], "turn_count": 10, "uncertain_turns": [4]})  # 7 turns ago at turn 11
    result = await CounselingEngine(Model(risks=("uncertain",)), store).respond("u", "hmm")
    assert result["safety_mode"] == "checkin" and result["handoff_status"] == "none" and store.notified == 0
    assert (await store.load("u"))["uncertain_turns"] == [11]
    # Exactly 5 turns ago still counts.
    store = Store()
    await store.save("u", {"messages": [], "turn_count": 10, "uncertain_turns": [6]})
    result = await CounselingEngine(Model(risks=("uncertain",)), store).respond("u", "hmm")
    assert result["handoff_reason"] == "model:uncertain_repeat" and store.notified == 1


@pytest.mark.asyncio
async def test_uncertain_hint_recheck_counts_as_an_uncertain_turn_and_repeat_escalates():
    store = Store()
    engine = CounselingEngine(Model(hint="uncertain", recheck="uncertain"), store)
    first = await engine.respond("u", "a")
    assert first["safety_mode"] == "checkin" and store.notified == 0
    assert (await store.load("u"))["uncertain_turns"] == [1]
    second = await engine.respond("u", "b")
    assert second["handoff_reason"] == "model:uncertain_repeat" and store.notified == 1


@pytest.mark.asyncio
async def test_routable_uncertain_cleared_by_medium_recheck_is_not_recorded():
    store, model = Store(), Model(risks=("uncertain", "safe"), routable=True)
    result = await CounselingEngine(model, store).respond("u", "ừ")
    assert result["safety_mode"] == "normal" and (await store.load("u"))["uncertain_turns"] == []


# --------------------------------------------------------------------------- urgent -> handoff

@pytest.mark.asyncio
async def test_urgent_model_hands_off_once_with_reason():
    store, model = Store(), Model(risks=("urgent",))
    engine = CounselingEngine(model, store)
    first = await engine.respond("u", "mình không ổn lắm")
    second = await engine.respond("u", "thật đó")
    assert first["handoff_status"] == "pending" and first["handoff_reason"] == "model:urgent" and first["safety_mode"] == "handoff"
    assert second["handoff_status"] == "pending" and store.notified == 1  # notification is sent once
    assert "reply" not in model.calls and first["assessment"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("risks", [("safe",), ("uncertain",)])
async def test_tier1_keyword_hands_off_even_if_model_says_safe_or_uncertain(risks):
    store, model = Store(), Model(risks=risks)
    result = await CounselingEngine(model, store).respond("u", "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and store.notified == 1
    assert result["handoff_reason"] == "keyword:suicide" and "safety_floor:suicide" in result["diagnostics"]
    assert "reply" not in model.calls


@pytest.mark.asyncio
async def test_model_can_never_lower_the_floor_even_with_recheck_and_hint_safe():
    store, model = Store(), Model(risks=("uncertain", "safe"), routable=True, recheck="safe")
    result = await CounselingEngine(model, store).respond("u", "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and result["handoff_reason"].startswith("keyword:")


@pytest.mark.asyncio
async def test_accumulation_floor_is_urgent_with_reason():
    store, model = Store(), Model()
    engine = CounselingEngine(model, store)
    results = [await engine.respond("u", t) for t in ("tớ muốn biến mất", "tớ thấy mình là gánh nặng", "tạm biệt mọi người")]
    assert [r["handoff_status"] for r in results[:2]] == ["none", "none"]
    assert results[2]["handoff_status"] == "pending" and results[2]["handoff_reason"] == "accumulation"
    assert store.notified == 1


@pytest.mark.asyncio
async def test_assessment_hint_urgent_confirmed_by_recheck_hands_off():
    store, model = Store(), Model(hint="urgent", recheck="urgent")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "model:urgent" and store.notified == 1


@pytest.mark.asyncio
async def test_safety_uncertain_but_assessment_hint_urgent_gets_a_recheck():
    # Two model opinions disagree upward: one stronger safety opinion decides.
    store, model = Store(), Model(risks=("uncertain",), hint="urgent", recheck="urgent")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and model.calls.count("safety") == 2
    store, model = Store(), Model(risks=("uncertain",), hint="urgent", recheck="safe")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["safety_mode"] == "checkin" and store.notified == 0  # never lower than the safety model's own "uncertain"


# --------------------------------------------------------------------------- degraded paths

@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ModelError("timeout"), ModelError("provider_error", status=503), ModelError("authentication", status=401)])
async def test_safety_error_replies_with_111_note_and_no_email(error):
    class SafetyDown(Model):
        async def complete(self, messages, purpose="reply", effort=None, fallback=False):
            if purpose == "safety":
                self.calls.append(purpose)
                raise error
            return await super().complete(messages, purpose, effort, fallback)
    store, model = Store(), SafetyDown()
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and store.notified == 0 and result["handoff_reason"] is None
    assert result["safety_mode"] == "unavailable" and "safety_unavailable_or_invalid" in result["diagnostics"]
    assert f"safety_model_error:{error.code}:{error.status or '-'}" in result["diagnostics"]
    assert result["response_parts"][-1] == "Mimi nghe nè.\n\n" + SAFETY_UNAVAILABLE_NOTE
    assert "111" in SAFETY_UNAVAILABLE_NOTE


@pytest.mark.asyncio
async def test_safety_error_with_keyword_floor_still_alerts():
    store, model = Store(), Model(fail={"safety"})
    result = await CounselingEngine(model, store).respond("u", "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and store.notified == 1 and result["handoff_reason"] == "keyword:suicide"
    assert "safety_unavailable_or_invalid" in result["diagnostics"]


@pytest.mark.asyncio
async def test_safety_error_but_urgent_assessment_hint_is_rechecked_not_ignored():
    # The dedicated check failed once; the stronger recheck answers: its verdict decides.
    class FlakySafety(Model):
        async def complete(self, messages, purpose="reply", effort=None, fallback=False):
            if purpose == "safety" and "Bộ phân tích phụ" not in messages[0]["content"]:
                self.calls.append(purpose)
                raise ModelError("timeout")
            return await super().complete(messages, purpose, effort, fallback)
    store, model = Store(), FlakySafety(hint="urgent", recheck="urgent")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and result["handoff_reason"] == "model:urgent"
    store, model = Store(), FlakySafety(hint="urgent", recheck="safe")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and result["safety_mode"] == "normal"  # the safety model answered "safe"


@pytest.mark.asyncio
async def test_both_safety_and_assessment_fail_sends_fixed_111_message_no_email():
    store, model = Store(), Model(fail={"safety", "assessment"})
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and store.notified == 0 and result["safety_mode"] == "degraded"
    assert "reply" not in model.calls
    assert result["response_parts"][-1] == REPLY_FAILURE_MESSAGE and "111" in REPLY_FAILURE_MESSAGE
    assert "không cần kể lại" in REPLY_FAILURE_MESSAGE
    assert (await store.load("u"))["messages"][-1]["content"] == REPLY_FAILURE_MESSAGE


@pytest.mark.asyncio
async def test_reply_failure_message_carries_111():
    store, model = Store(), Model(fail={"reply"})
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["response_parts"][-1] == REPLY_FAILURE_MESSAGE and "111" in result["response"]
    assert result["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_assessment_failure_uses_session_fallback_and_routes_on_safety_only():
    store = Store()
    await store.save("u", {"messages": [], "known": {"grade": "8"}, "user_role": "student", "need": "listen",
                           "topic": "bạn bè", "core_issue": "bị bơ", "handoff_status": "none"})
    model = Model(fail={"assessment"})
    result = await CounselingEngine(model, store).respond("u", "ừ")
    assert result["handoff_status"] == "none" and store.notified == 0 and "assessment_fallback" in result["diagnostics"]
    assert result["assessment"] == {"known": {"grade": "8"}, "missing_fields": [], "next_question_focus": "", "need": "listen",
                                    "intent": "support", "topic": "bạn bè", "core_issue": "bị bơ", "safety_hint": "safe",
                                    "user_role": "student"}
    saved = await store.load("u")
    assert saved["known"] == {"grade": "8"} and saved["need"] == "listen"  # session facts untouched by the failure


def test_fallback_assessment_defaults_and_ignores_garbage():
    out = fallback_assessment({"known": {}, "need": "bogus", "user_role": "alien", "topic": 5, "core_issue": None})
    assert out == {"known": {}, "missing_fields": [], "next_question_focus": "", "need": "comfort", "intent": "support",
                   "topic": "", "core_issue": "", "safety_hint": "safe", "user_role": "unknown"}


# --------------------------------------------------------------------------- bookkeeping

def test_uncertain_turns_are_normalized_on_load():
    assert canonical_session({"messages": []})["uncertain_turns"] == []
    assert canonical_session({"messages": [], "uncertain_turns": "x"})["uncertain_turns"] == []
    assert canonical_session({"messages": [], "uncertain_turns": [3, True, -1, "4", 7]})["uncertain_turns"] == [3, 7]


@pytest.mark.asyncio
async def test_handoff_reason_is_logged_as_a_code_never_text(caplog):
    secret = "ZQXSECRET"
    caplog.set_level(logging.DEBUG, logger="mimi_v2.turn")
    result = await CounselingEngine(Model(), Store()).respond("u", f"tớ muốn tự tử vì {secret}")
    line = next(r.getMessage() for r in caplog.records if r.name == "mimi_v2.turn")
    info = json.loads(line.split("turn ", 1)[1])
    assert info["handoff_reason"] == "keyword:suicide" and info["safety_mode"] == "handoff"
    assert secret not in line and secret not in json.dumps(result["diagnostics"])


@pytest.mark.asyncio
async def test_accepted_takeover_exposes_mode_and_no_reason():
    store = Store()
    await store.save("u", {"messages": [], "handoff_status": "accepted"})
    result = await CounselingEngine(Model(), store).respond("u", "xin chào")
    assert result["safety_mode"] == "takeover" and result["handoff_reason"] is None and result["response"] == ""
