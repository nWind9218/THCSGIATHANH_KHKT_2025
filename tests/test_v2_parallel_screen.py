"""Safety and assessment run concurrently; safety remains the gate. Offline."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from mimi_v2.core import CounselingEngine
from mimi_v2.models import ModelError
from mimi_v2.prompts import REPLY_FAILURE_MESSAGE, SAFETY_UNAVAILABLE_NOTE
from mimi_v2.store import InMemoryStore


def assessment(**changes):
    result = dict(known={"grade": "8"}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                  topic="school", core_issue="school pressure", safety_hint="safe", user_role="unknown")
    result.update(changes)
    return result


class Model:
    def __init__(self, risk="safe", hint="safe", fail=None, delay=0.05, recheck=None, routable=False):
        # recheck: answer to the second safety opinion asked when only the assessment hint flagged the turn
        # ("fail" makes that call raise).
        self.risk, self.hint, self.fail, self.delay, self.recheck, self.routable = risk, hint, fail, delay, recheck, routable
        self.efforts = []
        self.calls, self.active, self.max_active = [], 0, 0

    async def complete(self, messages, purpose, effort=None, fallback=False):
        self.calls.append(purpose)
        is_recheck = purpose == "safety" and "Bộ phân tích phụ" in messages[0]["content"]
        self.efforts.append((purpose, effort, is_recheck))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail == purpose or (is_recheck and self.recheck == "fail"):
                raise ModelError("timeout")
            risk = self.recheck if is_recheck and self.recheck else self.risk
            text = {"safety": json.dumps({"risk": risk, "reason": "x"}),
                    "assessment": json.dumps(assessment(safety_hint=self.hint)),
                    "reply": "Mimi hiểu."}[purpose]
            return SimpleNamespace(text=text, usage={"total_tokens": 1}, metadata={})
        finally:
            self.active -= 1


class Store(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.notified = 0

    async def notify(self, user_id, text):
        self.notified += 1
        return True


@pytest.mark.asyncio
async def test_safety_and_assessment_overlap_in_time():
    model = Model()
    result = await CounselingEngine(model, Store()).respond("u", "xin chào")
    assert model.max_active == 2
    by = {s["name"]: s for s in result["trace"]["spans"]}
    safety, assess = by["model.safety"], by["model.assessment"]
    assert assess["start"] < safety["start"] + safety["duration"] and safety["start"] < assess["start"] + assess["duration"]
    # Sequential would need >= 3 model delays; concurrent needs about 2 (screen + reply).
    assert result["trace"]["total"] < 0.05 * 2.8
    assert result["trace"]["unaccounted"] < 0.05
    assert result["handoff_status"] == "none" and result["assessment"]["known"] == {"grade": "8"}


@pytest.mark.asyncio
async def test_urgent_safety_discards_assessment_and_hands_off():
    store, model = Store(), Model(risk="urgent")
    engine = CounselingEngine(model, store)
    result = await engine.respond("u", "tin nhắn")
    assert result["handoff_status"] == "pending" and store.notified == 1
    assert "reply" not in model.calls
    assert result["assessment"] == {}
    saved = await store.load("u")
    assert saved["known"] == {}  # assessment facts were not stored
    assert "assessment_discarded_not_safe" in result["diagnostics"]
    assert sorted(c["purpose"] for c in result["model_calls"]) == ["assessment", "safety"]
    assert result["handoff_reason"] == "model:urgent" and result["safety_mode"] == "handoff"


@pytest.mark.asyncio
async def test_assessment_failure_with_safe_safety_continues_with_fallback_and_no_alert():
    store, model = Store(), Model(fail="assessment")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and store.notified == 0 and "reply" in model.calls
    assert "assessment_unavailable_or_invalid" in result["diagnostics"] and "assessment_fallback" in result["diagnostics"]
    assert "assessment_model_error:timeout:-" in result["diagnostics"]
    assert result["handoff_reason"] is None


@pytest.mark.asyncio
async def test_assessment_failure_still_hands_off_when_safety_says_urgent():
    store, model = Store(), Model(risk="urgent", fail="assessment")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and store.notified == 1 and "reply" not in model.calls
    assert result["handoff_reason"] == "model:urgent"


@pytest.mark.asyncio
async def test_safety_failure_replies_normally_with_a_111_note_and_no_alert():
    store, model = Store(), Model(fail="safety")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and store.notified == 0 and "reply" in model.calls
    assert "safety_unavailable_or_invalid" in result["diagnostics"] and result["safety_mode"] == "unavailable"
    assert result["response_parts"][-1].endswith(SAFETY_UNAVAILABLE_NOTE) and "111" in result["response_parts"][-1]
    assert (await store.load("u"))["known"] == {"grade": "8"}


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["urgent", "uncertain"])
async def test_assessment_hint_confirmed_by_second_opinion_hands_off(hint):
    store, model = Store(), Model(hint=hint, recheck="urgent")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and "reply" not in model.calls and store.notified == 1
    assert "assessment_hint_rechecked:urgent" in result["diagnostics"]
    assert "assessment_discarded_not_safe" in result["diagnostics"]
    assert result["assessment"] == {} and (await store.load("u"))["known"] == {}
    assert model.calls.count("safety") == 2


@pytest.mark.asyncio
async def test_assessment_hint_cleared_by_second_opinion_continues_normally():
    store, model = Store(), Model(hint="uncertain", recheck="safe")
    result = await CounselingEngine(model, store).respond("u", "nay bị gọi lên bảng xỉu ngang")
    assert result["handoff_status"] == "none" and store.notified == 0 and "reply" in model.calls
    assert "assessment_hint_rechecked:safe" in result["diagnostics"]
    assert "assessment_discarded_not_safe" not in result["diagnostics"]
    assert result["assessment"]["known"] == {"grade": "8"}  # the assessment was kept, not discarded
    assert model.calls.count("safety") == 2


@pytest.mark.asyncio
async def test_failed_second_opinion_keeps_the_hint_level():
    # hint "uncertain" -> check-in reply (no alert); hint "urgent" -> handoff (never silently lowered).
    store, model = Store(), Model(hint="uncertain", recheck="fail")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and "reply" in model.calls and store.notified == 0
    assert "assessment_hint_recheck_failed" in result["diagnostics"] and result["safety_mode"] == "checkin"
    store, model = Store(), Model(hint="urgent", recheck="fail")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and "reply" not in model.calls and store.notified == 1
    assert "assessment_hint_recheck_failed" in result["diagnostics"] and result["handoff_reason"] == "model:urgent"


@pytest.mark.asyncio
@pytest.mark.parametrize("recheck,mode,handoff", [("uncertain", "checkin", "none"), ("safe", "normal", "none"), ("urgent", "handoff", "pending")])
async def test_hint_recheck_levels_map_to_checkin_normal_or_handoff(recheck, mode, handoff):
    store, model = Store(), Model(hint="uncertain", recheck=recheck)
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert (result["safety_mode"], result["handoff_status"]) == (mode, handoff)
    assert store.notified == (1 if handoff == "pending" else 0)
    assert ("reply" in model.calls) == (handoff == "none")


@pytest.mark.asyncio
async def test_second_opinion_prompt_effort_and_agreeing_hint_costs_nothing_extra():
    model = Model(hint="uncertain", recheck="safe", routable=True)
    await CounselingEngine(model, Store()).respond("u", "xin chào")
    assert [e for e in model.efforts if e[0] == "safety"] == [("safety", None, False), ("safety", "medium", True)]
    # No disagreement (both safe): exactly one safety call, no extra latency.
    model = Model()
    await CounselingEngine(model, Store()).respond("u", "xin chào")
    assert model.calls.count("safety") == 1


@pytest.mark.asyncio
async def test_deterministic_floor_still_wins_over_a_second_opinion_of_safe():
    # Whole turn: the keyword floor already hands off, and no second opinion is even needed.
    store, model = Store(), Model(hint="uncertain", recheck="safe")
    result = await CounselingEngine(model, store).respond("u", "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and "reply" not in model.calls
    assert "safety_floor:suicide" in result["diagnostics"]
    assert result["handoff_reason"] == "keyword:suicide"
    # The recheck itself re-applies the floor: a "safe" answer cannot lower it.
    engine = CounselingEngine(Model(recheck="safe"), Store())
    session = {"messages": [], "risk_signals": [], "turn_count": 1}
    state = {"session": session, "signals": {"tier1": ["suicide"], "tier2": []}, "diagnostics": [], "timings": {}, "usage": {}, "model_calls": []}
    assert await engine._recheck_hint(state, "uncertain") == "urgent"
    assert "assessment_hint_rechecked:safe" in state["diagnostics"] and "safety_floor:suicide" in state["diagnostics"]
    # Same for the accumulated tier-2 floor (3 distinct codes, one of them this turn).
    session = {"messages": [], "risk_signals": [{"code": "burden", "turn": 1}, {"code": "goodbye", "turn": 2}, {"code": "disappear", "turn": 3}], "turn_count": 3}
    state = {"session": session, "signals": {"tier1": [], "tier2": ["disappear"]}, "diagnostics": [], "timings": {}, "usage": {}, "model_calls": []}
    assert await engine._recheck_hint(state, "uncertain") == "urgent"


@pytest.mark.asyncio
async def test_both_calls_fail_sends_fixed_111_message_without_alert():
    class BothFail(Model):
        async def complete(self, messages, purpose):
            self.calls.append(purpose)
            raise ModelError("authentication", status=401)
    store, model = Store(), BothFail()
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "none" and store.notified == 0 and result["safety_mode"] == "degraded"
    assert sorted(model.calls) == ["assessment", "safety"]  # no reply call
    assert result["response_parts"][-1] == REPLY_FAILURE_MESSAGE and "111" in REPLY_FAILURE_MESSAGE
    assert "safety_and_assessment_unavailable" in result["diagnostics"]


@pytest.mark.asyncio
async def test_both_calls_fail_but_keyword_floor_still_alerts_once():
    class BothFail(Model):
        async def complete(self, messages, purpose):
            self.calls.append(purpose)
            raise ModelError("authentication", status=401)
    store = Store()
    result = await CounselingEngine(BothFail(), store).respond("u", "tớ muốn tự tử")
    assert result["handoff_status"] == "pending" and store.notified == 1
    assert result["handoff_reason"] == "keyword:suicide"
