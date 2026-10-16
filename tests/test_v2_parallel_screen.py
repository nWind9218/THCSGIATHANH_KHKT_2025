"""Safety and assessment run concurrently; safety remains the gate. Offline."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from mimi_v2.core import CounselingEngine
from mimi_v2.models import ModelError
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
@pytest.mark.parametrize("risk", ["urgent", "uncertain"])
async def test_unsafe_safety_discards_assessment_and_hands_off(risk):
    store, model = Store(), Model(risk=risk)
    engine = CounselingEngine(model, store)
    result = await engine.respond("u", "tin nhắn")
    assert result["handoff_status"] == "pending" and store.notified == 1
    assert "reply" not in model.calls
    assert result["assessment"] == {}
    saved = await store.load("u")
    assert saved["known"] == {}  # assessment facts were not stored
    assert "assessment_discarded_not_safe" in result["diagnostics"]
    assert sorted(c["purpose"] for c in result["model_calls"]) == ["assessment", "safety"]


@pytest.mark.asyncio
async def test_assessment_failure_with_safe_safety_still_fails_closed():
    store, model = Store(), Model(fail="assessment")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and store.notified == 1 and "reply" not in model.calls
    assert "assessment_unavailable_or_invalid" in result["diagnostics"]
    assert "assessment_model_error:timeout:-" in result["diagnostics"]


@pytest.mark.asyncio
async def test_safety_failure_fails_closed_even_if_assessment_succeeds():
    store, model = Store(), Model(fail="safety")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and "reply" not in model.calls
    assert "safety_unavailable_or_invalid" in result["diagnostics"]
    assert (await store.load("u"))["known"] == {}


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
async def test_failed_second_opinion_fails_closed_to_the_hint():
    store, model = Store(), Model(hint="uncertain", recheck="fail")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and "reply" not in model.calls and store.notified == 1
    assert "assessment_hint_recheck_failed" in result["diagnostics"]


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
    # The recheck itself re-applies the floor: a "safe" answer cannot lower it.
    engine = CounselingEngine(Model(recheck="safe"), Store())
    session = {"messages": [], "risk_signals": [], "turn_count": 1}
    state = {"session": session, "signals": {"tier1": ["suicide"], "tier2": []}, "diagnostics": [], "timings": {}, "usage": {}, "model_calls": []}
    assert await engine._recheck_hint(state, "uncertain") == "uncertain"
    assert "assessment_hint_rechecked:safe" in state["diagnostics"] and "safety_floor:suicide" in state["diagnostics"]
    # Same for the accumulated tier-2 floor (3 distinct codes, one of them this turn).
    session = {"messages": [], "risk_signals": [{"code": "burden", "turn": 1}, {"code": "goodbye", "turn": 2}, {"code": "disappear", "turn": 3}], "turn_count": 3}
    state = {"session": session, "signals": {"tier1": [], "tier2": ["disappear"]}, "diagnostics": [], "timings": {}, "usage": {}, "model_calls": []}
    assert await engine._recheck_hint(state, "uncertain") == "uncertain"


@pytest.mark.asyncio
async def test_both_calls_both_fail_notifies_once():
    class BothFail(Model):
        async def complete(self, messages, purpose):
            self.calls.append(purpose)
            raise ModelError("authentication", status=401)
    store, model = Store(), BothFail()
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and store.notified == 1
    assert sorted(model.calls) == ["assessment", "safety"]
