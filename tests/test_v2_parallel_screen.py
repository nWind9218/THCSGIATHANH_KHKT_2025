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
    def __init__(self, risk="safe", hint="safe", fail=None, delay=0.05):
        self.risk, self.hint, self.fail, self.delay = risk, hint, fail, delay
        self.calls, self.active, self.max_active = [], 0, 0

    async def complete(self, messages, purpose):
        self.calls.append(purpose)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail == purpose:
                raise ModelError("timeout")
            text = {"safety": json.dumps({"risk": self.risk, "reason": "x"}),
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
async def test_assessment_hint_urgent_still_triggers_handoff():
    store, model = Store(), Model(hint="urgent")
    result = await CounselingEngine(model, store).respond("u", "xin chào")
    assert result["handoff_status"] == "pending" and "reply" not in model.calls and store.notified == 1


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
