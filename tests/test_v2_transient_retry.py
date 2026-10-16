"""Transient-failure retry and per-turn diagnostic log; offline, no provider or storage."""
import json
import logging
from types import SimpleNamespace

import pytest

from mimi_v2 import core
from mimi_v2.core import CounselingEngine, is_transient
from mimi_v2.models import ModelError
from mimi_v2.store import InMemoryStore


def assessment():
    return dict(known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                topic="school", core_issue="school pressure", safety_hint="safe", user_role="unknown")


class ScriptedModel:
    """Raises the scripted errors for a purpose first, then answers normally."""
    def __init__(self, errors=None):
        self.errors = {k: list(v) for k, v in (errors or {}).items()}
        self.calls = []

    async def complete(self, messages, purpose):
        self.calls.append(purpose)
        if self.errors.get(purpose):
            raise self.errors[purpose].pop(0)
        if purpose == "safety":
            text = json.dumps({"risk": "safe", "reason": "fixture"})
        elif purpose == "assessment":
            text = json.dumps(assessment())
        else:
            text = "Mimi hiểu điều bạn vừa chia sẻ."
        return SimpleNamespace(text=text, usage={"total_tokens": 3}, metadata={"purpose": purpose})


class CountingStore(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.notified = 0

    async def notify(self, user_id, text):
        self.notified += 1
        return True


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(core, "RETRY_BACKOFF_SECONDS", 0)


@pytest.mark.parametrize("exc,expected", [
    (ModelError("provider_error", status=503), True),
    (ModelError("provider_error", status=500), True),
    (ModelError("network_error"), True),
    (ModelError("rate_limit", status=429), True),
    (ModelError("timeout"), False),
    (ModelError("authentication", status=401), False),
    (ModelError("insufficient_quota"), False),
    (ModelError("provider_error", status=400), False),
    (ModelError("provider_error"), False),
    (ModelError("refusal"), False),
    (ConnectionError("x"), False),
])
def test_only_fast_transport_failures_are_transient(exc, expected):
    assert is_transient(exc) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["safety", "assessment"])
async def test_one_transient_failure_is_retried_without_handoff(stage):
    store = CountingStore()
    model = ScriptedModel({stage: [ModelError("provider_error", status=503)]})
    result = await CounselingEngine(model, store).respond("u", "Hôm nay mình buồn quá")
    assert result["handoff_status"] == "none" and store.notified == 0
    assert model.calls.count(stage) == 2
    assert f"{stage}_retried:provider_error:503" in result["diagnostics"]


@pytest.mark.asyncio
async def test_second_transient_failure_still_fails_closed_with_error_code():
    store = CountingStore()
    errors = [ModelError("provider_error", status=503), ModelError("provider_error", status=502)]
    model = ScriptedModel({"safety": errors})
    result = await CounselingEngine(model, store).respond("u", "Hôm nay mình buồn quá")
    assert result["handoff_status"] == "pending" and store.notified == 1
    assert model.calls.count("safety") == 2
    assert "safety_unavailable_or_invalid" in result["diagnostics"]
    assert "safety_model_error:provider_error:502" in result["diagnostics"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ModelError("timeout"), ModelError("authentication", status=401), ModelError("insufficient_quota")])
async def test_non_transient_failure_is_not_retried(error):
    model = ScriptedModel({"safety": [error, error]})
    result = await CounselingEngine(model, CountingStore()).respond("u", "xin chào")
    assert model.calls.count("safety") == 1 and result["handoff_status"] == "pending"
    assert f"safety_model_error:{error.code}:{error.status or '-'}" in result["diagnostics"]


@pytest.mark.asyncio
async def test_slow_failed_attempt_is_not_retried(monkeypatch):
    monkeypatch.setattr(core, "RETRY_MAX_ATTEMPT_SECONDS", -1.0)
    model = ScriptedModel({"safety": [ModelError("provider_error", status=503)]})
    await CounselingEngine(model, CountingStore()).respond("u", "xin chào")
    assert model.calls.count("safety") == 1


@pytest.mark.asyncio
async def test_reply_failure_is_never_retried():
    model = ScriptedModel({"reply": [ModelError("provider_error", status=503)]})
    result = await CounselingEngine(model, CountingStore()).respond("u", "xin chào")
    assert model.calls.count("reply") == 1 and "reply_unavailable_or_invalid" in result["diagnostics"]


@pytest.mark.asyncio
async def test_turn_log_has_codes_and_timings_but_no_content_or_raw_user_id(caplog):
    secret, uid = "Bí mật riêng tư của em", "psid-1234567890"
    model = ScriptedModel({"safety": [ModelError("provider_error", status=503), ModelError("provider_error", status=503)]})
    with caplog.at_level(logging.INFO, logger="mimi_v2.turn"):
        await CounselingEngine(model, CountingStore()).respond(uid, secret)
    records = [r for r in caplog.records if r.name == "mimi_v2.turn"]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    line = records[0].getMessage()
    info = json.loads(line.split("turn ", 1)[1])
    assert info["handoff"] == "pending" and "safety_model_error:provider_error:503" in info["diagnostics"]
    assert "total" in info["timings"]
    assert secret not in line and uid not in line and "Mình rất lo cho bạn" not in line


@pytest.mark.asyncio
async def test_healthy_turn_logs_at_info_level(caplog):
    with caplog.at_level(logging.INFO, logger="mimi_v2.turn"):
        await CounselingEngine(ScriptedModel(), CountingStore()).respond("u", "xin chào")
    records = [r for r in caplog.records if r.name == "mimi_v2.turn"]
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert json.loads(records[0].getMessage().split("turn ", 1)[1])["calls"] == ["safety", "assessment", "reply"]
