"""Adaptive model/effort routing; offline, no provider or storage."""
import json
from types import SimpleNamespace

import pytest

from mimi_v2 import core
from mimi_v2.core import CounselingEngine
from mimi_v2.models import ModelConfig, ModelError, RoutedModel
from mimi_v2.store import InMemoryStore


def assessment(need="comfort"):
    return dict(known={}, missing_fields=[], next_question_focus="", need=need, intent="support",
                topic="school", core_issue="school pressure", safety_hint="safe", user_role="unknown")


class RecordingModel:
    """Routable stand-in: records (purpose, effort, fallback) and replays scripted risks/errors."""
    routable = True

    def __init__(self, risks=("safe",), errors=None, need="comfort", deliberations=("safe",)):
        self.risks, self.errors, self.need, self.calls = list(risks), {k: list(v) for k, v in (errors or {}).items()}, need, []
        self.deliberations = list(deliberations)

    async def complete(self, messages, purpose="reply", effort=None, fallback=False):
        self.calls.append((purpose, effort, fallback))
        if self.errors.get(purpose):
            raise self.errors[purpose].pop(0)
        if purpose == "safety":
            text = json.dumps({"risk": self.risks.pop(0), "reason": "fixture"})
        elif purpose == "deliberate":
            text = json.dumps(dict(cues=[], about_self=True, negated_or_denied=False, joking_or_hyperbole=False,
                                   quoted_or_hypothetical=False, intent="none", plan_or_means=False,
                                   abuse_or_danger_from_others=False, context_trend="", reason="fixture",
                                   risk=self.deliberations.pop(0)))
        elif purpose == "assessment":
            text = json.dumps(assessment(self.need))
        else:
            text = "Mimi hiểu điều bạn vừa chia sẻ."
        return SimpleNamespace(text=text, usage={"total_tokens": 3}, metadata={"purpose": purpose})


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(core, "RETRY_BACKOFF_SECONDS", 0)


async def turn(model):
    return await CounselingEngine(model, InMemoryStore()).respond("u1", "Hôm nay mình buồn")


@pytest.mark.asyncio
async def test_uncertain_safety_gets_medium_deliberation_that_can_clear_it():
    model = RecordingModel(risks=("uncertain",), deliberations=("safe",))
    result = await turn(model)
    assert [c for c in model.calls if c[0] in {"safety", "deliberate"}] == [("safety", None, False), ("deliberate", "medium", False)]
    assert result["handoff_status"] == "none" and "deliberation:safe" in result["diagnostics"] and result["safety_rounds"] == 2


@pytest.mark.asyncio
async def test_deliberation_never_lowers_urgent_triage_and_failure_keeps_triage_level():
    # Clear crisis (high-confidence urgent) is not delayed by any extra round.
    model = RecordingModel(risks=("urgent",))
    result = await turn(model)
    assert result["handoff_status"] == "pending" and result["safety_rounds"] == 1 and "deliberate" not in [c[0] for c in model.calls]
    result = await turn(RecordingModel(risks=("uncertain",), deliberations=("urgent",)))
    assert result["handoff_status"] == "pending"
    model = RecordingModel(risks=("uncertain",), errors={})
    # The deliberation call fails with a non-transient error: stays uncertain (a check-in, no alert), never "safe".
    original = model.complete

    async def flaky(messages, purpose="reply", effort=None, fallback=False):
        if purpose == "deliberate":
            raise ModelError("authentication", status=401)
        return await original(messages, purpose, effort, fallback)
    model.complete = flaky
    result = await turn(model)
    assert result["handoff_status"] == "none" and result["safety_mode"] == "checkin"
    assert "deliberation_failed" in result["diagnostics"]


@pytest.mark.asyncio
async def test_safe_safety_is_called_once_and_reply_effort_follows_need():
    model = RecordingModel(need="comfort")
    await turn(model)
    assert [c for c in model.calls if c[0] == "safety"] == [("safety", None, False)]
    assert ("reply", None, False) in model.calls
    model = RecordingModel(need="listen")
    await turn(model)
    assert ("reply", "none", False) in model.calls


@pytest.mark.asyncio
async def test_transient_reply_failure_retries_on_fallback_but_safety_does_not_switch_model():
    model = RecordingModel(errors={"reply": [ModelError("provider_error", status=503)]})
    await turn(model)
    assert [c for c in model.calls if c[0] == "reply"] == [("reply", None, False), ("reply", None, True)]


@pytest.mark.asyncio
async def test_plain_models_never_receive_router_arguments():
    class Plain:
        async def complete(self, messages, purpose):  # no effort/fallback parameters
            if purpose == "safety":
                return SimpleNamespace(text='{"risk":"uncertain","reason":"x"}', usage={}, metadata={})
            raise AssertionError("plain model has no assessment/reply")
    result = await turn(Plain())
    # No recheck for a plain model; uncertain is a check-in, and with the assessment/reply models failing too
    # the user gets the fixed 111 message instead of a teacher alert.
    assert result["handoff_status"] == "none" and not any(d.startswith("safety_rechecked") for d in result["diagnostics"])
    assert "111" in result["response"]


def test_router_defaults_and_env_overrides():
    router = RoutedModel(ModelConfig())
    assert (router._config_for("safety", None, False).model, router._config_for("safety", None, False).reasoning_effort) == ("gpt-5.4-mini", "low")
    assert router._config_for("assessment", None, False).reasoning_effort == "none"
    assert router._config_for("reply", "none", False).reasoning_effort == "none"
    # Fallback swaps the model, drops the primary-model effort, and never applies to safety.
    assert router._config_for("reply", "none", True).model == "gpt-5-mini"
    assert router._config_for("reply", "none", True).reasoning_effort == "low"
    assert router._config_for("safety", "medium", True).model == "gpt-5.4-mini"
    env = {"MIMI_V2_MODEL_REPLY": "gpt-5.4-nano", "MIMI_V2_EFFORT_SAFETY": "medium", "MIMI_V2_FALLBACK_MODEL": "off"}
    custom = RoutedModel.from_env(env)
    assert custom._config_for("reply", None, False).model == "gpt-5.4-nano"
    assert custom._config_for("safety", None, False).reasoning_effort == "medium"
    assert custom._config_for("reply", None, True).model == "gpt-5.4-nano"  # fallback disabled


def test_invalid_route_fails_at_startup_not_on_a_turn():
    with pytest.raises(ModelError) as exc:
        RoutedModel.from_env({"MIMI_V2_EFFORT_ASSESSMENT": "ultra"})
    assert exc.value.code == "unsupported_reasoning_effort"
