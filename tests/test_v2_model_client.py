"""The AsyncOpenAI client is built once and shared; it must never be rebuilt per call."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from mimi_v2.models import ModelConfig, OpenAIModel, RoutedModel

SHINE = "https://api.shineshop.dev/v1"


def _response():
    return {"status": "completed", "model": "gpt-5-mini", "usage": {"total_tokens": 3},
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "{\"ok\": true}"}]}]}


@pytest.fixture
def factory(monkeypatch):
    built = []

    def make(**kwargs):
        fake = MagicMock()
        fake.kwargs = kwargs
        fake.responses.create = AsyncMock(return_value=_response())
        fake.close = AsyncMock()
        built.append(fake)
        return fake

    monkeypatch.setattr(openai, "AsyncOpenAI", make)
    for name in ("MIMI_V2_BASE_URL", "MIMI_V2_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    return built


MSG = [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_one_client_for_many_calls(factory):
    model = OpenAIModel(ModelConfig(model="gpt-5-mini"), api_key="sk-test")
    for _ in range(4):
        await model.complete(MSG)
    assert len(factory) == 1
    factory[0].close.assert_not_awaited()
    assert factory[0].kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_routed_purposes_and_fallback_share_client(factory):
    routed = RoutedModel(api_key="sk-test")
    await routed.complete(MSG, purpose="reply")
    await routed.complete(MSG, purpose="safety")
    await routed.complete(MSG, purpose="assessment")
    await routed.complete(MSG, purpose="reply", fallback=True)
    await routed.complete(MSG, purpose="reply", effort="medium")
    assert len(factory) == 1
    assert factory[0].responses.create.await_count == 5
    await routed.aclose()
    factory[0].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_distinct_base_url_gets_distinct_client_and_key(factory):
    openai_model = OpenAIModel(api_key="sk-openai")
    shine_model = OpenAIModel(api_key="shine-key", base_url=SHINE)
    await openai_model.complete(MSG)
    await shine_model.complete(MSG)
    await openai_model.complete(MSG)
    assert len(factory) == 2
    by_url = {c.kwargs["base_url"]: c.kwargs["api_key"] for c in factory}
    assert by_url == {"https://api.openai.com/v1": "sk-openai", SHINE: "shine-key"}


@pytest.mark.asyncio
async def test_routed_distinct_endpoints_never_share(factory):
    a = RoutedModel(api_key="sk-openai")
    b = RoutedModel(api_key="shine-key", base_url=SHINE)
    await a.complete(MSG)
    await b.complete(MSG)
    assert {c.kwargs["api_key"] for c in factory} == {"sk-openai", "shine-key"}
    assert len(factory) == 2


@pytest.mark.asyncio
async def test_timeout_applied_per_call_on_shared_client(factory):
    routed = RoutedModel(ModelConfig(timeout=7.0), api_key="sk-test")
    await routed.complete(MSG)
    assert factory[0].responses.create.await_args.kwargs["timeout"] == 7.0
    slow = OpenAIModel(ModelConfig(timeout=1.5), api_key="sk-test", pool=routed._pool)
    await slow.complete(MSG)
    assert len(factory) == 1
    assert factory[0].responses.create.await_args.kwargs["timeout"] == 1.5


@pytest.mark.asyncio
async def test_wait_for_timeout_still_enforced(factory):
    async def hang(**kwargs):
        await asyncio.sleep(5)
    model = OpenAIModel(ModelConfig(timeout=0.05), api_key="sk-test")
    model.warm()
    factory[0].responses.create = AsyncMock(side_effect=hang)
    from mimi_v2.models import ModelError
    with pytest.raises(ModelError, match="timeout"):
        await model.complete(MSG)


@pytest.mark.asyncio
async def test_warm_builds_client_once_before_first_call(factory):
    routed = RoutedModel(api_key="sk-test")
    routed.warm()
    assert len(factory) == 1
    await routed.complete(MSG)
    await routed.complete(MSG, purpose="safety")
    assert len(factory) == 1


def test_warm_without_credentials_is_silent(factory, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAIModel().warm()
    assert factory == []


@pytest.mark.asyncio
async def test_injected_client_untouched(factory):
    fake = MagicMock()
    fake.responses.create = AsyncMock(return_value=_response())
    fake.close = AsyncMock()
    await OpenAIModel(client=fake).complete(MSG)
    assert "timeout" not in fake.responses.create.await_args.kwargs
    assert factory == []
