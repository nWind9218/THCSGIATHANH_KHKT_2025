"""Provider-independent verification; never loads .env or opens a network."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mimi_v2.models import ModelConfig, ModelError, OpenAIModel


def client(result=None, error=None):
    create = AsyncMock(return_value=result, side_effect=error)
    return SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ), create


def response(text="Chào bạn.", **overrides):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": text}]},
    ], "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}, **overrides}


def chat_response(content="Chào bạn.", finish_reason="stop", **fields):
    return {"choices": [{"finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content, **fields}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}}


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["responses", "chat_completions"])
async def test_stateless_payload_and_normalization(api):
    fake, create = client(response() if api == "responses" else chat_response())
    config = ModelConfig(api=api, reasoning_effort="minimal", verbosity="low")
    messages = [{"role": "system", "content": "Help students."}, {"role": "user", "content": "Chào."}]
    reply = await OpenAIModel(config, client=fake).complete(messages, purpose="reply")
    assert reply.text == "Chào bạn."
    assert reply.usage["total_tokens"] == 9
    assert reply.metadata["config"] == config.public_config()
    payload = create.call_args.kwargs
    assert payload["store"] is False
    assert "temperature" not in payload and "previous_response_id" not in payload
    assert payload["input" if api == "responses" else "messages"] == messages
    if api == "responses":
        assert payload["reasoning"] == {"effort": "minimal"}
        assert payload["text"] == {"verbosity": "low"}
        assert payload["max_output_tokens"] == 2048
    else:
        assert payload["reasoning_effort"] == "minimal"
        assert payload["verbosity"] == "low"
        assert payload["max_completion_tokens"] == 2048


@pytest.mark.asyncio
async def test_chat_blocks_usage_allowlist():
    raw = chat_response([{"type": "text", "text": "Xin "}, {"type": "text", "text": "chào"}])
    raw["usage"].update(secret="sk-secret", completion_tokens_details={"reasoning_tokens": 3, "secret": "sk-secret"})
    fake, _ = client(raw)
    reply = await OpenAIModel(ModelConfig(api="chat_completions"), client=fake).complete([{ "role": "user", "content": "Hello"}])
    assert reply.text == "Xin chào"
    assert reply.usage["completion_tokens_details"] == {"reasoning_tokens": 3}
    assert "sk-secret" not in repr(reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    (response(status="incomplete"), "incomplete_output"),
    (response(" "), "empty_output"),
    (response(output=[]), "empty_output"),
    (response(output=[{"type": "function_call", "arguments": "secret"}]), "unsupported_output"),
    (response(output=[{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "secret"}]}]), "refusal"),
])
async def test_responses_reject_unsafe_or_incomplete_output(raw, expected):
    fake, _ = client(raw)
    with pytest.raises(ModelError) as error:
        await OpenAIModel(client=fake).complete([{ "role": "user", "content": "Hello"}])
    assert error.value.code == expected
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    (chat_response(finish_reason="length"), "incomplete_output"),
    (chat_response(refusal="secret"), "refusal"),
    (chat_response(tool_calls=[{"name": "secret"}]), "unsupported_output"),
    (chat_response(None), "invalid_output"),
])
async def test_chat_rejects_non_text(raw, expected):
    fake, _ = client(raw)
    with pytest.raises(ModelError) as error:
        await OpenAIModel(ModelConfig(api="chat_completions"), client=fake).complete([{ "role": "user", "content": "Hello"}])
    assert error.value.code == expected


@pytest.mark.asyncio
async def test_timeout_no_retry():
    async def slow(**kwargs):
        await asyncio.sleep(1)
    fake, create = client()
    create.side_effect = slow
    with pytest.raises(ModelError, match="timeout"):
        await OpenAIModel(ModelConfig(timeout=0.001), client=fake).complete([{ "role": "user", "content": "Hello"}])
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_provider_error_does_not_leak_credentials_or_payload():
    failure = RuntimeError("sk-secret confidential conversation provider body")
    failure.status_code = 429
    fake, create = client(error=failure)
    with pytest.raises(ModelError) as error:
        await OpenAIModel(client=fake).complete([{ "role": "user", "content": "Hello"}])
    assert error.value.code == "rate_limit"
    assert error.value.status == 429
    assert str(error.value) == "rate_limit"
    assert error.value.__suppress_context__
    assert create.call_count == 1


@pytest.mark.parametrize("kwargs", [
    {"api": "bad"}, {"reasoning_effort": "none"}, {"model": "gpt-4.1-mini", "verbosity": "low"},
    {"model": "gpt-4.1-mini", "reasoning_effort": "low"}, {"model": "unknown", "reasoning_effort": "low"},
    {"timeout": float("nan")}, {"timeout": 0}, {"timeout": True}, {"max_output_tokens": 0},
    {"max_output_tokens": True}, {"model": ""},
    {"api": "chat_completions", "model": "gpt-5.4-pro"},
    {"model": "gpt-5.4-pro", "reasoning_effort": "none"},
    {"model": "gpt-5-chat-latest", "reasoning_effort": "low"},
])
def test_unsupported_configuration(kwargs):
    with pytest.raises(ModelError):
        ModelConfig(**kwargs)


def test_env_config_no_secrets_and_fallback():
    config = ModelConfig.from_env({"OPENAI_CHAT_MODEL": "gpt-4.1-mini", "OPENAI_API_KEY": "sk-secret", "MIMI_V2_API": "chat_completions", "MIMI_V2_TIMEOUT": "12", "MIMI_V2_MAX_OUTPUT_TOKENS": "100"})
    assert config.model == "gpt-4.1-mini" and config.timeout == 12
    assert config.max_output_tokens == 100
    assert "sk-secret" not in repr(config.public_config())
    assert ModelConfig.from_env({"OPENAI_CHAT_MODEL": "old", "MIMI_V2_MODEL": "gpt-5-mini"}).model == "gpt-5-mini"
    with pytest.raises(ModelError, match="invalid_configuration"):
        ModelConfig.from_env({"MIMI_V2_TIMEOUT": "sk-secret"})


@pytest.mark.asyncio
async def test_unsupported_input_not_sent():
    fake, create = client(response())
    with pytest.raises(ModelError, match="unsupported_input"):
        await OpenAIModel(client=fake).complete([{ "role": "tool", "content": "secret"}])
    create.assert_not_called()


@pytest.mark.asyncio
async def test_sdk_dump_supported_and_client_close_errors_redacted(monkeypatch):
    import openai

    class Result:
        def model_dump(self):
            return response()

    fake, create = client(Result())
    fake.close = AsyncMock(side_effect=RuntimeError("sk-secret"))
    factory = lambda **kwargs: fake
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    reply = await OpenAIModel(api_key="fake-test-key").complete([{ "role": "user", "content": "Hello"}])
    assert reply.text == "Chào bạn."
    fake.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_key_no_request(monkeypatch):
    for name in ("OPENAI_API_KEY", "MIMI_V2_BASE_URL", "MIMI_V2_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelError, match="missing_credential"):
        await OpenAIModel().complete([{ "role": "user", "content": "Hello"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("api,raw,code", [
    ("responses", response(), None),
    ("responses", response(status="incomplete"), "incomplete_output"),
    ("responses", response(output=[{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "sk-secret"}]}]), "refusal"),
    ("chat_completions", chat_response(), None),
    ("chat_completions", chat_response(finish_reason="length"), "incomplete_output"),
    ("chat_completions", chat_response(refusal="sk-secret"), "refusal"),
])
async def test_observer_meters_returned_usage_before_output_validation_once(api, raw, code):
    observations = []
    raw.update(model="gpt-5-mini-2025-08-07", id="sk-secret", prompt="private")
    raw["usage"]["secret"] = "sk-secret"
    fake, _ = client(raw)
    model = OpenAIModel(ModelConfig(api=api), client=fake, usage_observer=observations.append)
    if code:
        with pytest.raises(ModelError, match=code):
            await model.complete([{ "role": "user", "content": "private"}], purpose="assessment")
    else:
        await model.complete([{ "role": "user", "content": "private"}], purpose="assessment")
    assert len(observations) == 1
    assert observations[0]["usage"]["total_tokens"] == 9
    assert observations[0]["model"] == "gpt-5-mini-2025-08-07"
    assert set(observations[0]) == {"usage", "model", "status", "config"}
    assert "sk-secret" not in repr(observations) and "private" not in repr(observations)


@pytest.mark.asyncio
async def test_observer_failure_diagnostic_without_changing_answer(caplog):
    def broken(observation):
        raise RuntimeError("sk-secret private")
    fake, _ = client(response())
    answer = await OpenAIModel(client=fake, usage_observer=broken).complete([{ "role": "user", "content": "Hello"}])
    assert answer.text == "Chào bạn."
    assert "model_usage_observer_failed" in caplog.text
    assert "sk-secret" not in caplog.text and "private" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["responses", "chat_completions"])
@pytest.mark.parametrize("purpose", ["safety", "assessment"])
async def test_strict_schema_payload_keeps_verbosity(api, purpose):
    from mimi_v2.contracts import schema_for_purpose
    fake, create = client(response() if api == "responses" else chat_response())
    await OpenAIModel(ModelConfig(api=api, verbosity="low"), client=fake).complete(
        [{"role": "user", "content": "synthetic"}], purpose=purpose)
    kwargs = create.call_args.kwargs
    schema = schema_for_purpose(purpose)
    assert schema["strict"] is True
    if api == "responses":
        assert kwargs["text"]["verbosity"] == "low"
        assert kwargs["text"]["format"] == {"type": "json_schema", **schema}
    else:
        assert kwargs["verbosity"] == "low"
        assert kwargs["response_format"] == {"type": "json_schema", "json_schema": schema}


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["responses", "chat_completions"])
@pytest.mark.parametrize("purpose", ["legacy_safety", "legacy_assessment", "legacy_info_gap", "reply"])
async def test_legacy_baseline_purposes_have_no_provider_schema(api, purpose):
    fake, create = client(response() if api == "responses" else chat_response())
    await OpenAIModel(ModelConfig(api=api), client=fake).complete(
        [{"role": "user", "content": "synthetic"}], purpose=purpose)
    kwargs = create.call_args.kwargs
    assert "response_format" not in kwargs
    assert "format" not in kwargs.get("text", {})


@pytest.mark.asyncio
async def test_unknown_structured_capability_never_silently_downgrades():
    fake, create = client(response())
    with pytest.raises(ModelError, match="unsupported_structured_output"):
        await OpenAIModel(ModelConfig(model="unknown"), client=fake).complete(
            [{"role": "user", "content": "synthetic"}], purpose="assessment")
    create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_code,status,expected", [
    ("insufficient_quota", 429, "insufficient_quota"),
    ("rate_limit_exceeded", 429, "rate_limit"),
    (None, 429, "rate_limit"),
    ("invalid_api_key", 401, "authentication"),
    ("sk-secret private", 400, "provider_error"),
    (None, 500, "provider_error"),
])
async def test_provider_error_code_allowlist(provider_code, status, expected):
    failure = RuntimeError("sk-secret private provider body")
    failure.code = provider_code
    failure.status_code = status
    fake, _ = client(error=failure)
    with pytest.raises(ModelError) as error:
        await OpenAIModel(client=fake).complete([{ "role": "user", "content": "Hello"}])
    assert error.value.code == expected
    assert error.value.status == status
    assert "sk-secret" not in repr(error.value)


def test_endpoint_defaults_to_openai_and_ignores_ambient_base_url():
    from mimi_v2.models import OPENAI_BASE_URL, resolve_endpoint
    env = {"OPENAI_API_KEY": "sk-a", "OPENAI_BASE_URL": "https://evil.example/v1"}
    assert resolve_endpoint(env) == (OPENAI_BASE_URL, "sk-a")


def test_custom_endpoint_uses_only_its_own_key_and_allowlist():
    from mimi_v2.models import resolve_endpoint
    env = {"OPENAI_API_KEY": "sk-openai", "MIMI_V2_BASE_URL": "https://api.shineshop.dev/v1/"}
    assert resolve_endpoint(env) == ("https://api.shineshop.dev/v1", None)
    env["MIMI_V2_API_KEY"] = "shine-key"
    assert resolve_endpoint(env) == ("https://api.shineshop.dev/v1", "shine-key")
    with pytest.raises(ModelError, match="unapproved_base_url"):
        resolve_endpoint({"MIMI_V2_BASE_URL": "https://evil.example/v1"})


@pytest.mark.asyncio
async def test_custom_endpoint_client_gets_shineshop_url_and_key(monkeypatch):
    import openai
    seen = {}
    fake, _ = client(response())
    fake.close = AsyncMock()
    def factory(**kwargs):
        seen.update(kwargs)
        return fake
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    monkeypatch.delenv("MIMI_V2_BASE_URL", raising=False)
    await OpenAIModel(api_key="shine-key", base_url="https://api.shineshop.dev/v1").complete([{"role": "user", "content": "Hello"}])
    assert seen["base_url"] == "https://api.shineshop.dev/v1" and seen["api_key"] == "shine-key"
