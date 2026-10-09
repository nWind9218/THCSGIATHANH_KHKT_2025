"""Stateless OpenAI adapter. Conversation ownership stays in the v2 core."""
from __future__ import annotations

import asyncio
import math
import os
import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class ModelError(Exception):
    """Public errors never contain provider bodies, prompts or credentials."""

    def __init__(self, code: str, status: int | None = None):
        self.code = code
        self.type = "model_error"
        self.status = status if isinstance(status, int) and 100 <= status <= 599 else None
        super().__init__(code)


def _capabilities(model: str) -> tuple[set[str], bool]:
    # Exact families also accept pinned snapshots; unknown models remain usable
    # with default parameters, but optional capabilities must be known.
    if "chat-latest" in model or "codex" in model:
        return set(), False
    if model == "gpt-5" or model.startswith(("gpt-5-",)):
        if model.startswith("gpt-5-pro"):
            return {"high"}, True
        return {"minimal", "low", "medium", "high"}, True
    if model.startswith("gpt-5.4-pro"):
        return {"medium", "high", "xhigh"}, True
    if model.startswith("gpt-5.2-pro"):
        return {"medium", "high", "xhigh"}, True
    if model.startswith("gpt-5.1"):
        return {"none", "low", "medium", "high"}, True
    if model.startswith(("gpt-5.2", "gpt-5.4")):
        return {"none", "low", "medium", "high", "xhigh"}, True
    if model.startswith(("o3", "o4-mini")):
        return {"low", "medium", "high"}, False
    return set(), False


@dataclass(frozen=True)
class ModelConfig:
    api: str = "responses"
    model: str = "gpt-5-mini"
    reasoning_effort: str | None = None
    verbosity: str | None = None
    max_output_tokens: int = 2048
    timeout: float = 60.0

    def __post_init__(self):
        if self.api not in {"responses", "chat_completions"}:
            raise ModelError("unsupported_api")
        if not isinstance(self.model, str) or not self.model.strip() or any(c.isspace() for c in self.model):
            raise ModelError("invalid_model")
        if self.api == "chat_completions" and self.model.startswith(("gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro")):
            raise ModelError("unsupported_model_api")
        efforts, verbosity = _capabilities(self.model)
        if self.reasoning_effort is not None and self.reasoning_effort not in efforts:
            raise ModelError("unsupported_reasoning_effort")
        if self.verbosity is not None and (not verbosity or self.verbosity not in {"low", "medium", "high"}):
            raise ModelError("unsupported_verbosity")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ModelError("invalid_token_limit")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ModelError("invalid_timeout")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ModelConfig":
        env = os.environ if env is None else env
        try:
            return cls(
                api=env.get("MIMI_V2_API", "responses"),
                model=env.get("MIMI_V2_MODEL", env.get("OPENAI_CHAT_MODEL", "gpt-5-mini")),
                reasoning_effort=env.get("MIMI_V2_REASONING_EFFORT") or None,
                verbosity=env.get("MIMI_V2_VERBOSITY") or None,
                max_output_tokens=int(env.get("MIMI_V2_MAX_OUTPUT_TOKENS", "2048")),
                timeout=float(env.get("MIMI_V2_TIMEOUT", "60")),
            )
        except (ValueError, TypeError):
            raise ModelError("invalid_configuration") from None

    def public_config(self) -> dict:
        return asdict(self)

    def as_dict(self) -> dict:
        return self.public_config()


@dataclass
class ModelReply:
    text: str
    usage: dict
    metadata: dict


def _dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump()
    raise ModelError("invalid_output")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ModelError("invalid_output")
    parts = []
    for block in content:
        block = _dict(block)
        if block.get("type") in {"refusal"}:
            raise ModelError("refusal")
        if block.get("type") not in {"text", "output_text"} or not isinstance(block.get("text"), str):
            raise ModelError("unsupported_output")
        parts.append(block["text"])
    return "".join(parts)


def _usage(value: Any) -> dict:
    if value is None:
        return {}
    source = _dict(value)
    # Only documented counters, not arbitrary provider metadata.
    counters = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"}
    details = {"input_tokens_details", "output_tokens_details", "prompt_tokens_details", "completion_tokens_details"}
    result = {}
    for key, item in source.items():
        if key in counters and type(item) is int and item >= 0:
            result[key] = item
        elif key in details and isinstance(item, dict):
            result[key] = _usage(item)
    return result


class OpenAIModel:
    def __init__(self, config: ModelConfig | None = None, api_key: str | None = None, client: Any = None,
                 usage_observer: Any = None):
        self.config = config or ModelConfig.from_env()
        self._api_key = api_key
        self._client = client
        self._usage_observer = usage_observer

    def _observe_usage(self, raw: dict) -> None:
        if self._usage_observer is None:
            return
        try:
            usage = _usage(raw.get("usage"))
            status = raw.get("status")
            if self.config.api == "chat_completions":
                choices = raw.get("choices")
                status = "completed" if isinstance(choices, list) and len(choices) == 1 and _dict(choices[0]).get("finish_reason") == "stop" else "incomplete"
            if status not in {"completed", "failed", "incomplete", "cancelled", "queued", "in_progress"}:
                status = "unknown"
            # Configured model is public; returned model can include a provider
            # snapshot suffix. Never forward arbitrary provider text fields.
            model = raw.get("model")
            if not isinstance(model, str) or not model.startswith(self.config.model) or len(model) > 128 or not all(c.isascii() and (c.isalnum() or c in "-._") for c in model):
                model = self.config.model
            self._usage_observer({"usage": usage, "model": model, "status": status,
                                  "config": self.config.public_config()})
        except Exception:
            # Observer outages must not turn a valid provider answer into an
            # invalid answer; collectors can mark this safe diagnostic incomplete.
            logging.getLogger(__name__).warning("model_usage_observer_failed")

    async def complete(self, messages: list[dict], purpose: str = "reply") -> ModelReply:
        if not isinstance(messages, list) or not messages:
            raise ModelError("invalid_input")
        clean = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant"} or not isinstance(message.get("content"), str):
                raise ModelError("unsupported_input")
            clean.append({"role": message["role"], "content": message["content"]})
        client = self._client
        owned = client is None
        try:
            if owned:
                from openai import AsyncOpenAI
                key = self._api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ModelError("missing_credential")
                # Explicit canonical endpoint: ambient proxy/base-url variables
                # cannot redirect student content or credentials elsewhere.
                client = AsyncOpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=self.config.timeout)
            kwargs = {"model": self.config.model, "store": False}
            if self.config.api == "responses":
                kwargs.update(input=clean, max_output_tokens=self.config.max_output_tokens)
                if self.config.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
                if self.config.verbosity:
                    kwargs["text"] = {"verbosity": self.config.verbosity}
                call = client.responses.create(**kwargs)
            else:
                kwargs.update(messages=clean, max_completion_tokens=self.config.max_output_tokens)
                if self.config.reasoning_effort:
                    kwargs["reasoning_effort"] = self.config.reasoning_effort
                if self.config.verbosity:
                    kwargs["verbosity"] = self.config.verbosity
                call = client.chat.completions.create(**kwargs)
            raw = _dict(await asyncio.wait_for(call, timeout=self.config.timeout))
            self._observe_usage(raw)
            if self.config.api == "responses":
                if raw.get("status") != "completed":
                    raise ModelError("incomplete_output")
                parts = []
                for output in raw.get("output", []):
                    output = _dict(output)
                    if output.get("type") == "reasoning":
                        continue
                    if output.get("type") != "message" or output.get("role") != "assistant" or output.get("status") not in {None, "completed"}:
                        raise ModelError("unsupported_output")
                    parts.append(_text(output.get("content")))
                text = "".join(parts)
            else:
                choices = raw.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ModelError("invalid_output")
                choice = _dict(choices[0])
                if choice.get("finish_reason") != "stop":
                    raise ModelError("incomplete_output")
                message = _dict(choice.get("message"))
                if message.get("refusal"):
                    raise ModelError("refusal")
                if message.get("tool_calls") or message.get("function_call") or message.get("audio"):
                    raise ModelError("unsupported_output")
                text = _text(message.get("content"))
            if not text.strip():
                raise ModelError("empty_output")
            return ModelReply(text.strip(), _usage(raw.get("usage")), {"config": self.config.public_config()})
        except ModelError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            raise ModelError("timeout") from None
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            code = "timeout" if type(exc).__name__ == "APITimeoutError" else "provider_error"
            raise ModelError(code, status=status) from None
        finally:
            if owned and client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
