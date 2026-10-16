"""Webhook routing checks; no Facebook or storage connections are opened."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import api.facebook_webhook as webhook
import mimi_v2.production as production


@pytest.mark.asyncio
async def test_webhook_calls_shared_v2_engine(monkeypatch):
    engine = SimpleNamespace(respond=AsyncMock(return_value={"response_text": "v2"}))
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    assert await webhook._process_message("local", "hello") == "v2"
    engine.respond.assert_awaited_once_with("local", "hello")


@pytest.mark.asyncio
async def test_accepted_takeover_stays_silent(monkeypatch):
    engine = SimpleNamespace(respond=AsyncMock(return_value={"response_text": "", "handoff_status": "accepted"}))
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    assert await webhook._process_message("local", "hello") == ""


@pytest.mark.asyncio
async def test_accept_only_authorized_admin(monkeypatch):
    monkeypatch.setattr(webhook, "ADMIN_FACEBOOK_PSIDS", ["admin"])
    acknowledge = AsyncMock()
    engine = SimpleNamespace(store=SimpleNamespace(acknowledge=acknowledge))
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    process = AsyncMock(return_value="ordinary")
    send = AsyncMock()
    monkeypatch.setattr(webhook, "_process_message", process)
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    await webhook.handle_facebook_event("parent", "#accept student")
    acknowledge.assert_not_awaited()
    process.assert_awaited_once_with("parent", "#accept student")
    await webhook.handle_facebook_event("admin", "#accept student")
    acknowledge.assert_awaited_once_with("student")


@pytest.mark.asyncio
async def test_accept_rejection_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(webhook, "ADMIN_FACEBOOK_PSIDS", ["admin"])
    engine = SimpleNamespace(store=SimpleNamespace(acknowledge=AsyncMock(side_effect=ValueError("not pending"))))
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    send = AsyncMock()
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    await webhook.handle_facebook_event("admin", "#accept student")
    assert "chưa có" in send.await_args.args[1]
