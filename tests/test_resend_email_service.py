from __future__ import annotations

import pytest

import memory.supabase_client as supabase_client


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.content = b"{}" if payload is not None else b""

    def json(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_send_emergency_email_resend_success(monkeypatch):
    captured: dict = {}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            captured["endpoint"] = endpoint
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse(202, {"id": "email_123"}, "accepted")

    monkeypatch.setenv("RESEND_API_KEY", "re_test_123456789")
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Mimi Bot <noreply@example.com>")
    monkeypatch.setenv("EMERGENCY_EMAIL_RECIPIENTS", "teacher1@example.com,teacher2@example.com")
    monkeypatch.setenv("RESEND_API_BASE_URL", "https://api.resend.com")
    monkeypatch.setenv("RESEND_REQUEST_TIMEOUT", "12")
    monkeypatch.setattr(supabase_client.httpx, "AsyncClient", _FakeAsyncClient)

    sent = await supabase_client.send_emergency_email(
        user_id="u-test",
        summary="Emergency signal",
        raw_message="Em rat tuyet vong",
    )

    assert sent is True
    assert captured["endpoint"] == "https://api.resend.com/emails"
    assert captured["timeout"] == 12.0
    assert captured["headers"]["Authorization"] == "Bearer re_test_123456789"
    assert captured["json"]["from"] == "Mimi Bot <noreply@example.com>"
    assert captured["json"]["to"] == ["teacher1@example.com", "teacher2@example.com"]
    assert "KHAN CAP" in captured["json"]["subject"].upper() or "KHẨN CẤP" in captured["json"]["subject"]


@pytest.mark.asyncio
async def test_send_emergency_email_resend_missing_config(monkeypatch):
    was_called = {"post": False}

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            was_called["post"] = True
            return _FakeResponse(202, {"id": "email_123"}, "accepted")

    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Mimi Bot <noreply@example.com>")
    monkeypatch.setenv("EMERGENCY_EMAIL_RECIPIENTS", "teacher1@example.com")
    monkeypatch.setattr(supabase_client.httpx, "AsyncClient", _FakeAsyncClient)

    sent = await supabase_client.send_emergency_email(
        user_id="u-test",
        summary="Emergency signal",
        raw_message="help",
    )

    assert sent is False
    assert was_called["post"] is False


@pytest.mark.asyncio
async def test_send_emergency_email_resend_api_error(monkeypatch):
    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            return _FakeResponse(403, None, "forbidden")

    monkeypatch.setenv("RESEND_API_KEY", "re_test_123456789")
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Mimi Bot <noreply@example.com>")
    monkeypatch.setenv("EMERGENCY_EMAIL_RECIPIENTS", "teacher1@example.com")
    monkeypatch.setenv("RESEND_API_BASE_URL", "https://api.resend.com")
    monkeypatch.setattr(supabase_client.httpx, "AsyncClient", _FakeAsyncClient)

    sent = await supabase_client.send_emergency_email(
        user_id="u-test",
        summary="Emergency signal",
        raw_message="help",
    )

    assert sent is False


@pytest.mark.asyncio
async def test_send_emergency_email_resend_invalid_timeout(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_123456789")
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Mimi Bot <noreply@example.com>")
    monkeypatch.setenv("EMERGENCY_EMAIL_RECIPIENTS", "teacher1@example.com")
    monkeypatch.setenv("RESEND_REQUEST_TIMEOUT", "abc")

    sent = await supabase_client.send_emergency_email(
        user_id="u-test",
        summary="Emergency signal",
        raw_message="help",
    )

    assert sent is False


@pytest.mark.asyncio
async def test_send_emergency_email_resend_batch_fail_then_partial_success(monkeypatch):
    call_log: list[list[str]] = []

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, json, headers):
            recipients = json.get("to", [])
            call_log.append(list(recipients))
            if len(recipients) > 1:
                return _FakeResponse(400, None, "batch failed")
            if recipients[0] == "teacher1@example.com":
                return _FakeResponse(202, {"id": "ok-1"}, "accepted")
            return _FakeResponse(422, None, "invalid recipient")

    monkeypatch.setenv("RESEND_API_KEY", "re_test_123456789")
    monkeypatch.setenv("RESEND_FROM_EMAIL", "Mimi Bot <noreply@example.com>")
    monkeypatch.setenv("EMERGENCY_EMAIL_RECIPIENTS", "teacher1@example.com,teacher2@example.com")
    monkeypatch.setenv("RESEND_API_BASE_URL", "https://api.resend.com")
    monkeypatch.setattr(supabase_client.httpx, "AsyncClient", _FakeAsyncClient)

    sent = await supabase_client.send_emergency_email(
        user_id="u-test",
        summary="Emergency signal",
        raw_message="help",
    )

    assert sent is True
    assert call_log[0] == ["teacher1@example.com", "teacher2@example.com"]
    assert ["teacher1@example.com"] in call_log
    assert ["teacher2@example.com"] in call_log
