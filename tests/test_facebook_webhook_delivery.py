import hashlib
import hmac
import json

import httpx
import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

import api.facebook_webhook as webhook


@pytest.mark.asyncio
async def test_send_message_keeps_access_token_out_of_url(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            request = httpx.Request("POST", url)
            return httpx.Response(200, request=request, json={"message_id": "ok"})

    monkeypatch.setattr(webhook, "FACEBOOK_PAGE_ACCESS_TOKEN", "secret-page-token")
    monkeypatch.setattr(webhook.httpx, "AsyncClient", FakeClient)

    await webhook._send_messenger_message("123", "hello")

    assert "access_token" not in captured["url"]
    assert captured["headers"] == {"Authorization": "Bearer secret-page-token"}
    assert "params" not in captured


@pytest.mark.asyncio
async def test_delivery_error_does_not_expose_access_token(monkeypatch, caplog):
    token = "secret-page-token"

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **_kwargs):
            request = httpx.Request("POST", url)
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "Invalid recipient",
                        "type": "OAuthException",
                        "code": 100,
                    }
                },
            )

    monkeypatch.setattr(webhook, "FACEBOOK_PAGE_ACCESS_TOKEN", token)
    monkeypatch.setattr(webhook.httpx, "AsyncClient", FakeClient)

    with pytest.raises(RuntimeError, match="status 400") as error:
        await webhook._send_messenger_message("invalid", "hello")

    assert token not in str(error.value)
    assert token not in caplog.text


@pytest.mark.asyncio
async def test_probe_suppresses_messenger_delivery(monkeypatch):
    app_secret = "test-app-secret"
    monkeypatch.setattr(webhook, "FACEBOOK_APP_SECRET", app_secret)
    payload = {
        "object": "page",
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": webhook.E2E_PROBE_PSID},
                        "message": {"text": "probe"},
                    }
                ]
            }
        ],
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    headers = [
        (b"content-type", b"application/json"),
        (b"x-hub-signature-256", f"sha256={signature}".encode()),
        (webhook.E2E_PROBE_HEADER.encode(), b"1"),
    ]
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/webhook/facebook",
            "headers": headers,
        },
        receive,
    )
    background_tasks = BackgroundTasks()

    result = await webhook.receive_webhook(request, background_tasks)

    assert result == {"status": "ok"}
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].args == (
        webhook.E2E_PROBE_PSID,
        "probe",
        True,
    )
