"""Webhook to teacher-console mirroring; Facebook, storage and Redis are all faked."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

import api.facebook_webhook as webhook
import mimi_v2.console_log as console_log
import mimi_v2.production as production


def engine_returning(text="", status="none"):
    store = SimpleNamespace(acknowledge=AsyncMock(), resume=AsyncMock())
    engine = SimpleNamespace(
        respond=AsyncMock(return_value={"response_text": text, "handoff_status": status}), store=store)
    return engine


@pytest.fixture
def env(monkeypatch):
    engine = engine_returning("Mimi nghe đây", "none")
    send = AsyncMock()
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    monkeypatch.setattr(webhook, "ADMIN_FACEBOOK_PSIDS", ["admin"])
    return SimpleNamespace(engine=engine, send=send, log=console_log.get_console_log())


def roles(entries):
    return [(e["role"], e["text"], e["status"]) for e in entries]


def test_public_helpers():
    assert webhook.send_messenger_text is webhook._send_messenger_message
    assert webhook.RESUME_MESSAGE == "Mimi đã quay trở lại rồi đây! Mình tiếp tục câu chuyện nhé?"


@pytest.mark.asyncio
async def test_student_message_is_logged_before_engine_and_reply_after(env):
    seen = {}

    async def respond(psid, text):
        seen["history"] = roles(await env.log.history(psid))
        return {"response_text": "Mimi nghe đây", "handoff_status": "none"}
    env.engine.respond = respond
    await webhook.handle_facebook_event("s1", "xin chào")
    assert seen["history"] == [("student", "xin chào", None)]
    assert roles(await env.log.history("s1")) == [
        ("student", "xin chào", None), ("mimi", "Mimi nghe đây", "sent")]
    env.send.assert_awaited_once_with("s1", "Mimi nghe đây")
    assert (await env.log.get_meta("s1"))["unread"] == 1


@pytest.mark.asyncio
async def test_handoff_status_is_mirrored_after_engine(env):
    env.engine.respond.return_value = {"response_text": "đã báo thầy cô", "handoff_status": "pending"}
    queue = env.log.bus.subscribe()
    await webhook.handle_facebook_event("s1", "em muốn gặp thầy cô")
    assert (await env.log.get_meta("s1"))["handoff_status"] == "pending"
    types = []
    while not queue.empty():
        types.append(queue.get_nowait()["type"])
    assert "status" in types
    env.engine.respond.return_value = {"response_text": "", "handoff_status": "accepted"}
    await webhook.handle_facebook_event("s1", "alo")
    meta = await env.log.get_meta("s1")
    assert meta["handoff_status"] == "accepted" and meta["unread"] == 2
    assert [e["role"] for e in await env.log.history("s1")] == ["student", "mimi", "student"]


@pytest.mark.asyncio
async def test_unknown_status_is_not_mirrored_and_empty_reply_not_logged(env):
    env.engine.respond.return_value = {"response_text": "", "handoff_status": "unknown"}
    await webhook.handle_facebook_event("s1", "hi")
    assert (await env.log.get_meta("s1"))["handoff_status"] == "none"
    assert [e["role"] for e in await env.log.history("s1")] == ["student"]
    env.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_delivery_is_logged_as_failed(env):
    env.send.side_effect = RuntimeError("Messenger delivery failed with status 400")
    await webhook.handle_facebook_event("s1", "hi")
    assert roles(await env.log.history("s1"))[-1] == ("mimi", "Mimi nghe đây", "failed")


@pytest.mark.asyncio
async def test_console_failure_never_blocks_bot_reply(env, monkeypatch, caplog):
    class Broken:
        def __getattr__(self, name):
            async def fail(*args, **kwargs):
                raise ConnectionError("redis down")
            return fail
    monkeypatch.setattr(console_log, "get_console_log", lambda: Broken())
    with caplog.at_level("WARNING"):
        await webhook.handle_facebook_event("s1", "hi")
    env.engine.respond.assert_awaited_once_with("s1", "hi")
    env.send.assert_awaited_once_with("s1", "Mimi nghe đây")
    assert "Teacher console append failed" in caplog.text
    assert "Failed to process Messenger event" not in caplog.text


@pytest.mark.asyncio
async def test_console_factory_failure_never_blocks_bot_reply(env, monkeypatch):
    def explode():
        raise RuntimeError("no redis configured")
    monkeypatch.setattr(console_log, "get_console_log", explode)
    await webhook.handle_facebook_event("s1", "hi")
    env.send.assert_awaited_once_with("s1", "Mimi nghe đây")


@pytest.mark.asyncio
async def test_probe_psid_is_never_logged(env):
    await webhook.handle_facebook_event(webhook.E2E_PROBE_PSID, "probe", suppress_delivery=True)
    env.engine.respond.assert_awaited_once()
    assert await env.log.list_conversations() == []
    await webhook.handle_facebook_event(webhook.E2E_PROBE_PSID, "probe")
    assert await env.log.list_conversations() == []


@pytest.mark.asyncio
async def test_admin_accept_mirrors_status_and_adds_system_entry(env):
    await webhook.handle_facebook_event("admin", "#accept student")
    env.engine.store.acknowledge.assert_awaited_once_with("student")
    meta = await env.log.get_meta("student")
    assert meta["handoff_status"] == "accepted"
    assert roles(await env.log.history("student")) == [("system", "Đã tiếp nhận qua Messenger", None)]
    assert await env.log.history("admin") == []
    env.engine.respond.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_accept_and_non_admin_leave_console_untouched(env):
    env.engine.store.acknowledge.side_effect = ValueError("not pending")
    await webhook.handle_facebook_event("admin", "#accept student")
    assert await env.log.history("student") == []
    await webhook.handle_facebook_event("admin", "#whoami")
    assert await env.log.list_conversations() == []


@pytest.mark.asyncio
async def test_admin_resume_mirrors_status_and_adds_system_entry(env):
    await env.log.append("student", "student", "hi")
    await env.log.set_status("student", "accepted")
    await webhook.handle_facebook_event("admin", "#resume student")
    env.engine.store.resume.assert_awaited_once_with("student")
    assert (await env.log.get_meta("student"))["handoff_status"] == "none"
    assert roles(await env.log.history("student"))[-1] == ("system", "Đã trả lại cho Mimi qua Messenger", None)
    assert env.send.await_args_list[-1].args == ("student", webhook.RESUME_MESSAGE)


@pytest.mark.asyncio
async def test_unauthorized_resume_is_treated_as_ordinary_student_text(env):
    await webhook.handle_facebook_event("parent", "#resume student")
    env.engine.store.resume.assert_not_awaited()
    assert roles(await env.log.history("parent"))[0] == ("student", "#resume student", None)


async def deliver(payload, monkeypatch=None):
    body = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}
    request = Request({"type": "http", "method": "POST", "path": "/webhook/facebook",
                       "headers": [(b"content-type", b"application/json")]}, receive)
    tasks = BackgroundTasks()
    await webhook.receive_webhook(request, tasks)
    return tasks.tasks


def echo(text="cô trả lời", app_id=None):
    message = {"is_echo": True, "text": text}
    if app_id is not None:
        message["app_id"] = app_id
    return {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "PAGE"}, "recipient": {"id": "student9"}, "message": message}]}]}


@pytest.mark.asyncio
async def test_echo_is_never_processed_as_student_message(monkeypatch):
    monkeypatch.delenv("FACEBOOK_APP_ID", raising=False)
    assert await deliver(echo(app_id=123)) == []
    monkeypatch.setenv("FACEBOOK_APP_ID", "111")
    assert await deliver(echo(app_id=111)) == []
    assert await deliver(echo(app_id=None)) == []
    normal = {"object": "page", "entry": [{"messaging": [{"sender": {"id": "s"}, "message": {"text": "hi"}}]}]}
    assert len(await deliver(normal)) == 1


@pytest.mark.asyncio
async def test_page_inbox_echo_is_logged_as_teacher_reply(monkeypatch):
    monkeypatch.setenv("FACEBOOK_APP_ID", "111")
    tasks = await deliver(echo("cô nhắn từ Inbox", app_id=263902037430900))
    assert len(tasks) == 1
    await tasks[0].func(*tasks[0].args, **tasks[0].kwargs)
    history = await console_log.get_console_log().history("student9")
    assert [(e["role"], e["author"], e["text"]) for e in history] == [
        ("teacher", "Facebook Page Inbox", "cô nhắn từ Inbox")]
    assert (await console_log.get_console_log().get_meta("student9"))["unread"] == 0


class _FakeModel:
    """Model stub: safety verdict + assessment JSON, any other purpose returns `reply`."""
    def __init__(self, reply, risk="safe"):
        self.reply, self.risk = reply, risk

    async def complete(self, messages, purpose="reply"):
        text = {
            "safety": json.dumps({"risk": self.risk, "reason": "fixture"}),
            "assessment": json.dumps(dict(
                known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                topic="school", core_issue="school", safety_hint="safe", user_role="unknown")),
        }.get(purpose, self.reply)
        return SimpleNamespace(text=text, usage={}, metadata={})


def real_engine(monkeypatch, env, reply, risk="safe"):
    from mimi_v2.core import CounselingEngine
    from mimi_v2.store import InMemoryStore
    engine = CounselingEngine(_FakeModel(reply, risk), InMemoryStore())
    engine.store = SimpleNamespace(acknowledge=AsyncMock(), resume=AsyncMock(), load=engine.store.load)
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    return engine


@pytest.mark.asyncio
async def test_first_turn_logs_greeting_and_reply_as_two_plain_text_mimi_entries(env, monkeypatch):
    from mimi_v2.prompts import GREETING_MESSAGE
    real_engine(monkeypatch, env, r"Ta có $x^2 - 5x + 6 = 0$ nên $x = \frac{5 \pm 1}{2}$ nha")
    await webhook.handle_facebook_event("s1", "giải giúp tớ phương trình")
    history = await env.log.history("s1")
    assert [(e["role"], e["status"]) for e in history] == [("student", None), ("mimi", "sent"), ("mimi", "sent")]
    assert history[1]["text"] == GREETING_MESSAGE
    reply = history[2]["text"]
    assert "$" not in reply and "\\frac" not in reply and "\\pm" not in reply
    # What the console shows is byte-for-byte what was handed to Messenger, in the same order.
    assert [c.args for c in env.send.await_args_list] == [("s1", e["text"]) for e in history[1:]]
    await webhook.handle_facebook_event("s1", "cảm ơn nha")
    assert [e["role"] for e in await env.log.history("s1")][3:] == ["student", "mimi"]  # no second greeting


@pytest.mark.asyncio
async def test_handoff_turn_logs_crisis_text_then_greeting_and_mirrors_status(env, monkeypatch):
    from mimi_v2.prompts import GREETING_MESSAGE, HANDOFF_MESSAGE
    real_engine(monkeypatch, env, "không dùng tới", risk="urgent")
    await webhook.handle_facebook_event("s1", "tớ muốn tự tử")
    history = await env.log.history("s1")
    assert [e["role"] for e in history] == ["student", "mimi", "mimi"]
    assert history[1]["text"].startswith(HANDOFF_MESSAGE) and history[2]["text"] == GREETING_MESSAGE
    assert (await env.log.get_meta("s1"))["handoff_status"] == "pending"


@pytest.mark.asyncio
async def test_each_part_is_logged_with_its_own_delivery_status(env):
    env.engine.respond.return_value = {"response_parts": ["một", "hai"], "response_text": "một\n\nhai",
                                       "handoff_status": "pending"}
    env.send.side_effect = [None, RuntimeError("Messenger delivery failed with status 400")]
    await webhook.handle_facebook_event("s1", "hi")
    assert roles(await env.log.history("s1")) == [
        ("student", "hi", None), ("mimi", "một", "sent"), ("mimi", "hai", "failed")]
    assert (await env.log.get_meta("s1"))["handoff_status"] == "pending"  # mirrored even though delivery failed


@pytest.mark.asyncio
async def test_probe_with_parts_skips_console_and_delivery(env):
    env.engine.respond.return_value = {"response_parts": ["a", "b"], "response_text": "a\n\nb"}
    await webhook.handle_facebook_event(webhook.E2E_PROBE_PSID, "hi", suppress_delivery=True)
    env.send.assert_not_awaited()
    assert await env.log.list_conversations() == []
