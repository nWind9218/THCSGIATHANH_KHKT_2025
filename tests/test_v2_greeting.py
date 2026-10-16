"""One-time AI disclosure: ordering, persistence, history hygiene and Messenger delivery. Offline."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import api.facebook_webhook as webhook
import mimi_v2.production as production
from mimi_v2.core import CounselingEngine
from mimi_v2.prompts import CLOSING_MESSAGE, GREETING_MESSAGE, HANDOFF_MESSAGE, SESSION_LOAD_FAILURE_MESSAGE
from mimi_v2.store import InMemoryStore

REPLY = "Mimi hiểu nè."


def assessment(**changes):
    result = dict(known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                  topic="school", core_issue="school", safety_hint="safe", user_role="unknown")
    result.update(changes)
    return result


class Model:
    def __init__(self, risk="safe", **changes):
        self.risk, self.changes, self.calls = risk, changes, []

    async def complete(self, messages, purpose="reply"):
        self.calls.append(purpose)
        text = {"safety": json.dumps({"risk": self.risk, "reason": "fixture"}),
                "assessment": json.dumps(assessment(**self.changes))}.get(purpose, REPLY)
        return SimpleNamespace(text=text, usage={}, metadata={})


@pytest.mark.asyncio
async def test_first_turn_greets_then_replies_and_second_turn_does_not():
    store = InMemoryStore()
    engine = CounselingEngine(Model(), store)
    first = await engine.respond("u", "chào Mimi")
    assert first["response_parts"] == [GREETING_MESSAGE, REPLY]
    assert first["response"] == first["response_text"] == GREETING_MESSAGE + "\n\n" + REPLY
    assert (await store.load("u"))["greeted"] is True
    second = await engine.respond("u", "tớ mệt quá")
    assert second["response_parts"] == [REPLY] and second["response"] == REPLY


@pytest.mark.asyncio
async def test_greeting_never_enters_llm_history():
    store = InMemoryStore()
    engine = CounselingEngine(Model(), store)
    result = await engine.respond("u", "chào Mimi")
    assert [(m["role"], m["content"]) for m in result["messages"]] == [("user", "chào Mimi"), ("assistant", REPLY)]
    stored = await store.load("u")
    assert all(GREETING_MESSAGE not in m["content"] for m in stored["messages"])


@pytest.mark.asyncio
async def test_urgent_first_turn_puts_crisis_text_before_the_greeting():
    store = InMemoryStore()
    engine = CounselingEngine(Model(risk="urgent"), store)
    result = await engine.respond("u", "tớ muốn tự tử")
    assert len(result["response_parts"]) == 2
    assert result["response_parts"][0].startswith(HANDOFF_MESSAGE) and result["response_parts"][1] == GREETING_MESSAGE
    assert result["messages"][-1]["content"] == result["response_parts"][0]
    assert (await store.load("u"))["greeted"] is True
    # Already greeted: the next handoff-pending turn is a plain single message.
    again = await engine.respond("u", "tớ vẫn ở đây")
    assert GREETING_MESSAGE not in again["response_parts"]


@pytest.mark.asyncio
async def test_closing_reply_is_greeted_on_first_contact():
    result = await CounselingEngine(Model(intent="closing"), InMemoryStore()).respond("u", "tạm biệt nha")
    assert result["response_parts"] == [GREETING_MESSAGE, CLOSING_MESSAGE]


@pytest.mark.asyncio
async def test_accepted_handoff_turns_stay_silent_and_do_not_greet():
    store = InMemoryStore()
    await store.save("u", {"handoff_status": "accepted", "messages": []})
    model = Model()
    engine = CounselingEngine(model, store)
    result = await engine.respond("u", "thầy cô ơi")
    assert result["response_parts"] == [] and result["response"] == result["response_text"] == ""
    assert (await store.load("u"))["greeted"] is False and model.calls == []
    # After the takeover ends, the first bot reply carries the greeting.
    await store.resume("u")
    resumed = await engine.respond("u", "mình quay lại rồi")
    assert resumed["response_parts"] == [GREETING_MESSAGE, REPLY]


@pytest.mark.asyncio
async def test_session_saved_before_the_feature_is_greeted_once():
    store = InMemoryStore()
    await store.save("u", {"known": {}, "messages": [{"role": "user", "content": "cũ"}, {"role": "assistant", "content": "cũ"}]})
    engine = CounselingEngine(Model(), store)
    assert (await engine.respond("u", "tin mới"))["response_parts"][0] == GREETING_MESSAGE
    assert GREETING_MESSAGE not in (await engine.respond("u", "tin nữa"))["response_parts"]


@pytest.mark.asyncio
async def test_session_load_failure_returns_only_the_fallback():
    class Broken(InMemoryStore):
        async def load(self, *args):
            raise RuntimeError("down")
    result = await CounselingEngine(Model(), Broken()).respond("u", "hello")
    assert result["response_parts"] == [SESSION_LOAD_FAILURE_MESSAGE] and result["response"] == SESSION_LOAD_FAILURE_MESSAGE
    assert "111" in SESSION_LOAD_FAILURE_MESSAGE


@pytest.mark.asyncio
async def test_webhook_sends_each_part_as_its_own_message_in_order(monkeypatch):
    engine = CounselingEngine(Model(), InMemoryStore())
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    send = AsyncMock()
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    await webhook.handle_facebook_event("psid", "chào Mimi")
    assert [c.args for c in send.await_args_list] == [("psid", GREETING_MESSAGE), ("psid", REPLY)]
    send.reset_mock()
    await webhook.handle_facebook_event("psid", "tiếp nè")
    assert [c.args for c in send.await_args_list] == [("psid", REPLY)]


@pytest.mark.asyncio
async def test_webhook_crisis_text_is_sent_before_greeting(monkeypatch):
    engine = CounselingEngine(Model(risk="urgent"), InMemoryStore())
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    send = AsyncMock()
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    await webhook.handle_facebook_event("psid", "tớ muốn tự tử")
    texts = [c.args[1] for c in send.await_args_list]
    assert len(texts) == 2 and texts[0].startswith(HANDOFF_MESSAGE) and texts[1] == GREETING_MESSAGE


@pytest.mark.asyncio
async def test_webhook_probe_suppresses_every_part_and_accepted_sends_nothing(monkeypatch):
    engine = SimpleNamespace(respond=AsyncMock(return_value={"response_parts": ["a", "b"], "response_text": "a\n\nb"}))
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    send = AsyncMock()
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    await webhook.handle_facebook_event("probe", "hi", suppress_delivery=True)
    send.assert_not_awaited()
    engine.respond.return_value = {"response_parts": [], "response_text": "", "handoff_status": "accepted"}
    await webhook.handle_facebook_event("psid", "hi")
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_falls_back_to_response_text_without_parts(monkeypatch):
    engine = SimpleNamespace(respond=AsyncMock(return_value={"response_text": " solo "}))
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    assert await webhook._process_message("psid", "hi") == ["solo"]


def test_gradio_display_shows_greeting_and_keeps_it_on_later_turns():
    from mimi_v2.gradio_prod import display_messages
    first = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": REPLY}],
             "response_parts": [GREETING_MESSAGE, REPLY]}
    shown = display_messages(first)
    assert [m["content"] for m in shown] == ["hi", GREETING_MESSAGE, REPLY]
    second = {"messages": first["messages"] + [{"role": "user", "content": "b"}, {"role": "assistant", "content": REPLY}],
              "response_parts": [REPLY]}
    assert [m["content"] for m in display_messages(second, shown)] == ["hi", GREETING_MESSAGE, REPLY, "b", REPLY]
    crisis = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "H"}], "response_parts": ["H", GREETING_MESSAGE]}
    assert [m["content"] for m in display_messages(crisis)] == ["x", "H", GREETING_MESSAGE]
