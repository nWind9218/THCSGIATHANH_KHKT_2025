from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from graph.workflow import build_counseling_graph


class _Resp:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs

    async def ainvoke(self, _messages):
        if not self.outputs:
            raise RuntimeError("FakeLLM exhausted")
        return _Resp(self.outputs.pop(0))


@pytest.mark.asyncio
async def test_emergency_flow(monkeypatch):
    import graph.nodes as nodes

    fake_llm = FakeLLM(["YES"])

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(nodes, "load_takeover_flag", lambda _user_id: _false())
    monkeypatch.setattr(nodes, "load_history", lambda _user_id: _empty_history())
    monkeypatch.setattr(nodes, "load_topic", lambda _user_id: _empty_text())
    monkeypatch.setattr(nodes, "search_user_memory_kb", lambda **_kwargs: _empty_text())
    monkeypatch.setattr(nodes, "notify_human_admin", _noop)
    monkeypatch.setattr(nodes, "set_takeover_flag", _noop)

    graph = build_counseling_graph()
    result = await graph.ainvoke(
        {
            "user_id": "u1",
            "messages": [{"role": "user", "content": "Em muon ket thuc tat ca"}],
        },
        config={"configurable": {"thread_id": "u1"}},
    )

    assert result["human_takeover"] is True
    assert "1800 599 920" in result["response_text"]


@pytest.mark.asyncio
async def test_simple_flow(monkeypatch):
    import graph.nodes as nodes

    fake_llm = FakeLLM(
        [
            "NO",
            json.dumps({"intent": "simple", "topic": "mood"}),
            "Minh o day voi ban nhe.",
            json.dumps({"should_store": False, "memory": ""}),
        ]
    )

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(nodes, "load_takeover_flag", lambda _user_id: _false())
    monkeypatch.setattr(nodes, "load_history", lambda _user_id: _empty_history())
    monkeypatch.setattr(nodes, "load_topic", lambda _user_id: _empty_text())
    monkeypatch.setattr(nodes, "search_user_memory_kb", lambda **_kwargs: _empty_text())
    monkeypatch.setattr(nodes, "save_history", _noop)
    monkeypatch.setattr(nodes, "save_topic", _noop)
    monkeypatch.setattr(nodes, "update_user_longterm_style", _noop)

    graph = build_counseling_graph()
    result = await graph.ainvoke(
        {
            "user_id": "u2",
            "messages": [{"role": "user", "content": "Hom nay em hoi met"}],
        },
        config={"configurable": {"thread_id": "u2"}},
    )

    assert result["response_text"] == "Minh o day voi ban nhe."
    assert result["intent_category"] == "simple"


@pytest.mark.asyncio
async def test_complex_missing_info_flow(monkeypatch):
    import graph.nodes as nodes

    fake_llm = FakeLLM(
        [
            "NO",
            json.dumps({"intent": "complex", "topic": "ap luc gia dinh"}),
            json.dumps({"scratchpad": ["chua biet nguyen nhan"], "verdict": "missing"}),
            "Ban ke them giup minh, chuyen nay lien quan den ai nhe?",
            json.dumps({"should_store": False, "memory": ""}),
        ]
    )

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(nodes, "load_takeover_flag", lambda _user_id: _false())
    monkeypatch.setattr(nodes, "load_history", lambda _user_id: _empty_history())
    monkeypatch.setattr(nodes, "load_topic", lambda _user_id: _empty_text())
    monkeypatch.setattr(nodes, "search_user_memory_kb", lambda **_kwargs: _empty_text())
    monkeypatch.setattr(nodes, "save_history", _noop)
    monkeypatch.setattr(nodes, "save_topic", _noop)
    monkeypatch.setattr(nodes, "update_user_longterm_style", _noop)

    graph = build_counseling_graph()
    result = await graph.ainvoke(
        {
            "user_id": "u3",
            "messages": [{"role": "user", "content": "Em buon va khong muon di hoc"}],
        },
        config={"configurable": {"thread_id": "u3"}},
    )

    assert result["intent_category"] == "complex"
    assert result["info_gap_status"] == "missing"
    assert "Ban ke them" in result["response_text"]


@pytest.mark.asyncio
async def test_complex_sufficient_flow(monkeypatch):
    import graph.nodes as nodes

    fake_llm = FakeLLM(
        [
            "NO",
            json.dumps({"intent": "complex", "topic": "co don"}),
            json.dumps({"scratchpad": ["du boi canh"], "verdict": "sufficient"}),
            "Phan hoi tong hop theo guideline.",
            json.dumps({"should_store": False, "memory": ""}),
        ]
    )

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)
    monkeypatch.setattr(nodes, "load_takeover_flag", lambda _user_id: _false())
    monkeypatch.setattr(nodes, "load_history", lambda _user_id: _empty_history())
    monkeypatch.setattr(nodes, "load_topic", lambda _user_id: _empty_text())
    monkeypatch.setattr(nodes, "search_user_memory_kb", lambda **_kwargs: _empty_text())
    monkeypatch.setattr(nodes, "search_psychology_kb", lambda *_args, **_kwargs: _kb_text())
    monkeypatch.setattr(nodes, "save_history", _noop)
    monkeypatch.setattr(nodes, "save_topic", _noop)
    monkeypatch.setattr(nodes, "update_user_longterm_style", _noop)

    graph = build_counseling_graph()
    result = await graph.ainvoke(
        {
            "user_id": "u4",
            "messages": [{"role": "user", "content": "Em thay co don vi mau thuan ban be"}],
        },
        config={"configurable": {"thread_id": "u4"}},
    )

    assert result["intent_category"] == "complex"
    assert result["info_gap_status"] == "sufficient"
    assert result["kb_guidelines"] == "mock-kb"
    assert result["response_text"] == "Phan hoi tong hop theo guideline."


def test_websocket_pubsub_e2e_short(monkeypatch):
    from api.chat_ws import router as chat_router
    from api.admin_ws import router as admin_router
    import api.chat_ws as chat_ws
    import api.admin_ws as admin_ws
    import memory.redis_client as redis_memory

    pubsub_redis = _FakeRedisPubSub()

    class _Graph:
        async def ainvoke(self, state):
            user_msg = state["messages"][-1]["content"]
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "da nhan"}],
                "response_text": f"Echo: {user_msg}",
                "intent_category": "simple",
                "is_emergency": False,
                "human_takeover": False,
                "current_topic": "test",
            }

    async def _fake_history(_user_id):
        return []

    async def _fake_topic(_user_id):
        return ""

    async def _fake_get_redis_client():
        return pubsub_redis

    monkeypatch.setattr(chat_ws, "build_counseling_graph", lambda: _Graph())
    monkeypatch.setattr(chat_ws, "load_history", _fake_history)
    monkeypatch.setattr(chat_ws, "load_topic", _fake_topic)
    monkeypatch.setattr(admin_ws, "get_redis_client", _fake_get_redis_client)
    monkeypatch.setattr(redis_memory, "get_redis_client", _fake_get_redis_client)

    app = FastAPI()
    app.include_router(chat_router, prefix="/ws")
    app.include_router(admin_router, prefix="/ws/admin")

    user_id = "student-e2e"

    with TestClient(app) as client:
        with client.websocket_connect("/ws/admin/admin/admin-1") as admin_socket:
            admin_ack = admin_socket.receive_json()
            assert admin_ack["type"] == "connection_acknowledged"

            admin_socket.send_json({"command": "subscribe", "user_id": user_id})
            subscribe_result = admin_socket.receive_json()
            assert subscribe_result["type"] == "command_result"
            assert subscribe_result["command"] == "subscribe"

            with client.websocket_connect(f"/ws/chat/{user_id}") as chat_socket:
                chat_ack = chat_socket.receive_json()
                assert chat_ack["type"] == "connection_acknowledged"

                chat_socket.send_json({"message": "xin chao"})
                bot_response = chat_socket.receive_json()
                assert bot_response["type"] == "bot_response"
                assert bot_response["response"] == "Echo: xin chao"

            # End-to-end check: chat endpoint published to Redis, admin endpoint received via Pub/Sub.
            admin_alert = admin_socket.receive_json()
            # Current admin_ws payload shape lets alert_data.type override envelope type.
            assert admin_alert["type"] == "chat_message"
            assert admin_alert["alert_type"] == "chat_message"
            assert admin_alert["user_id"] == user_id
            assert admin_alert["user_input"] == "xin chao"
            assert admin_alert["bot_response"] == "Echo: xin chao"


async def _false():
    return False


async def _empty_history():
    return []


async def _empty_text():
    return ""


async def _kb_text():
    return "mock-kb"


class _FakeRedisPubSub:
    def __init__(self):
        self._channels: dict[str, list[asyncio.Queue]] = {}

    def pubsub(self):
        return _FakePubSub(self)

    async def publish(self, channel: str, payload: str):
        queues = self._channels.get(channel, [])
        for queue in queues:
            await queue.put({"type": "message", "data": payload})

    def _register(self, channel: str, queue: asyncio.Queue):
        self._channels.setdefault(channel, []).append(queue)

    def _unregister(self, channel: str, queue: asyncio.Queue):
        if channel not in self._channels:
            return
        self._channels[channel] = [q for q in self._channels[channel] if q is not queue]
        if not self._channels[channel]:
            del self._channels[channel]


class _FakePubSub:
    def __init__(self, broker: _FakeRedisPubSub):
        self._broker = broker
        self._queue: asyncio.Queue = asyncio.Queue()
        self._subscribed: set[str] = set()

    async def subscribe(self, channel: str):
        if channel in self._subscribed:
            return
        self._subscribed.add(channel)
        self._broker._register(channel, self._queue)

    async def unsubscribe(self, channel: str):
        if channel not in self._subscribed:
            return
        self._subscribed.remove(channel)
        self._broker._unregister(channel, self._queue)

    async def close(self):
        for channel in list(self._subscribed):
            await self.unsubscribe(channel)

    async def listen(self):
        while True:
            message = await self._queue.get()
            yield message
