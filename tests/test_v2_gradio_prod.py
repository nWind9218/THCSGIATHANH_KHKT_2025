"""Gradio production wiring checks; no network, Redis or database is opened."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mimi_v2 import gradio_prod
from mimi_v2.gradio_prod import TEST_PREFIX, make_user_id


def test_session_ids_always_carry_test_prefix():
    assert make_user_id().startswith(TEST_PREFIX)
    assert make_user_id("Toán Lớp 8!").startswith(TEST_PREFIX)
    # Attempts to impersonate a real PSID are neutralised, not honoured.
    assert make_user_id("../1234567890123456") == TEST_PREFIX + "1234567890123456"
    assert len(make_user_id("x" * 500)) <= len(TEST_PREFIX) + 40


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["1234567890", "", TEST_PREFIX, "admin", None])
async def test_real_or_empty_ids_are_refused_everywhere(bad):
    engine = SimpleNamespace(respond=AsyncMock(), store=SimpleNamespace(resume=AsyncMock(), acknowledge=AsyncMock(), _redis=AsyncMock()))
    for call in (lambda: gradio_prod.send(engine, bad, "hi"), lambda: gradio_prod.clear_handoff(engine, bad),
                 lambda: gradio_prod.accept_handoff(engine, bad), lambda: gradio_prod.delete_session(engine.store, bad)):
        with pytest.raises(ValueError):
            await call()
    engine.respond.assert_not_awaited()
    engine.store.resume.assert_not_awaited()
    engine.store.acknowledge.assert_not_awaited()
    engine.store._redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_handoff_resumes_and_reloads_session():
    store = SimpleNamespace(resume=AsyncMock(), load=AsyncMock(return_value={"handoff_status": "none", "messages": []}))
    uid = make_user_id("a")
    session = await gradio_prod.clear_handoff(SimpleNamespace(store=store), uid)
    store.resume.assert_awaited_once_with(uid)
    assert session["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_delete_removes_only_this_sessions_keys():
    redis = SimpleNamespace(delete=AsyncMock())
    store = SimpleNamespace(_redis=AsyncMock(return_value=redis))
    uid = make_user_id("abc")
    await gradio_prod.delete_session(store, uid)
    keys = redis.delete.await_args.args
    assert keys and all(uid in key for key in keys)


def test_app_builds_without_connecting(monkeypatch):
    # Building the UI must not open any connection (pools start on first use).
    import utils.database as database
    monkeypatch.setattr(database, "start_pooling", lambda: pytest.fail("must not connect while building"))
    assert gradio_prod.build_app() is not None
