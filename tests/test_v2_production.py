"""Production adapter contracts exercised against fake Redis only."""
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from mimi_v2.production import ProductionStore


class FakePipeline:
    def __init__(self, redis):
        self.redis, self.commands = redis, []
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    def set(self, key, value, **kwargs):
        self.commands.append((key, value, kwargs))
        return self
    async def execute(self):
        self.redis.transactions.append(list(self.commands))
        for key, value, _ in self.commands:
            self.redis.values[key] = value


class FakeRedis:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.transactions = []
        self.locks = {}
        self.lock_options = []
    async def mget(self, *keys):
        return [self.values.get(key) for key in keys]
    def pipeline(self, transaction):
        assert transaction is True
        return FakePipeline(self)
    def lock(self, key, **options):
        self.lock_options.append(options)
        return self.locks.setdefault(key, asyncio.Lock())


def store_for(redis):
    async def factory():
        return redis
    return ProductionStore(factory)


@pytest.mark.asyncio
async def test_legacy_history_topic_and_bytes_takeover_are_preserved():
    history = [{"role": "user", "content": "cũ"}, {"role": "assistant", "content": "phản hồi"}]
    redis = FakeRedis({"chat:a:history": json.dumps(history), "chat:a:topic": b"school", "chat:a:takeover": b"true"})
    store = store_for(redis)
    result = await store.load("a")
    assert result["messages"] == history
    assert result["topic"] == result["current_topic"] == "school"
    assert result["handoff_status"] == "accepted"
    result["messages"].clear()
    assert (await store.load("a"))["messages"] == history
    assert redis.transactions == []


@pytest.mark.asyncio
async def test_pending_delivery_is_not_acceptance_and_resume_is_atomic():
    redis = FakeRedis()
    store = store_for(redis)
    session = dict(messages=[{"role": "user", "content": "help"}], topic="new topic", known={"grade": "9"}, handoff_status="pending", notification_sent=True)
    await store.save("a", session)
    assert (await store.load("a"))["handoff_status"] == "pending"
    assert "chat:a:takeover" not in redis.values
    await store.acknowledge("a")
    assert (await store.load("a"))["handoff_status"] == "accepted"
    assert redis.values["chat:a:takeover"] == "true"
    assert len(redis.transactions[-1]) == 4
    await store.resume("a")
    result = await store.load("a")
    assert result["handoff_status"] == "none"
    assert result["notification_sent"] is False
    assert result["messages"] == session["messages"]
    assert result["known"] == {"grade": "9"}
    assert redis.values["chat:a:takeover"] == "false"
    assert len(redis.transactions[-1]) == 4
    assert all(option["thread_local"] is False for option in redis.lock_options)


@pytest.mark.asyncio
@pytest.mark.parametrize("meta", [None, {"handoff_status": "none"}, {"handoff_status": "accepted"}])
async def test_acknowledge_rejects_nonpending_without_writes(meta):
    redis = FakeRedis({"mimi:v2:session:a": json.dumps(meta)} if meta else {})
    with pytest.raises(ValueError, match="pending"):
        await store_for(redis).acknowledge("a")
    assert redis.transactions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [
    ("chat:a:history", "broken"), ("chat:a:history", "{}"),
    ("mimi:v2:session:a", "[]"), ("mimi:v2:session:a", ""),
    ("chat:a:takeover", "unknown"),
    ("mimi:v2:session:a", '{"handoff_status":"garbage"}'),
])
async def test_corrupt_storage_fails_read_instead_of_reset(key, value):
    redis = FakeRedis({key: value})
    with pytest.raises(ValueError):
        await store_for(redis).load("a")
    assert redis.values[key] == value
    assert redis.transactions == []


@pytest.mark.asyncio
async def test_save_window_metadata_snapshot_and_topic_alias():
    redis = FakeRedis()
    store = store_for(redis)
    session = {"messages": [{"role": "user", "content": str(i)} for i in range(30)], "current_topic": "legacy", "known": {"grade": "8"}}
    await store.save("a", session)
    assert len(json.loads(redis.values["chat:a:history"])) == 20
    metadata = json.loads(redis.values["mimi:v2:session:a"])
    assert "messages" not in metadata
    assert metadata["known"] == {"grade": "8"}
    session["known"]["grade"] = "9"
    assert json.loads(redis.values["mimi:v2:session:a"])["known"]["grade"] == "8"
    assert redis.values["chat:a:topic"] == "legacy"
    assert all(cmd[2]["ex"] == 604800 for cmd in redis.transactions[0])


@pytest.mark.parametrize("threshold", [True, False, "0.5", None, float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_similarity_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        ProductionStore(min_similarity=threshold)


@pytest.mark.parametrize("threshold", [0, 0.35, 1])
def test_valid_similarity_threshold(threshold):
    assert ProductionStore(min_similarity=threshold).min_similarity == threshold


@pytest.mark.asyncio
async def test_operation_deadline_includes_dependency_acquisition():
    from mimi_v2.production import bounded
    cancelled = asyncio.Event()
    @bounded(0.01)
    async def blocked_dependency():
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.set()
    with pytest.raises(TimeoutError):
        await blocked_dependency()
    assert cancelled.is_set()


def test_production_model_timeout_cannot_exceed_distributed_lease_budget(monkeypatch):
    from mimi_v2 import production
    from mimi_v2.models import ModelError
    monkeypatch.setattr(production, "_engine", None)
    monkeypatch.setenv("MIMI_V2_TIMEOUT", "61")
    with pytest.raises(ModelError, match="production_timeout_exceeds_lease_budget"):
        production.get_production_engine()
    assert production._engine is None


@pytest.mark.asyncio
async def test_real_load_deadline_cancels_stalled_redis_factory_without_write(monkeypatch):
    from mimi_v2 import production
    original_timeout = asyncio.timeout
    monkeypatch.setattr(production.asyncio, "timeout", lambda seconds: original_timeout(min(seconds, 0.01)))
    cancelled = asyncio.Event()
    async def factory():
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.set()
    with pytest.raises(TimeoutError):
        await ProductionStore(factory).load("a")
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_notify_independent_failures_and_delivery_is_not_human_acceptance(monkeypatch):
    attempts = []
    async def log(*args):
        attempts.append("log")
        raise RuntimeError("log unavailable")
    async def email(*args):
        attempts.append("email")
        return True
    monkeypatch.setitem(sys.modules, "memory", SimpleNamespace(log_emergency=log, send_emergency_email=email))
    redis = FakeRedis()
    assert await store_for(redis).notify("a", "test") is True
    assert set(attempts) == {"log", "email"}
    assert redis.transactions == []


@pytest.mark.asyncio
async def test_search_deadline_includes_embedding_and_never_reaches_sql(monkeypatch):
    from mimi_v2 import production
    original_timeout = asyncio.timeout
    monkeypatch.setattr(production.asyncio, "timeout", lambda seconds: original_timeout(min(seconds, 0.01)))
    cancelled = asyncio.Event()
    async def get_connection():
        return None
    async def embed(_text):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.set()
    async def fetch(*args):
        raise AssertionError("SQL must not run when embedding times out")
    monkeypatch.setitem(sys.modules, "utils.database", SimpleNamespace(get_pg_connection=get_connection, fetchall=fetch))
    monkeypatch.setitem(sys.modules, "utils.embeddings", SimpleNamespace(embed_text=embed, vector_literal=lambda value: value))
    with pytest.raises(TimeoutError):
        await ProductionStore().search("a", "test")
    assert cancelled.is_set()
