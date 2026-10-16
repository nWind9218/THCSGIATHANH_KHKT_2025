"""Teacher console data layer against dict-backed Redis only."""
import asyncio
import json

import pytest

from mimi_v2.console_log import ConsoleLog, EventBus, MemoryConsoleLog, get_console_log, get_event_bus
from mimi_v2.production import ProductionStore


def drain(queue):
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


@pytest.mark.asyncio
async def test_append_and_history_keep_order_and_shape():
    log = MemoryConsoleLog()
    first = await log.append("a", "student", "xin chào")
    await log.append("a", "mimi", "chào bạn", status="sent")
    await log.append("a", "teacher", "cô đây", author="co.lan", status="failed")
    assert set(first) == {"id", "ts", "role", "text", "author", "status"}
    assert first["author"] is None and first["status"] is None
    entries = await log.history("a")
    assert [e["role"] for e in entries] == ["student", "mimi", "teacher"]
    assert entries[2]["author"] == "co.lan" and entries[2]["status"] == "failed"
    assert [e["text"] for e in await log.history("a", limit=2)] == ["chào bạn", "cô đây"]
    assert await log.history("a", limit=0) == []
    assert await log.history("missing") == []
    assert len({e["id"] for e in entries}) == 3


@pytest.mark.asyncio
async def test_history_is_trimmed_to_max_entries_and_skips_corrupt_rows():
    log = MemoryConsoleLog(max_entries=3)
    for i in range(5):
        await log.append("a", "student", f"m{i}")
    assert [e["text"] for e in await log.history("a")] == ["m2", "m3", "m4"]
    log.memory.data["console:log:a"].insert(0, "{not json")
    assert [e["text"] for e in await log.history("a")] == ["m2", "m3", "m4"]


@pytest.mark.asyncio
async def test_append_writes_one_transaction_with_ttls_then_publishes():
    log = MemoryConsoleLog(ttl_seconds=100)
    queue = log.bus.subscribe()
    await log.append("a", "student", "hi")
    assert len(log.memory.batches) == 1
    transaction, commands = log.memory.batches[0]
    assert transaction is True
    assert commands.count("expire") == 3 and "rpush" in commands and "zadd" in commands
    assert log.memory.ttls == {"console:log:a": 100, "console:meta:a": 100, "console:conversations": 100}
    assert drain(queue) == [{"type": "message", "psid": "a"}]


@pytest.mark.asyncio
async def test_failed_write_does_not_publish():
    class Broken:
        def pipeline(self, transaction):
            raise ConnectionError("down")

    async def factory():
        return Broken()
    log = ConsoleLog(factory)
    queue = log.bus.subscribe()
    with pytest.raises(ConnectionError):
        await log.append("a", "student", "hi")
    assert queue.empty()


@pytest.mark.asyncio
async def test_meta_tracks_unread_preview_and_student_timestamp():
    log = MemoryConsoleLog()
    assert await log.get_meta("a") == {
        "psid": "a", "alias": None, "handoff_status": "none", "last_ts": None,
        "last_student_ts": None, "last_preview": None, "last_role": None, "unread": 0}
    await log.append("a", "student", "một")
    await log.append("a", "student", "hai")
    mimi = await log.append("a", "mimi", "trả lời")
    meta = await log.get_meta("a")
    assert meta["unread"] == 2 and meta["last_role"] == "mimi" and meta["last_preview"] == "trả lời"
    assert meta["last_ts"] == mimi["ts"] and meta["last_student_ts"] <= mimi["ts"]
    assert isinstance(meta["last_ts"], float) and isinstance(meta["unread"], int)
    await log.append("a", "student", "một\n  hai   " + "x" * 200)
    meta = await log.get_meta("a")
    assert meta["last_preview"].startswith("một hai x") and len(meta["last_preview"]) == 120
    assert meta["unread"] == 3
    await log.mark_read("a")
    assert (await log.get_meta("a"))["unread"] == 0
    await log.mark_read("never-seen")
    assert "console:meta:never-seen" not in log.memory.data


@pytest.mark.asyncio
async def test_teacher_and_system_entries_do_not_touch_student_fields():
    log = MemoryConsoleLog()
    await log.append("a", "student", "hi")
    before = await log.get_meta("a")
    await log.append("a", "system", "note")
    await log.append("a", "teacher", "reply", author="t")
    after = await log.get_meta("a")
    assert after["unread"] == 1 and after["last_student_ts"] == before["last_student_ts"]
    assert after["last_role"] == "teacher"


@pytest.mark.asyncio
async def test_list_conversations_newest_first_and_limited():
    log = MemoryConsoleLog()
    for psid in ("a", "b", "c"):
        await log.append(psid, "student", f"from {psid}")
        await asyncio.sleep(0.002)
    await log.append("a", "teacher", "latest", author="t")
    items = await log.list_conversations()
    assert [i["psid"] for i in items] == ["a", "c", "b"]
    assert items[0]["last_role"] == "teacher" and items[0]["handoff_status"] == "none"
    assert set(items[0]) == {"psid", "alias", "handoff_status", "last_ts", "last_student_ts",
                             "last_preview", "last_role", "unread"}
    assert [i["psid"] for i in await log.list_conversations(limit=2)] == ["a", "c"]
    assert await MemoryConsoleLog().list_conversations() == []


@pytest.mark.asyncio
async def test_list_conversations_skips_and_prunes_members_without_meta():
    log = MemoryConsoleLog()
    await log.append("a", "student", "hi")
    await asyncio.sleep(0.002)
    await log.append("b", "student", "yo")
    del log.memory.data["console:meta:a"]  # TTL hit the hash but not the index
    assert [i["psid"] for i in await log.list_conversations()] == ["b"]
    assert set(log.memory.data["console:conversations"]) == {"b"}


@pytest.mark.asyncio
async def test_append_prunes_index_entries_older_than_ttl():
    log = MemoryConsoleLog(ttl_seconds=100)
    log.memory.data["console:conversations"] = {"old": 1.0}
    await log.append("a", "student", "hi")
    assert set(log.memory.data["console:conversations"]) == {"a"}
    assert "zremrangebyscore" in log.memory.batches[0][1]


@pytest.mark.asyncio
async def test_set_status_publishes_only_on_change():
    log = MemoryConsoleLog()
    await log.append("a", "student", "hi")
    queue = log.bus.subscribe()
    await log.set_status("a", "none")
    assert queue.empty()
    await log.set_status("a", "pending")
    await log.set_status("a", "pending")
    await log.set_status("a", "accepted")
    assert drain(queue) == [{"type": "status", "psid": "a"}] * 2
    assert (await log.get_meta("a"))["handoff_status"] == "accepted"
    await log.set_status("a", "none")
    assert (await log.list_conversations())[0]["handoff_status"] == "none"
    with pytest.raises(ValueError):
        await log.set_status("a", "bogus")


@pytest.mark.asyncio
async def test_unknown_stored_status_reads_as_none():
    log = MemoryConsoleLog()
    await log.append("a", "student", "hi")
    log.memory.data["console:meta:a"]["handoff_status"] = "weird"
    assert (await log.get_meta("a"))["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_alias_is_stripped_trimmed_and_published():
    log = MemoryConsoleLog()
    queue = log.bus.subscribe()
    await log.set_alias("a", "  Lan 8A  ")
    assert (await log.get_meta("a"))["alias"] == "Lan 8A"
    await log.set_alias("a", "y" * 100)
    assert len((await log.get_meta("a"))["alias"]) == 60
    await log.set_alias("a", "   ")
    assert (await log.get_meta("a"))["alias"] is None
    assert drain(queue) == [{"type": "alias", "psid": "a"}] * 3


@pytest.mark.asyncio
async def test_event_bus_fans_out_unsubscribes_and_resyncs_on_overflow():
    bus = EventBus()
    fast, slow = bus.subscribe(), bus.subscribe()
    assert fast.maxsize == 100
    bus.publish({"type": "message", "psid": "a"})
    assert drain(fast) == [{"type": "message", "psid": "a"}]
    assert drain(slow) == [{"type": "message", "psid": "a"}]
    bus.unsubscribe(slow)
    bus.unsubscribe(slow)  # idempotent
    bus.publish({"type": "status", "psid": "b"})
    assert drain(fast) == [{"type": "status", "psid": "b"}] and slow.empty()
    for i in range(100):
        bus.publish({"type": "message", "psid": str(i)})
    assert fast.full()
    bus.publish({"type": "message", "psid": "overflow"})
    assert drain(fast) == [{"type": "resync"}]
    bus.publish({"type": "message", "psid": "after"})
    assert drain(fast) == [{"type": "message", "psid": "after"}]


def test_singletons_share_one_bus(monkeypatch):
    import mimi_v2.console_log as module
    monkeypatch.setattr(module, "_bus", None)
    monkeypatch.setattr(module, "_log", None)
    assert get_event_bus() is get_event_bus()
    assert get_console_log() is get_console_log()
    assert get_console_log().bus is get_event_bus()


@pytest.mark.asyncio
async def test_default_redis_comes_from_database_module(monkeypatch):
    import utils.database as database
    memory = MemoryConsoleLog().memory

    async def client():
        return memory
    monkeypatch.setattr(database, "get_redis_client", client)
    await ConsoleLog().append("a", "student", "hi")
    assert json.loads(memory.data["console:log:a"][0])["text"] == "hi"


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
        self.values, self.transactions, self.locks = dict(values or {}), [], {}

    async def mget(self, *keys):
        return [self.values.get(key) for key in keys]

    def pipeline(self, transaction):
        return FakePipeline(self)

    def lock(self, key, **options):
        return self.locks.setdefault(key, asyncio.Lock())


@pytest.mark.asyncio
@pytest.mark.parametrize("meta", [None, {"handoff_status": "none"}, {"handoff_status": "pending"}, {"handoff_status": "accepted"}])
async def test_takeover_accepts_from_any_state_and_is_idempotent(meta):
    redis = FakeRedis({"mimi:v2:session:a": json.dumps(meta)} if meta else {})

    async def factory():
        return redis
    store = ProductionStore(factory)
    for _ in range(2):
        await store.takeover("a")
        assert (await store.load("a"))["handoff_status"] == "accepted"
        assert redis.values["chat:a:takeover"] == "true"
        assert json.loads(redis.values["mimi:v2:session:a"])["handoff_status"] == "accepted"
    assert len(redis.transactions) == 2 and len(redis.transactions[-1]) == 4
    await store.resume("a")
    assert (await store.load("a"))["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_set_typing_publishes_a_flag_only_and_tracks_active_psids():
    log = MemoryConsoleLog()
    queue = log.bus.subscribe()
    log.set_typing("s1", True)
    log.set_typing("s2", True)
    log.set_typing("s1", False)
    assert drain(queue) == [
        {"type": "typing", "psid": "s1", "active": True}, {"type": "typing", "psid": "s2", "active": True},
        {"type": "typing", "psid": "s1", "active": False}]
    assert log.bus.typing == {"s2"}
    assert await log.list_conversations() == [] and log.memory.data == {}   # nothing is stored


@pytest.mark.asyncio
async def test_typing_event_overflow_still_resyncs():
    bus = EventBus(maxsize=1)
    queue = bus.subscribe()
    bus.publish({"type": "typing", "psid": "s1", "active": True})
    bus.publish({"type": "typing", "psid": "s1", "active": False})
    assert drain(queue) == [{"type": "resync"}]
