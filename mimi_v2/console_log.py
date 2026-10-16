"""Teacher console data layer: per-student message log, conversation index and change events."""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from typing import Literal

from .production import IO_TIMEOUT, bounded

Role = Literal["student", "mimi", "teacher", "system"]
HandoffStatus = Literal["none", "pending", "accepted"]

LOG_KEY = "console:log:{}"
META_KEY = "console:meta:{}"
INDEX_KEY = "console:conversations"
STATUSES = ("none", "pending", "accepted")
PREVIEW_CHARS = 120
ALIAS_CHARS = 60


class EventBus:
    """In-process fan-out (one uvicorn process today). Never blocks publishers."""

    def __init__(self, maxsize: int = 100):
        self._maxsize = maxsize
        self._queues: set[asyncio.Queue] = set()
        self.typing: set[str] = set()      # psids Mimi is composing a reply for (for clients that just connected)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._queues.discard(queue)

    def publish(self, event: dict) -> None:
        if event.get("type") == "typing" and event.get("psid"):
            (self.typing.add if event.get("active") else self.typing.discard)(event["psid"])
        for queue in list(self._queues):
            try:
                queue.put_nowait(dict(event))
            except asyncio.QueueFull:
                # A slow client lost events: drop its backlog and tell it to refetch everything.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync"})


def _number(value, cast=float):
    try:
        return cast(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class ConsoleLog:
    """Redis-backed log. Events carry no data; readers refetch through the read methods."""

    def __init__(self, redis_factory=None, *, bus: EventBus | None = None,
                 max_entries: int = 500, ttl_seconds: int = 2_592_000):
        self._redis_factory = redis_factory
        self.bus = bus if bus is not None else EventBus()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

    async def _redis(self):
        if self._redis_factory is not None:
            return await self._redis_factory()
        from utils.database import get_redis_client
        return await get_redis_client()

    @staticmethod
    def _meta(psid: str, raw: dict | None) -> dict:
        raw = raw or {}
        status = raw.get("handoff_status")
        return {
            "psid": psid, "alias": raw.get("alias") or None,
            "handoff_status": status if status in STATUSES else "none",
            "last_ts": _number(raw.get("last_ts")), "last_student_ts": _number(raw.get("last_student_ts")),
            "last_preview": raw.get("last_preview") or None, "last_role": raw.get("last_role") or None,
            "unread": _number(raw.get("unread"), int) or 0,
        }

    @bounded(IO_TIMEOUT)
    async def append(self, psid: str, role: Role, text: str, *, author: str | None = None,
                     status: str | None = None) -> dict:
        now = time.time()
        entry = {"id": f"{int(now * 1000)}-{secrets.token_hex(3)}", "ts": now, "role": role,
                 "text": text, "author": author, "status": status}
        meta = {"last_ts": now, "last_role": role, "last_preview": " ".join(text.split())[:PREVIEW_CHARS]}
        if role == "student":
            meta["last_student_ts"] = now
        log_key, meta_key = LOG_KEY.format(psid), META_KEY.format(psid)
        redis = await self._redis()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.rpush(log_key, json.dumps(entry, ensure_ascii=False))
            pipe.ltrim(log_key, -self.max_entries, -1)
            pipe.expire(log_key, self.ttl_seconds)
            pipe.hset(meta_key, mapping=meta)
            if role == "student":
                pipe.hincrby(meta_key, "unread", 1)
            pipe.expire(meta_key, self.ttl_seconds)
            pipe.zadd(INDEX_KEY, {psid: now})
            pipe.zremrangebyscore(INDEX_KEY, "-inf", now - self.ttl_seconds)  # meta của các mục này đã hết hạn
            pipe.expire(INDEX_KEY, self.ttl_seconds)
            await pipe.execute()
        self.bus.publish({"type": "message", "psid": psid})
        return entry

    def set_typing(self, psid: str, active: bool) -> None:
        """Ephemeral 'Mimi is replying' flag: bus only, nothing stored, no text."""
        self.bus.publish({"type": "typing", "psid": psid, "active": bool(active)})

    @bounded(IO_TIMEOUT)
    async def history(self, psid: str, limit: int = 200) -> list[dict]:
        if limit <= 0:
            return []
        redis = await self._redis()
        entries = []
        for raw in await redis.lrange(LOG_KEY.format(psid), -limit, -1):
            try:
                entry = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    @bounded(IO_TIMEOUT)
    async def list_conversations(self, limit: int = 50) -> list[dict]:
        redis = await self._redis()
        psids = await redis.zrevrange(INDEX_KEY, 0, max(limit, 1) - 1)
        if not psids:
            return []
        async with redis.pipeline(transaction=False) as pipe:
            for psid in psids:
                pipe.hgetall(META_KEY.format(psid))
            metas = await pipe.execute()
        expired = [psid for psid, meta in zip(psids, metas) if not meta]
        if expired:
            try:
                await redis.zrem(INDEX_KEY, *expired)   # dọn mục mồ côi, lỗi cũng bỏ qua
            except Exception:
                pass
        return [self._meta(psid, meta) for psid, meta in zip(psids, metas) if meta]

    @bounded(IO_TIMEOUT)
    async def get_meta(self, psid: str) -> dict:
        redis = await self._redis()
        return self._meta(psid, await redis.hgetall(META_KEY.format(psid)))

    @bounded(IO_TIMEOUT)
    async def set_status(self, psid: str, status: HandoffStatus) -> None:
        if status not in STATUSES:
            raise ValueError("Invalid handoff status")
        redis, key = await self._redis(), META_KEY.format(psid)
        if (await redis.hget(key, "handoff_status") or "none") == status:
            return
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, "handoff_status", status)
            pipe.expire(key, self.ttl_seconds)
            await pipe.execute()
        self.bus.publish({"type": "status", "psid": psid})

    @bounded(IO_TIMEOUT)
    async def set_alias(self, psid: str, alias: str) -> None:
        redis, key = await self._redis(), META_KEY.format(psid)
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, "alias", alias.strip()[:ALIAS_CHARS])
            pipe.expire(key, self.ttl_seconds)
            await pipe.execute()
        self.bus.publish({"type": "alias", "psid": psid})

    @bounded(IO_TIMEOUT)
    async def mark_read(self, psid: str) -> None:
        redis, key = await self._redis(), META_KEY.format(psid)
        if await redis.exists(key):
            await redis.hset(key, "unread", 0)


class _MemoryPipeline:
    def __init__(self, redis: "_MemoryRedis", transaction: bool):
        self.redis, self.transaction, self.commands = redis, transaction, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def __getattr__(self, name: str):
        def queue(*args, **kwargs):
            self.commands.append((name, args, kwargs))
            return self
        return queue

    async def execute(self) -> list:
        self.redis.batches.append((self.transaction, [name for name, _, _ in self.commands]))
        return [getattr(self.redis, f"_{name}")(*args, **kwargs) for name, args, kwargs in self.commands]


class _MemoryRedis:
    """The few Redis commands ConsoleLog uses, over dicts (decode_responses=True semantics)."""

    def __init__(self):
        self.data: dict[str, object] = {}
        self.ttls: dict[str, int] = {}
        self.batches: list[tuple[bool, list[str]]] = []

    def pipeline(self, transaction: bool = True) -> _MemoryPipeline:
        return _MemoryPipeline(self, transaction)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(self, f"_{name}")

        async def call(*args, **kwargs):
            return method(*args, **kwargs)
        return call

    def _rpush(self, key, *values):
        items = self.data.setdefault(key, [])
        items.extend(str(v) for v in values)
        return len(items)

    def _ltrim(self, key, start, stop):
        self.data[key] = self.data.get(key, [])[start:None if stop == -1 else stop + 1]
        return True

    def _lrange(self, key, start, stop):
        return list(self.data.get(key, [])[start:None if stop == -1 else stop + 1])

    def _expire(self, key, seconds):
        if key in self.data:
            self.ttls[key] = seconds
        return key in self.data

    def _exists(self, key):
        return int(key in self.data)

    def _hset(self, key, field=None, value=None, mapping=None):
        pairs = dict(mapping or {})
        if field is not None:
            pairs[field] = value
        self.data.setdefault(key, {}).update({k: str(v) for k, v in pairs.items()})
        return len(pairs)

    def _hget(self, key, field):
        return self.data.get(key, {}).get(field)

    def _hgetall(self, key):
        return dict(self.data.get(key, {}))

    def _hincrby(self, key, field, amount=1):
        fields = self.data.setdefault(key, {})
        fields[field] = str(int(fields.get(field, 0)) + amount)
        return int(fields[field])

    def _zadd(self, key, mapping):
        self.data.setdefault(key, {}).update({k: float(v) for k, v in mapping.items()})
        return len(mapping)

    def _zrem(self, key, *members):
        index = self.data.get(key, {})
        return sum(index.pop(m, None) is not None for m in members)

    def _zremrangebyscore(self, key, low, high):
        index = self.data.get(key, {})
        lo = float("-inf") if low == "-inf" else float(low)
        hi = float("inf") if high == "+inf" else float(high)
        gone = [m for m, score in index.items() if lo <= score <= hi]
        for m in gone:
            del index[m]
        return len(gone)

    def _zrevrange(self, key, start, stop):
        ranked = sorted(self.data.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return [member for member, _ in ranked[start:None if stop == -1 else stop + 1]]


class MemoryConsoleLog(ConsoleLog):
    """Dict-backed twin with identical behaviour, for tests and scripts/console_demo.py."""

    def __init__(self, *, bus: EventBus | None = None, max_entries: int = 500, ttl_seconds: int = 2_592_000):
        self.memory = _MemoryRedis()

        async def factory():
            return self.memory
        super().__init__(factory, bus=bus, max_entries=max_entries, ttl_seconds=ttl_seconds)


_bus: EventBus | None = None
_log: ConsoleLog | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def get_console_log() -> ConsoleLog:
    global _log
    if _log is None:
        _log = ConsoleLog(bus=get_event_bus())
    return _log
