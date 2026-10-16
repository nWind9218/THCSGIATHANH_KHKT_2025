"""Explicit production boundary; local tools must never instantiate this store."""
from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import wraps


IO_TIMEOUT = 10.0
SEARCH_TIMEOUT = 15.0
NOTIFY_TIMEOUT = 15.0
MODEL_TIMEOUT_LIMIT = 60.0
LOCK_LEASE = 300.0


def bounded(seconds):
    """Bound the entire operation, including dependency acquisition and retries."""
    def decorate(method):
        @wraps(method)
        async def wrapped(*args, **kwargs):
            async with asyncio.timeout(seconds):
                return await method(*args, **kwargs)
        return wrapped
    return decorate


class ProductionStore:
    """Read legacy Redis history without migration; keep v2 metadata separately."""

    def __init__(self, redis_factory=None, *, min_similarity: float = 0.35):
        if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)) or not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        self._redis_factory = redis_factory
        self.min_similarity = min_similarity

    async def _redis(self):
        if self._redis_factory is not None:
            return await self._redis_factory()
        from utils.database import get_redis_client
        return await get_redis_client()

    @asynccontextmanager
    async def conversation_lock(self, user_id: str):
        async with asyncio.timeout(IO_TIMEOUT):
            redis = await self._redis()
        # 3 calls≤180s + load/save≤20s + retrieval/notify≤30s + 70s headroom.
        # Model provider retries are disabled. Lock acquisition precedes lease start.
        async with redis.lock(f"mimi:v2:lock:{user_id}", timeout=LOCK_LEASE,
                              blocking_timeout=20, thread_local=False):
            yield

    @bounded(IO_TIMEOUT)
    async def load(self, user_id: str) -> dict:
        redis = await self._redis()
        raw_meta, raw_history, topic, takeover = await redis.mget(
            f"mimi:v2:session:{user_id}", f"chat:{user_id}:history",
            f"chat:{user_id}:topic", f"chat:{user_id}:takeover",
        )
        def parse(raw, expected, fallback):
            try:
                value = json.loads(raw) if raw is not None else deepcopy(fallback)
            except (ValueError, TypeError):
                raise ValueError("Invalid stored session data") from None
            if not isinstance(value, expected):
                raise ValueError("Invalid stored session data")
            return value
        session = parse(raw_meta, dict, {})
        session["messages"] = parse(raw_history, list, [])[-20:]
        if isinstance(topic, bytes):
            topic = topic.decode("utf-8")
        if isinstance(takeover, bytes):
            takeover = takeover.decode("utf-8")
        session.setdefault("current_topic", session.get("topic", topic or ""))
        session.setdefault("topic", session["current_topic"])
        session.setdefault("handoff_status", "none")
        if session["handoff_status"] not in {"none", "pending", "accepted"}:
            raise ValueError("Invalid stored handoff status")
        if takeover is not None and str(takeover).lower() not in {"true", "false"}:
            raise ValueError("Invalid stored takeover flag")
        if str(takeover).lower() == "true":
            session["handoff_status"] = "accepted"
        return session

    @bounded(IO_TIMEOUT)
    async def save(self, user_id: str, session: dict) -> None:
        await self._write(user_id, session)

    @bounded(IO_TIMEOUT)
    async def _write(self, user_id: str, session: dict, takeover: bool | None = None) -> None:
        redis = await self._redis()
        metadata = {key: deepcopy(value) for key, value in session.items() if key != "messages"}
        async with redis.pipeline(transaction=True) as pipe:
            pipe.set(f"chat:{user_id}:history",
                     json.dumps(session.get("messages", [])[-20:], ensure_ascii=False), ex=604800)
            pipe.set(f"chat:{user_id}:topic", session.get("topic", session.get("current_topic", "")), ex=604800)
            pipe.set(f"mimi:v2:session:{user_id}", json.dumps(metadata, ensure_ascii=False), ex=604800)
            if takeover is not None:
                pipe.set(f"chat:{user_id}:takeover", "true" if takeover else "false", ex=2592000)
            await pipe.execute()

    @bounded(SEARCH_TIMEOUT)
    async def search(self, user_id: str, query: str) -> dict:
        from utils.database import get_pg_connection, fetchall
        from utils.embeddings import embed_text, vector_literal
        await get_pg_connection()
        vector = vector_literal(await embed_text(query))
        queries = [
            ("psychology", "SELECT id, content, source_url, 1-(embedding <=> $1::vector) AS similarity FROM psychology_kb WHERE 1-(embedding <=> $1::vector) >= $2 ORDER BY embedding <=> $1::vector LIMIT 3", (vector, self.min_similarity)),
            ("student_knowledge", "SELECT id, user_intent, response_mode, should_do, should_not_do, sample_responses, 1-(embedding <=> $1::vector) AS similarity FROM student_knowledge WHERE 1-(embedding <=> $1::vector) >= $2 ORDER BY embedding <=> $1::vector LIMIT 3", (vector, self.min_similarity)),
            ("user_memory", "SELECT id, content, 1-(embedding <=> $1::vector) AS similarity FROM user_memory_kb WHERE user_id=$2 AND 1-(embedding <=> $1::vector) >= $3 ORDER BY embedding <=> $1::vector LIMIT 3", (vector, user_id, self.min_similarity)),
        ]
        values = await asyncio.gather(*(fetchall(sql, *args) for _, sql, args in queries), return_exceptions=True)
        result = {"psychology": "", "student_knowledge": "", "user_memory": "", "sources": [], "errors": []}
        for (kind, _, _), rows in zip(queries, values):
            if isinstance(rows, BaseException):
                result["errors"].append(f"{kind}_unavailable")
                continue
            chunks = []
            for record in rows:
                row = dict(record)
                if kind == "student_knowledge":
                    chunks.append("\n".join(f"{field}: {row.get(field, '')}" for field in
                        ("user_intent", "response_mode", "should_do", "should_not_do", "sample_responses")))
                else:
                    chunks.append(row.get("content") or "")
                result["sources"].append({"kind": kind, "id": str(row.get("id", "")),
                    "similarity": float(row.get("similarity", 0)), "source_url": row.get("source_url")})
            result[kind] = "\n\n".join(chunks)
        return result

    @bounded(NOTIFY_TIMEOUT)
    async def notify(self, user_id: str, text: str) -> bool:
        # Independent attempts: a log failure must not prevent the email.
        from memory import log_emergency, send_emergency_email
        summary = f"Mimi v2 pending human review: {user_id}"
        outcomes = await asyncio.gather(
            log_emergency(user_id, summary),
            send_emergency_email(user_id, summary, text),
            return_exceptions=True,
        )
        # Delivery means email API accepted a request, never a teacher acknowledgement.
        return outcomes[1] is True

    async def acknowledge(self, user_id: str) -> None:
        async with self.conversation_lock(user_id):
            session = await self.load(user_id)
            if session.get("handoff_status") != "pending":
                raise ValueError("Only a pending handoff can be accepted")
            session["handoff_status"] = "accepted"
            await self._write(user_id, session, takeover=True)

    async def resume(self, user_id: str) -> None:
        async with self.conversation_lock(user_id):
            session = await self.load(user_id)
            session["handoff_status"] = "none"
            session["notification_sent"] = False
            await self._write(user_id, session, takeover=False)


_engine = None


def get_production_engine():
    global _engine
    if _engine is None:
        from .core import CounselingEngine
        from .models import OpenAIModel, ModelConfig, ModelError
        config = ModelConfig.from_env()
        if config.timeout > MODEL_TIMEOUT_LIMIT:
            raise ModelError("production_timeout_exceeds_lease_budget")
        _engine = CounselingEngine(OpenAIModel(config), ProductionStore())
    return _engine
