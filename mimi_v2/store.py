"""Dependency-free, isolated session store for local simulation and evaluation."""
from __future__ import annotations

from copy import deepcopy
import asyncio
from typing import Protocol


class SessionStore(Protocol):
    async def load(self, user_id: str) -> dict: ...
    async def save(self, user_id: str, session: dict) -> None: ...
    async def search(self, user_id: str, query: str) -> dict: ...
    async def notify(self, user_id: str, text: str) -> bool: ...


class InMemoryStore:
    """No network or production imports. Not a production persistence backend."""

    def __init__(self, knowledge: dict | None = None):
        self.sessions: dict[str, dict] = {}
        self.knowledge = deepcopy(knowledge or {})
        self.alerts: list[dict] = []
        self._locks: dict[str, asyncio.Lock] = {}

    def session_lock(self, user_id: str) -> asyncio.Lock:
        return self._locks.setdefault(user_id, asyncio.Lock())

    async def load(self, user_id: str) -> dict:
        return deepcopy(self.sessions.get(user_id, {}))

    async def save(self, user_id: str, session: dict) -> None:
        self.sessions[user_id] = deepcopy(session)

    async def search(self, user_id: str, query: str) -> dict:
        return {key: str(self.knowledge.get(key, "")) for key in
                ("psychology", "student_knowledge", "user_memory")}

    async def notify(self, user_id: str, text: str) -> bool:
        self.alerts.append({"user_id": user_id, "text": text})
        return True

    async def reset(self, user_id: str) -> None:
        async with self.session_lock(user_id):
            self.sessions.pop(user_id, None)

    async def acknowledge(self, user_id: str) -> None:
        async with self.session_lock(user_id):
            session = await self.load(user_id)
            if session.get("handoff_status") != "pending":
                raise ValueError("Only a pending handoff can be accepted")
            session["handoff_status"] = "accepted"
            await self.save(user_id, session)

    async def resume(self, user_id: str) -> None:
        async with self.session_lock(user_id):
            session = await self.load(user_id)
            session.update(handoff_status="none", notification_sent=False)
            await self.save(user_id, session)
