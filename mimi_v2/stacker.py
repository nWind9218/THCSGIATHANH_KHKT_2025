"""In-process per-user message stacking (debounce) for chat channels.

Students often send several short messages in a row ("ê", "t buồn", "kiểu bị bơ á"). Answering each one separately
makes Mimi talk over the student and makes later messages queue behind the per-conversation Redis lock. The stacker
collects messages per user and hands them to ``process_batch`` as ONE turn:

* a turn starts once the user has been quiet for ``window`` seconds (``MIMI_V2_STACK_WINDOW``, default 2.5, 0 = off),
  or ``max_wait`` seconds after the oldest waiting message so a chatty user cannot delay the answer forever;
* messages that arrive while a turn is running are queued and become the next turn (never dropped, never waiting on
  the Redis lock; the quiet-time rule applies, so a message that already waited out the window goes right away);
* every submitted message ends up in exactly one batch, in arrival order.

State lives in this process, so it assumes a single replica (see AGENT_DESIGN_DOCS/TEACHER_CONSOLE_CONTRACT.md); with
several replicas stacking would simply be per replica. Pending messages are lost only if the process stops mid-wait.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 2.5
DEFAULT_MAX_WAIT = 15.0
DEFAULT_MAX_BATCH = 20
DEFAULT_MAX_CHARS = 6000


def stack_window_from_env() -> float:
    """Seconds to wait for more messages; invalid values fall back to the default, negatives disable stacking."""
    raw = os.getenv("MIMI_V2_STACK_WINDOW", "").strip()
    if not raw:
        return DEFAULT_WINDOW
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid MIMI_V2_STACK_WINDOW=%r; using %s", raw, DEFAULT_WINDOW)
        return DEFAULT_WINDOW
    if value != value or value in (float("inf"), float("-inf")):   # NaN / inf
        return DEFAULT_WINDOW
    return max(0.0, value)


@dataclass
class StackItem:
    """One received message plus whatever the channel wants to carry along (typing tasks, delivery flags)."""
    text: str
    suppress: bool = False
    received_at: float | None = None
    typing: list = field(default_factory=list)
    arrived: float = 0.0          # loop time at which it entered the queue (set by the stacker)


@dataclass
class _State:
    psid: str
    pending: list[StackItem] = field(default_factory=list)
    worker: "asyncio.Task | None" = None
    inflight: int = 0                                     # submits between "created state" and "queued"
    arrival_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


BatchHandler = Callable[[str, "list[StackItem]"], Awaitable[None]]
ArrivalHook = Callable[[str, StackItem], Awaitable[None]]


class MessageStacker:
    def __init__(
        self,
        process_batch: BatchHandler,
        *,
        on_arrival: ArrivalHook | None = None,
        window: float | Callable[[], float] | None = None,
        max_wait: float = DEFAULT_MAX_WAIT,
        max_batch: int = DEFAULT_MAX_BATCH,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self._process_batch = process_batch
        self._on_arrival = on_arrival
        self._window = window                  # None: read MIMI_V2_STACK_WINDOW at every use
        self.max_wait = max_wait
        self.max_batch = max(1, max_batch)
        self.max_chars = max_chars
        self._states: dict[str, _State] = {}
        self._tasks: set[asyncio.Task] = set()  # strong refs: the loop only keeps weak ones

    # -- configuration ---------------------------------------------------------------------------------------------

    @property
    def window(self) -> float:
        value = self._window() if callable(self._window) else self._window
        return stack_window_from_env() if value is None else max(0.0, float(value))

    @window.setter
    def window(self, value: "float | Callable[[], float] | None") -> None:
        self._window = value

    @property
    def enabled(self) -> bool:
        return self.window > 0

    # -- public API ------------------------------------------------------------------------------------------------

    async def submit(self, psid: str, item: StackItem) -> None:
        """Queue a message; returns as soon as it is queued (the turn runs in a background task)."""
        state = self._states.get(psid)
        if state is None:
            state = self._states[psid] = _State(psid)
        state.inflight += 1          # no await since the lookup: keeps the state alive until we are queued
        try:
            async with state.arrival_lock:      # arrival hooks (console log) and queue order stay identical
                if self._on_arrival is not None:
                    try:
                        await self._on_arrival(psid, item)
                    except Exception:
                        logger.exception("Stacker arrival hook failed for %s", psid)
                item.arrived = asyncio.get_running_loop().time()
                state.pending.append(item)
                if state.worker is None:
                    task = asyncio.get_running_loop().create_task(self._run(state), name=f"stacker:{psid}")
                    state.worker = task
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        finally:
            state.inflight -= 1
            self._gc(state)

    def pending(self, psid: str) -> list[StackItem]:
        state = self._states.get(psid)
        return list(state.pending) if state else []

    def active(self) -> bool:
        return bool(self._states)

    async def drain(self) -> None:
        """Wait until every queued message has been processed (tests, graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await asyncio.sleep(0)        # let the done-callbacks drop finished tasks (gather may not yield)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- internals -------------------------------------------------------------------------------------------------

    def _gc(self, state: _State) -> None:
        if state.worker is None and not state.pending and state.inflight == 0 and self._states.get(state.psid) is state:
            del self._states[state.psid]

    async def _debounce(self, state: _State) -> None:
        loop = asyncio.get_running_loop()
        while state.pending:
            window = self.window
            deadline = min(state.pending[-1].arrived + window, state.pending[0].arrived + max(self.max_wait, window))
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    def _take(self, state: _State) -> list[StackItem]:
        count, chars = 0, 0
        for item in state.pending:
            if count and (count >= self.max_batch or chars + len(item.text) + 1 > self.max_chars):
                break
            count += 1
            chars += len(item.text) + 1
        batch, state.pending = state.pending[:count], state.pending[count:]
        return batch

    async def _run(self, state: _State) -> None:
        try:
            while True:
                await self._debounce(state)
                batch = self._take(state)
                if batch:
                    try:
                        await self._process_batch(state.psid, batch)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("Stacked turn failed for %s (%d message(s))", state.psid, len(batch))
                if not state.pending:      # no await between this check and the finally: no message can slip in
                    break
        finally:
            state.worker = None
            self._gc(state)
