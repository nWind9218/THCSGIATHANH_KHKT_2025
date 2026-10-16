"""Message stacking: quick consecutive Messenger messages become one engine turn. Everything is faked and in-memory."""
import asyncio
import json
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

import api.facebook_webhook as webhook
import mimi_v2.production as production
from mimi_v2.stacker import MessageStacker, StackItem, stack_window_from_env

WINDOW = 0.05


class FakeEngine:
    """Records every turn; a turn can be held open with `hold` and can fail on chosen texts."""

    def __init__(self):
        self.calls = []                  # (psid, text) in the order turns started
        self.started = asyncio.Event()
        self.hold = None                 # asyncio.Event: the next turn waits on it
        self.fail_on = set()
        self.delay = 0.0
        self.store = SimpleNamespace(acknowledge=AsyncMock(), resume=AsyncMock())

    async def respond(self, psid, text):
        self.calls.append((psid, text))
        self.started.set()
        if self.hold is not None:
            hold, self.hold = self.hold, None
            await hold.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_on & set(text.split("\n")):
            raise RuntimeError("model down")
        return {"response_text": f"reply#{len(self.calls)}", "handoff_status": "none"}


@pytest.fixture
def env(monkeypatch):
    engine = FakeEngine()
    send = AsyncMock()
    actions = []

    async def fake_actions(psid, *names):
        actions.extend(f"{name}:{psid}" for name in names)

    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    monkeypatch.setattr(webhook, "_send_messenger_message", send)
    monkeypatch.setattr(webhook, "_send_sender_actions", fake_actions)
    monkeypatch.setattr(webhook, "ADMIN_FACEBOOK_PSIDS", ["admin"])
    monkeypatch.delenv("MIMI_V2_STACK_WINDOW", raising=False)
    webhook.reset_stacker()
    stacker = webhook.get_stacker()
    stacker.window = WINDOW
    import mimi_v2.console_log as console_log
    yield SimpleNamespace(engine=engine, send=send, actions=actions, stacker=stacker, log=console_log.get_console_log())
    webhook.reset_stacker()


async def wait_until(predicate, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


def entries(history):
    return [(e["role"], e["text"]) for e in history]


# --- core behaviour -------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_three_quick_messages_become_one_turn_with_one_set_of_replies(env):
    for text in ("ê", "t buồn", "kiểu bị bơ á"):
        await webhook.enqueue_facebook_event("s1", text)
    assert env.engine.calls == []                      # still inside the window; enqueue never blocks on the turn
    await env.stacker.drain()
    assert env.engine.calls == [("s1", "ê\nt buồn\nkiểu bị bơ á")]
    env.send.assert_awaited_once_with("s1", "reply#1")
    assert entries(await env.log.history("s1")) == [
        ("student", "ê"), ("student", "t buồn"), ("student", "kiểu bị bơ á"), ("mimi", "reply#1")]
    assert not env.stacker.active()


@pytest.mark.asyncio
async def test_window_restarts_with_every_new_message(env):
    await webhook.enqueue_facebook_event("s1", "a")
    await asyncio.sleep(WINDOW * 0.6)
    await webhook.enqueue_facebook_event("s1", "b")
    await asyncio.sleep(WINDOW * 0.6)                  # > WINDOW since "a", < WINDOW since "b"
    assert env.engine.calls == []
    await env.stacker.drain()
    assert env.engine.calls == [("s1", "a\nb")]


@pytest.mark.asyncio
async def test_message_during_a_running_turn_is_merged_into_the_next_turn(env):
    env.engine.hold = gate = asyncio.Event()
    await webhook.enqueue_facebook_event("s1", "một")
    await asyncio.wait_for(env.engine.started.wait(), 2)
    await webhook.enqueue_facebook_event("s1", "hai")    # Mimi is mid-answer: queued, no lock wait, no drop
    assert [c[1] for c in env.engine.calls] == ["một"]
    gate.set()
    await env.stacker.drain()
    assert [c[1] for c in env.engine.calls] == ["một", "hai"]
    assert [c.args[1] for c in env.send.await_args_list] == ["reply#1", "reply#2"]
    assert entries(await env.log.history("s1")) == [
        ("student", "một"), ("student", "hai"), ("mimi", "reply#1"), ("mimi", "reply#2")]


@pytest.mark.asyncio
async def test_two_messages_during_a_running_turn_become_one_next_turn(env):
    env.engine.hold = gate = asyncio.Event()
    await webhook.enqueue_facebook_event("s1", "một")
    await asyncio.wait_for(env.engine.started.wait(), 2)
    await webhook.enqueue_facebook_event("s1", "hai")
    await webhook.enqueue_facebook_event("s1", "ba")
    gate.set()
    await env.stacker.drain()
    assert [c[1] for c in env.engine.calls] == ["một", "hai\nba"]
    assert env.send.await_count == 2


@pytest.mark.asyncio
async def test_a_message_that_already_waited_out_the_window_goes_right_after_the_turn(env):
    env.engine.hold = gate = asyncio.Event()
    loop = asyncio.get_running_loop()
    env.stacker.window = 0.3
    await webhook.enqueue_facebook_event("s1", "một")
    await asyncio.wait_for(env.engine.started.wait(), 2)      # first turn starts after the 0.3 s window
    await webhook.enqueue_facebook_event("s1", "hai")
    await asyncio.sleep(0.35)                                 # "hai" is already older than the window
    released = loop.time()
    gate.set()
    await wait_until(lambda: len(env.engine.calls) == 2, timeout=2)
    assert loop.time() - released < 0.15                       # no second full window after the turn
    await env.stacker.drain()
    assert [c[1] for c in env.engine.calls] == ["một", "hai"]


@pytest.mark.asyncio
async def test_window_zero_keeps_the_old_one_turn_per_message_behaviour(env, monkeypatch):
    env.stacker.window = None
    monkeypatch.setenv("MIMI_V2_STACK_WINDOW", "0")
    for text in ("a", "b", "c"):
        await webhook.enqueue_facebook_event("s1", text)
        assert env.engine.calls[-1] == ("s1", text)      # finished before enqueue returned
    assert [c[1] for c in env.engine.calls] == ["a", "b", "c"]
    assert env.send.await_count == 3 and not env.stacker.active()
    assert entries(await env.log.history("s1")) == [
        ("student", "a"), ("mimi", "reply#1"), ("student", "b"), ("mimi", "reply#2"), ("student", "c"), ("mimi", "reply#3")]


@pytest.mark.asyncio
async def test_engine_exception_keeps_the_queue_working(env):
    env.engine.fail_on = {"hỏng"}
    await webhook.enqueue_facebook_event("s1", "hỏng")
    await env.stacker.drain()
    assert env.send.await_count == 0
    assert entries(await env.log.history("s1")) == [("student", "hỏng")]      # logged on arrival despite the failure
    assert env.log.bus.typing == set()
    await webhook.enqueue_facebook_event("s1", "còn đó không")
    await webhook.enqueue_facebook_event("s1", "alo")
    await env.stacker.drain()
    assert [c[1] for c in env.engine.calls] == ["hỏng", "còn đó không\nalo"]
    env.send.assert_awaited_once_with("s1", "reply#2")


@pytest.mark.asyncio
async def test_failing_batch_does_not_swallow_messages_queued_behind_it(env):
    env.engine.fail_on = {"x"}
    env.engine.hold = gate = asyncio.Event()
    await webhook.enqueue_facebook_event("s1", "x")
    await webhook.enqueue_facebook_event("s1", "y")
    await asyncio.wait_for(env.engine.started.wait(), 2)
    await webhook.enqueue_facebook_event("s1", "z")
    gate.set()
    await env.stacker.drain()
    assert [c[1] for c in env.engine.calls] == ["x\ny", "z"]
    env.send.assert_awaited_once_with("s1", "reply#2")


@pytest.mark.asyncio
async def test_different_students_are_stacked_independently(env):
    await webhook.enqueue_facebook_event("s1", "a")
    await webhook.enqueue_facebook_event("s2", "x")
    await webhook.enqueue_facebook_event("s1", "b")
    await webhook.enqueue_facebook_event("s2", "y")
    await env.stacker.drain()
    assert sorted(env.engine.calls) == [("s1", "a\nb"), ("s2", "x\ny")]


# --- not stacked ----------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_admin_commands_are_immediate_and_not_stacked(env):
    env.stacker.window = 30.0
    await asyncio.wait_for(webhook.enqueue_facebook_event("admin", "#whoami"), 2)
    env.send.assert_awaited_once_with("admin", "ID Facebook (PSID) của bạn là:\nadmin")
    await asyncio.wait_for(webhook.enqueue_facebook_event("admin", "#accept student"), 2)
    env.engine.store.acknowledge.assert_awaited_once_with("student")
    await asyncio.wait_for(webhook.enqueue_facebook_event("admin", "#resume student"), 2)
    env.engine.store.resume.assert_awaited_once_with("student")
    assert env.engine.calls == [] and not env.stacker.active()
    # A student's pending messages are not merged with (or held up by) a command from someone else.
    await webhook.enqueue_facebook_event("student", "hi")
    await asyncio.wait_for(webhook.enqueue_facebook_event("admin", "#whoami"), 2)
    assert env.stacker.pending("student")[0].text == "hi"
    await env.stacker.aclose()


@pytest.mark.asyncio
async def test_resume_from_a_non_admin_is_an_ordinary_stacked_message(env):
    await webhook.enqueue_facebook_event("parent", "#resume student")
    await webhook.enqueue_facebook_event("parent", "cho em hỏi")
    await env.stacker.drain()
    env.engine.store.resume.assert_not_awaited()
    assert env.engine.calls == [("parent", "#resume student\ncho em hỏi")]


@pytest.mark.asyncio
async def test_probe_is_immediate_suppressed_and_never_logged(env):
    env.stacker.window = 30.0
    await asyncio.wait_for(
        webhook.enqueue_facebook_event(webhook.E2E_PROBE_PSID, "probe", True, received_at=0.0), 2)
    assert env.engine.calls == [(webhook.E2E_PROBE_PSID, "probe")]
    env.send.assert_not_awaited()
    assert await env.log.list_conversations() == [] and env.actions == [] and not env.stacker.active()


@pytest.mark.asyncio
async def test_suppressed_stacked_batch_delivers_nothing(env):
    env.engine.hold = None
    await env.stacker.submit("s1", StackItem(text="a", suppress=True))
    await env.stacker.submit("s1", StackItem(text="b", suppress=True))
    await env.stacker.drain()
    assert env.engine.calls == [("s1", "a\nb")]
    env.send.assert_not_awaited()
    assert env.actions == []


# --- console / typing / takeover / safety ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_message_is_logged_individually_on_arrival_before_the_turn(env):
    env.engine.hold = gate = asyncio.Event()
    await webhook.enqueue_facebook_event("s1", "a")
    await webhook.enqueue_facebook_event("s1", "b")
    assert entries(await env.log.history("s1")) == [("student", "a"), ("student", "b")]    # window still open
    assert env.log.bus.typing == {"s1"}
    await asyncio.wait_for(env.engine.started.wait(), 2)
    await webhook.enqueue_facebook_event("s1", "c")
    assert entries(await env.log.history("s1")) == [("student", "a"), ("student", "b"), ("student", "c")]
    gate.set()
    await env.stacker.drain()
    texts = [t for r, t in entries(await env.log.history("s1")) if r == "student"]
    assert texts == ["a", "b", "c"]
    assert env.log.bus.typing == set()


@pytest.mark.asyncio
async def test_seen_and_typing_are_sent_on_every_arrival_and_not_switched_off_when_replying(env):
    await webhook.enqueue_facebook_event("s1", "a")
    await webhook.enqueue_facebook_event("s1", "b")
    await asyncio.sleep(0)
    assert env.actions == ["mark_seen:s1", "typing_on:s1"] * 2     # dots stay on while waiting in the window
    await env.stacker.drain()
    assert "typing_off:s1" not in env.actions


@pytest.mark.asyncio
async def test_accepted_takeover_turn_stays_silent_and_switches_typing_off(env):
    env.engine.respond = AsyncMock(return_value={"response_text": "", "response_parts": [], "handoff_status": "accepted"})
    await webhook.enqueue_facebook_event("s1", "alo")
    await webhook.enqueue_facebook_event("s1", "cô ơi")
    await env.stacker.drain()
    env.engine.respond.assert_awaited_once_with("s1", "alo\ncô ơi")
    env.send.assert_not_awaited()
    assert env.actions[-1] == "typing_off:s1"
    assert (await env.log.get_meta("s1"))["handoff_status"] == "accepted" and env.log.bus.typing == set()


@pytest.mark.asyncio
async def test_typing_resumes_for_the_queued_turn(env):
    env.engine.hold = gate = asyncio.Event()
    await webhook.enqueue_facebook_event("s1", "a")
    await asyncio.wait_for(env.engine.started.wait(), 2)
    await webhook.enqueue_facebook_event("s1", "b")
    env.actions.clear()
    gate.set()
    await wait_until(lambda: env.send.await_count >= 1)
    await asyncio.sleep(0)
    assert env.actions[-2:] == ["mark_seen:s1", "typing_on:s1"]       # dots again while turn 2 waits/runs
    await env.stacker.drain()
    assert env.log.bus.typing == set()


class _RiskModel:
    """Safety verdict is urgent only when the student text mentions self-harm; everything else is benign."""

    async def complete(self, messages, purpose="reply"):
        urgent = "tự tử" in json.dumps(messages, ensure_ascii=False)
        text = {
            "safety": json.dumps({"risk": "urgent" if urgent else "safe", "reason": "fixture"}),
            "assessment": json.dumps(dict(
                known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                topic="school", core_issue="school", safety_hint="safe", user_role="unknown")),
        }.get(purpose, "mình đây nè")
        return SimpleNamespace(text=text, usage={}, metadata={})


@pytest.mark.asyncio
async def test_risky_message_inside_a_batch_still_triggers_handoff(env, monkeypatch):
    from mimi_v2.core import CounselingEngine
    from mimi_v2.prompts import HANDOFF_MESSAGE
    from mimi_v2.store import InMemoryStore
    engine = CounselingEngine(_RiskModel(), InMemoryStore())
    engine.store = SimpleNamespace(acknowledge=AsyncMock(), resume=AsyncMock(), load=engine.store.load)
    monkeypatch.setattr(production, "get_production_engine", lambda: engine)
    for text in ("ê", "tớ muốn tự tử", "thôi đùa á"):
        await webhook.enqueue_facebook_event("s1", text)
    await env.stacker.drain()
    history = await env.log.history("s1")
    assert [e["text"] for e in history if e["role"] == "student"] == ["ê", "tớ muốn tự tử", "thôi đùa á"]
    assert any(e["role"] == "mimi" and e["text"].startswith(HANDOFF_MESSAGE) for e in history)
    assert (await env.log.get_meta("s1"))["handoff_status"] == "pending"


# --- batching limits and generic stacker ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_chatty_student_cannot_postpone_the_answer_forever():
    batches = []

    async def process(psid, items):
        batches.append([i.text for i in items])

    stacker = MessageStacker(process, window=0.05, max_wait=0.2)
    for n in range(14):                                  # a message every 30 ms for ~0.42 s, never quiet for 50 ms
        await stacker.submit("s1", StackItem(text=str(n)))
        await asyncio.sleep(0.03)
    await stacker.drain()
    assert len(batches) >= 2
    assert [t for batch in batches for t in batch] == [str(n) for n in range(14)]


@pytest.mark.asyncio
async def test_oversized_backlog_is_split_in_order_without_loss():
    batches = []

    async def process(psid, items):
        batches.append([i.text for i in items])

    stacker = MessageStacker(process, window=0.02, max_batch=3)
    for n in range(8):
        await stacker.submit("s1", StackItem(text=str(n)))
    await stacker.drain()
    assert batches == [["0", "1", "2"], ["3", "4", "5"], ["6", "7"]]
    stacker2 = MessageStacker(process, window=0.02, max_chars=10)
    batches.clear()
    for text in ("aaaa", "bbbb", "cccc"):
        await stacker2.submit("s1", StackItem(text=text))
    await stacker2.drain()
    assert batches == [["aaaa", "bbbb"], ["cccc"]]


@pytest.mark.asyncio
async def test_failing_arrival_hook_still_queues_the_message():
    seen = []

    async def process(psid, items):
        seen.extend(i.text for i in items)

    async def broken(psid, item):
        raise ConnectionError("redis down")

    stacker = MessageStacker(process, on_arrival=broken, window=0.01)
    await stacker.submit("s1", StackItem(text="a"))
    await stacker.drain()
    assert seen == ["a"]


@pytest.mark.asyncio
async def test_background_worker_is_referenced_until_it_finishes():
    done = asyncio.Event()

    async def process(psid, items):
        await asyncio.sleep(0.05)
        done.set()

    stacker = MessageStacker(process, window=0.01)
    await stacker.submit("s1", StackItem(text="a"))
    import gc
    gc.collect()
    await asyncio.wait_for(done.wait(), 2)
    await stacker.drain()
    assert not stacker._tasks and not stacker.active()


@pytest.mark.parametrize("raw,expected", [
    (None, 2.5), ("", 2.5), ("1.5", 1.5), ("0", 0.0), ("-3", 0.0), ("abc", 2.5), ("nan", 2.5), ("inf", 2.5)])
def test_window_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("MIMI_V2_STACK_WINDOW", raising=False)
    else:
        monkeypatch.setenv("MIMI_V2_STACK_WINDOW", raw)
    assert stack_window_from_env() == expected


# --- property-style: nothing lost, nothing duplicated, order kept --------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(8))
async def test_no_message_is_lost_under_random_interleavings(seed):
    rng = random.Random(seed)
    sent = {"p1": [], "p2": [], "p3": []}
    got = {psid: [] for psid in sent}
    in_turn = set()

    async def process(psid, items):
        assert psid not in in_turn, "two turns of one student overlapped"
        in_turn.add(psid)
        try:
            await asyncio.sleep(rng.choice([0, 0.005, 0.02, 0.06]))
            if rng.random() < 0.15:
                got[psid].extend(i.text for i in items)       # delivered to the engine, then it blows up
                raise RuntimeError("boom")
            got[psid].extend(i.text for i in items)
        finally:
            in_turn.discard(psid)

    stacker = MessageStacker(process, window=0.02, max_wait=0.1, max_batch=rng.choice([2, 5, 20]))
    for n in range(40):
        psid = rng.choice(list(sent))
        text = f"{psid}-{n}"
        sent[psid].append(text)
        await stacker.submit(psid, StackItem(text=text))
        await asyncio.sleep(rng.choice([0, 0, 0.002, 0.01, 0.03, 0.05]))
    await asyncio.wait_for(stacker.drain(), 10)
    assert got == sent
    assert not stacker.active() and not stacker._tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(5))
async def test_webhook_flow_logs_and_processes_every_message_exactly_once(env, seed):
    rng = random.Random(100 + seed)
    env.engine.delay = 0.01
    env.engine.fail_on = {"m3"}
    texts = [f"m{n}" for n in range(25)]
    for text in texts:
        await webhook.enqueue_facebook_event("s1", text)
        await asyncio.sleep(rng.choice([0, 0.005, 0.02, 0.08]))
    await asyncio.wait_for(env.stacker.drain(), 10)
    processed = [t for _, joined in env.engine.calls for t in joined.split("\n")]
    logged = [t for r, t in entries(await env.log.history("s1")) if r == "student"]
    assert processed == texts and logged == texts
    assert env.log.bus.typing == set()


# --- receive_webhook: dedupe by mid, probe args unchanged -----------------------------------------------------------

async def deliver(payload, headers=()):
    body = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/webhook/facebook",
                       "headers": [(b"content-type", b"application/json"), *headers]}, receive)
    tasks = BackgroundTasks()
    result = await webhook.receive_webhook(request, tasks)
    return tasks.tasks, result


def message_payload(sender, text, mid=None, key="message"):
    body = {"text": text} if key == "message" else {"payload": text}
    if mid:
        body["mid"] = mid
    return {"object": "page", "entry": [{"messaging": [{"sender": {"id": sender}, key: body}]}]}


@pytest.fixture
def clean_mids(monkeypatch):
    monkeypatch.setattr(webhook, "_seen_mids", {})


@pytest.mark.asyncio
async def test_webhook_enqueues_through_the_stacker_and_skips_duplicate_mids(env, clean_mids):
    tasks, _ = await deliver(message_payload("s1", "hi", mid="m.1"))
    assert len(tasks) == 1 and tasks[0].func is webhook.enqueue_facebook_event and tasks[0].args[:2] == ("s1", "hi")
    again, result = await deliver(message_payload("s1", "hi", mid="m.1"))      # Facebook redelivery
    assert again == [] and result == {"status": "ok"}
    other, _ = await deliver(message_payload("s1", "hi", mid="m.2"))
    assert len(other) == 1
    no_mid, _ = await deliver(message_payload("s1", "hi"))
    no_mid2, _ = await deliver(message_payload("s1", "hi"))
    assert len(no_mid) == 1 and len(no_mid2) == 1                           # nothing to dedupe on without a mid


@pytest.mark.asyncio
async def test_duplicate_delivery_is_processed_once_end_to_end(env, clean_mids):
    for _ in range(2):
        tasks, _ = await deliver(message_payload("s1", "một mình", mid="m.9"))
        for task in tasks:
            await task.func(*task.args, **task.kwargs)
    await env.stacker.drain()
    assert env.engine.calls == [("s1", "một mình")]


@pytest.mark.asyncio
async def test_burst_in_one_payload_is_stacked_in_order(env, clean_mids):
    payload = {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "s1"}, "message": {"text": text, "mid": f"m.{n}"}} for n, text in enumerate(["a", "b", "c"])]}]}
    tasks, _ = await deliver(payload)
    assert len(tasks) == 3
    for task in tasks:
        await task.func(*task.args, **task.kwargs)
    await env.stacker.drain()
    assert env.engine.calls == [("s1", "a\nb\nc")]


@pytest.mark.asyncio
async def test_postback_mid_is_also_deduped(env, clean_mids):
    first, _ = await deliver(message_payload("s1", "GET_STARTED", mid="pb.1", key="postback"))
    second, _ = await deliver(message_payload("s1", "GET_STARTED", mid="pb.1", key="postback"))
    assert len(first) == 1 and second == []


def test_seen_mid_ttl_and_bound(monkeypatch, clean_mids):
    clock = [1000.0]
    monkeypatch.setattr(webhook, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    assert not webhook._is_duplicate_mid("a") and webhook._is_duplicate_mid("a")
    clock[0] += webhook._SEEN_MID_TTL + 1
    assert not webhook._is_duplicate_mid("a")                     # expired -> accepted again
    monkeypatch.setattr(webhook, "_SEEN_MID_MAX", 3)
    for mid in ("b", "c", "d", "e", "f"):
        webhook._is_duplicate_mid(mid)
    assert len(webhook._seen_mids) <= 3
    assert not webhook._is_duplicate_mid(None) and not webhook._is_duplicate_mid("")
