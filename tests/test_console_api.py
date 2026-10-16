import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import console_auth as auth
from api import teacher_console as console
from api.facebook_webhook import RESUME_MESSAGE

USERNAME = "co_lan"
PASSWORD = "mat-khau-thu-nghiem"
SECRET = "s" * 40
PSID = "psid_pending_1"
CSRF = {"X-Requested-With": "mimi-console"}
HOUR = 3600


@pytest.fixture(scope="module")
def password_hash():
    return auth.hash_password(PASSWORD)


# --- in-test fakes ---------------------------------------------------------------------------------------

class FakeBus:
    def __init__(self):
        self.queues = []

    def subscribe(self):
        queue = asyncio.Queue(maxsize=100)
        self.queues.append(queue)
        return queue

    def unsubscribe(self, queue):
        if queue in self.queues:
            self.queues.remove(queue)

    def publish(self, event):
        for queue in self.queues:
            queue.put_nowait(dict(event))


class FakeLog:
    def __init__(self):
        self.metas, self.entries, self.bus = {}, {}, FakeBus()

    def seed(self, psid, status="none", *, last_student_ago=60.0, alias=None, unread=0, text="xin chào"):
        now = time.time()
        student_ts = None if last_student_ago is None else now - last_student_ago
        self.metas[psid] = {
            "psid": psid, "alias": alias, "handoff_status": status, "last_ts": student_ts or now,
            "last_student_ts": student_ts, "last_preview": text, "last_role": "student", "unread": unread,
        }
        self.entries[psid] = [{"id": "1-a", "ts": student_ts or now, "role": "student", "text": text, "author": None, "status": None}]

    async def append(self, psid, role, text, *, author=None, status=None):
        entry = {"id": f"{len(self.entries.setdefault(psid, []))}-x", "ts": time.time(), "role": role, "text": text, "author": author, "status": status}
        self.entries[psid].append(entry)
        meta = self.metas.setdefault(psid, {"psid": psid, "handoff_status": "none", "unread": 0, "alias": None, "last_student_ts": None})
        meta.update(last_ts=entry["ts"], last_role=role, last_preview=text)
        return entry

    async def history(self, psid, limit=200):
        return list(self.entries.get(psid, []))[-limit:]

    async def list_conversations(self, limit=50):
        return [dict(m) for m in sorted(self.metas.values(), key=lambda m: -m["last_ts"])][:limit]

    async def get_meta(self, psid):
        return dict(self.metas.get(psid) or {
            "psid": psid, "alias": None, "handoff_status": "none", "last_ts": None,
            "last_student_ts": None, "last_preview": None, "last_role": None, "unread": 0,
        })

    async def set_status(self, psid, status):
        self.metas[psid]["handoff_status"] = status

    async def set_alias(self, psid, alias):
        self.metas[psid]["alias"] = alias.strip()[:60] or None

    async def mark_read(self, psid):
        self.metas[psid]["unread"] = 0


class FakeStore:
    def __init__(self):
        self.status, self.fail = {}, False

    async def load(self, psid):
        if self.fail:
            raise RuntimeError("redis down")
        return {"handoff_status": self.status.get(psid, "none")}

    async def takeover(self, psid):
        if self.fail:
            raise RuntimeError("redis down")
        self.status[psid] = "accepted"

    async def resume(self, psid):
        if self.fail:
            raise RuntimeError("redis down")
        self.status[psid] = "none"


class FakeSender:
    def __init__(self):
        self.calls, self.fail = [], False

    async def __call__(self, psid, text):
        if self.fail:
            raise RuntimeError("Messenger delivery failed with status 400")
        self.calls.append((psid, text))


@pytest.fixture
def env(monkeypatch, password_hash):
    monkeypatch.setenv("CONSOLE_ADMIN_USERNAME", USERNAME)
    monkeypatch.setenv("CONSOLE_ADMIN_PASSWORD_HASH", password_hash)
    monkeypatch.setenv("CONSOLE_SESSION_SECRET", SECRET)
    monkeypatch.setenv("CONSOLE_COOKIE_SECURE", "false")
    monkeypatch.delenv("CONSOLE_TRUST_PROXY", raising=False)
    auth.login_limiter.reset()
    yield
    auth.login_limiter.reset()


@pytest.fixture
def world(env):
    log, store, sender = FakeLog(), FakeStore(), FakeSender()
    log.seed(PSID, "pending", unread=2, text="Con thấy rất buồn")
    store.status[PSID] = "pending"
    app = FastAPI()
    app.include_router(console.router)
    app.dependency_overrides.update({
        console.get_log: lambda: log, console.get_store: lambda: store,
        console.get_sender: lambda: sender, console.get_bus: lambda: log.bus,
    })
    client = TestClient(app)
    return type("World", (), {"app": app, "client": client, "log": log, "store": store, "sender": sender})


@pytest.fixture
def api(world):
    """TestClient already carrying a valid session cookie."""
    world.client.cookies.set(auth.COOKIE_NAME, auth.sign_session(USERNAME, SECRET), path="/console")
    return world


def url(path=""):
    return f"/console/api/conversations{path}"


# --- password hashing and cookie ------------------------------------------------------------------------

def test_password_hash_format_and_verification(password_hash):
    algorithm, iterations, salt, digest = password_hash.split("$")
    assert algorithm == "pbkdf2_sha256" and int(iterations) >= 390_000 and salt and digest
    assert auth.verify_password(PASSWORD, password_hash)
    assert not auth.verify_password("sai", password_hash)
    assert auth.hash_password(PASSWORD) != password_hash  # random salt


@pytest.mark.parametrize("stored", ["", "plain", "md5$1$a$b", "pbkdf2_sha256$10$YQ==$YQ==", "pbkdf2_sha256$x$YQ==$YQ==", "pbkdf2_sha256$390000$!!$!!"])
def test_malformed_hash_never_matches(stored):
    assert not auth.verify_password("x", stored)


def test_session_cookie_roundtrip_tamper_and_expiry():
    token = auth.sign_session(USERNAME, SECRET)
    assert auth.read_session(token, SECRET) == USERNAME
    assert auth.read_session(token, "x" * 40) is None
    payload, _, signature = token.rpartition(".")
    assert auth.read_session(payload + "." + "0" * len(signature), SECRET) is None
    assert auth.read_session(auth.sign_session(USERNAME, SECRET, now=time.time() - 13 * HOUR), SECRET) is None
    assert auth.read_session("garbage", SECRET) is None and auth.read_session(None, SECRET) is None


def test_login_rate_limiter_window():
    limiter = auth.LoginRateLimiter(max_failures=2, window=10)
    limiter.record_failure("ip", now=100)
    assert not limiter.blocked("ip", now=101)
    limiter.record_failure("ip", now=102)
    assert limiter.blocked("ip", now=103)
    assert not limiter.blocked("ip", now=113)
    assert not limiter.blocked("other", now=103)


# --- auth endpoints ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["CONSOLE_ADMIN_USERNAME", "CONSOLE_ADMIN_PASSWORD_HASH", "CONSOLE_SESSION_SECRET"])
def test_console_disabled_when_env_incomplete(world, monkeypatch, missing):
    monkeypatch.delenv(missing)
    for response in (world.client.get("/console/api/me"), world.client.post("/console/api/login", headers=CSRF, json={"username": "a", "password": "b"})):
        assert response.status_code == 503
        assert response.json() == {"detail": "Bàn hỗ trợ chưa được cấu hình"}


def test_short_session_secret_disables_console(world, monkeypatch):
    monkeypatch.setenv("CONSOLE_SESSION_SECRET", "too-short")
    assert world.client.get(url()).status_code == 503


def test_login_success_sets_hardened_cookie_and_me_works(world):
    response = world.client.post("/console/api/login", headers=CSRF, json={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200 and response.json() == {"username": USERNAME}
    cookie = response.headers["set-cookie"].lower()
    assert "mimi_console=" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    assert "path=/console" in cookie and "max-age=43200" in cookie and "secure" not in cookie
    assert world.client.get("/console/api/me").json() == {"username": USERNAME}


def test_cookie_is_secure_by_default(world, monkeypatch):
    monkeypatch.delenv("CONSOLE_COOKIE_SECURE")
    response = world.client.post("/console/api/login", headers=CSRF, json={"username": USERNAME, "password": PASSWORD})
    assert "secure" in response.headers["set-cookie"].lower().split("; ")


def test_login_rejects_bad_credentials_without_saying_which(world):
    for body in ({"username": USERNAME, "password": "sai"}, {"username": "ai_do", "password": PASSWORD}):
        response = world.client.post("/console/api/login", headers=CSRF, json=body)
        assert response.status_code == 401 and response.json() == {"detail": "Sai tài khoản hoặc mật khẩu"}
    assert "set-cookie" not in response.headers


def test_login_rate_limited_after_five_failures_even_with_right_password(world):
    bad = {"username": USERNAME, "password": "sai"}
    assert [world.client.post("/console/api/login", headers=CSRF, json=bad).status_code for _ in range(5)] == [401] * 5
    good = {"username": USERNAME, "password": PASSWORD}
    assert world.client.post("/console/api/login", headers=CSRF, json=good).status_code == 429


def test_successful_login_clears_failure_count(world):
    bad = {"username": USERNAME, "password": "sai"}
    for _ in range(4):
        world.client.post("/console/api/login", headers=CSRF, json=bad)
    assert world.client.post("/console/api/login", headers=CSRF, json={"username": USERNAME, "password": PASSWORD}).status_code == 200
    for _ in range(4):
        assert world.client.post("/console/api/login", headers=CSRF, json=bad).status_code == 401


def test_login_bad_body_is_vietnamese_422(world):
    response = world.client.post("/console/api/login", headers=CSRF, json={"username": USERNAME})
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_logout_clears_cookie(world):
    world.client.post("/console/api/login", headers=CSRF, json={"username": USERNAME, "password": PASSWORD})
    assert world.client.get("/console/api/me").status_code == 200
    response = world.client.post("/console/api/logout", headers=CSRF)
    assert response.status_code == 200
    assert 'mimi_console=""' in response.headers["set-cookie"] or "max-age=0" in response.headers["set-cookie"].lower()
    assert world.client.get("/console/api/me").status_code == 401


def test_endpoints_require_a_valid_session(world):
    assert world.client.get("/console/api/me").status_code == 401
    assert world.client.get(url()).status_code == 401
    assert world.client.get("/console/api/events").status_code == 401
    assert world.client.post(url(f"/{PSID}/accept"), headers=CSRF).status_code == 401
    world.client.cookies.set(auth.COOKIE_NAME, auth.sign_session(USERNAME, SECRET)[:-3] + "abc", path="/console")
    assert world.client.get("/console/api/me").status_code == 401
    world.client.cookies.set(auth.COOKIE_NAME, auth.sign_session("nguoi_khac", SECRET), path="/console")
    assert world.client.get("/console/api/me").status_code == 401


def test_mutating_requests_need_csrf_header(api):
    assert api.client.post(url(f"/{PSID}/accept")).status_code == 403
    assert api.client.post(url(f"/{PSID}/accept"), headers={"X-Requested-With": "other"}).status_code == 403
    assert api.client.post(url(f"/{PSID}/messages"), json={"text": "x"}).status_code == 403
    assert api.client.post(url(f"/{PSID}/resume"), json={}).status_code == 403
    assert api.client.put(url(f"/{PSID}/alias"), json={"alias": "x"}).status_code == 403
    assert api.client.post("/console/api/logout").status_code == 403
    assert api.client.post("/console/api/login", json={"username": USERNAME, "password": PASSWORD}).status_code == 403
    assert api.store.status[PSID] == "pending" and api.sender.calls == []


# --- conversations --------------------------------------------------------------------------------------------

def seed_many(log):
    log.seed("old_none", "none", last_student_ago=500)
    log.seed("new_none", "none", last_student_ago=10)
    log.seed("acc", "accepted", last_student_ago=200, alias="Bạn A")
    log.seed("pend_old", "pending", last_student_ago=900)


def test_list_sorts_by_status_then_recency_and_counts(api):
    seed_many(api.log)
    body = api.client.get(url()).json()
    assert [c["psid"] for c in body["conversations"]] == [PSID, "pend_old", "acc", "new_none", "old_none"]
    assert [c["handoff_status"] for c in body["conversations"]] == ["pending", "pending", "accepted", "none", "none"]
    assert body["counts"] == {"pending": 2, "accepted": 1, "none": 2, "all": 5}
    assert isinstance(body["server_time"], float)
    acc = next(c for c in body["conversations"] if c["psid"] == "acc")
    assert acc["can_reply"] is True and acc["window_expires_at"] == pytest.approx(acc["last_student_ts"] + 86400)


def test_list_filter_and_counts_stay_global(api):
    seed_many(api.log)
    body = api.client.get(url() + "?filter=pending").json()
    assert {c["psid"] for c in body["conversations"]} == {PSID, "pend_old"}
    assert body["counts"]["all"] == 5
    assert [c["psid"] for c in api.client.get(url() + "?filter=accepted").json()["conversations"]] == ["acc"]
    assert api.client.get(url() + "?filter=bogus").status_code == 422


def test_can_reply_requires_accepted_and_open_window(api):
    api.log.seed("acc_expired", "accepted", last_student_ago=25 * HOUR)
    api.log.seed("acc_no_student", "accepted", last_student_ago=None)
    items = {c["psid"]: c for c in api.client.get(url()).json()["conversations"]}
    assert items["acc_expired"]["can_reply"] is False
    assert items["acc_no_student"]["window_expires_at"] is None and items["acc_no_student"]["can_reply"] is False
    assert items[PSID]["can_reply"] is False  # pending, not accepted


def test_detail_returns_messages_marks_read_and_repairs_mirror(api):
    api.store.status[PSID] = "accepted"  # store moved on (e.g. #accept via Messenger), mirror still pending
    body = api.client.get(url(f"/{PSID}")).json()
    assert body["messages"][0]["text"] == "Con thấy rất buồn"
    assert body["conversation"]["handoff_status"] == "accepted" and body["conversation"]["can_reply"] is True
    assert body["conversation"]["unread"] == 0
    assert api.log.metas[PSID]["handoff_status"] == "accepted" and api.log.metas[PSID]["unread"] == 0


def test_detail_survives_store_failure(api):
    api.store.fail = True
    response = api.client.get(url(f"/{PSID}"))
    assert response.status_code == 200 and response.json()["conversation"]["handoff_status"] == "pending"


def test_unknown_or_malformed_psid_is_404(api):
    assert api.client.get(url("/khong_co")).status_code == 404
    assert api.client.get(url("/khong_co")).json() == {"detail": "Không tìm thấy cuộc trò chuyện"}
    assert api.client.get(url("/" + "a" * 65)).status_code == 404
    assert api.client.get(url("/bad.psid")).status_code == 404
    assert api.client.post(url("/khong_co/accept"), headers=CSRF).status_code == 404
    assert api.client.put(url("/bad.psid/alias"), headers=CSRF, json={"alias": "x"}).status_code == 404


def test_accept_takes_over_once(api):
    response = api.client.post(url(f"/{PSID}/accept"), headers=CSRF)
    assert response.status_code == 200
    assert response.json()["conversation"]["handoff_status"] == "accepted"
    assert api.store.status[PSID] == "accepted" and api.log.metas[PSID]["handoff_status"] == "accepted"
    system = [e for e in api.log.entries[PSID] if e["role"] == "system"]
    assert [e["text"] for e in system] == [f"Thầy/cô {USERNAME} đã tiếp nhận"]
    api.client.post(url(f"/{PSID}/accept"), headers=CSRF)  # idempotent, no duplicate system entry
    assert len([e for e in api.log.entries[PSID] if e["role"] == "system"]) == 1


def test_accept_works_from_none_state(api):
    api.log.seed("calm", "none")
    assert api.client.post(url("/calm/accept"), headers=CSRF).json()["conversation"]["handoff_status"] == "accepted"
    assert api.store.status["calm"] == "accepted"


def test_accept_store_failure_is_503_and_changes_nothing(api):
    api.store.fail = True
    response = api.client.post(url(f"/{PSID}/accept"), headers=CSRF)
    assert response.status_code == 503 and isinstance(response.json()["detail"], str)
    assert api.log.metas[PSID]["handoff_status"] == "pending"


# --- sending ------------------------------------------------------------------------------------------------------

def accept(api, psid=PSID):
    assert api.client.post(url(f"/{psid}/accept"), headers=CSRF).status_code == 200


def test_send_requires_accept_first(api):
    response = api.client.post(url(f"/{PSID}/messages"), headers=CSRF, json={"text": "chào em"})
    assert response.status_code == 409 and response.json() == {"detail": "Hãy bấm Tiếp nhận trước khi trả lời"}
    assert api.sender.calls == []


def test_send_uses_store_not_stale_mirror(api):
    api.store.status[PSID] = "none"
    api.log.metas[PSID]["handoff_status"] = "accepted"  # stale mirror
    response = api.client.post(url(f"/{PSID}/messages"), headers=CSRF, json={"text": "chào em"})
    assert response.status_code == 409 and api.sender.calls == []


def test_send_success_logs_teacher_entry(api):
    accept(api)
    response = api.client.post(url(f"/{PSID}/messages"), headers=CSRF, json={"text": "  Cô nghe em đây  "})
    assert response.status_code == 200
    message = response.json()["message"]
    assert (message["role"], message["text"], message["author"], message["status"]) == ("teacher", "Cô nghe em đây", USERNAME, "sent")
    assert api.sender.calls == [(PSID, "Cô nghe em đây")]
    assert api.log.entries[PSID][-1] == message


def test_send_blocked_after_24_hours(api):
    api.log.seed("late", "accepted", last_student_ago=24 * HOUR + 5)
    api.store.status["late"] = "accepted"
    response = api.client.post(url("/late/messages"), headers=CSRF, json={"text": "xin lỗi"})
    assert response.status_code == 409
    assert response.json()["detail"].startswith("Đã quá 24 giờ kể từ tin nhắn cuối của học sinh")
    assert api.sender.calls == []


def test_send_blocked_when_student_never_wrote(api):
    api.log.seed("ghost", "accepted", last_student_ago=None)
    api.store.status["ghost"] = "accepted"
    assert api.client.post(url("/ghost/messages"), headers=CSRF, json={"text": "a"}).status_code == 409


def test_send_failure_records_failed_entry_and_returns_502(api):
    accept(api)
    api.sender.fail = True
    response = api.client.post(url(f"/{PSID}/messages"), headers=CSRF, json={"text": "chào em"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Facebook chưa nhận tin nhắn, thầy/cô thử lại nhé"}
    last = api.log.entries[PSID][-1]
    assert (last["role"], last["status"], last["author"], last["text"]) == ("teacher", "failed", USERNAME, "chào em")


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {"text": "x" * 4001}, {}, {"text": 5}, {"txt": "a"}])
def test_send_validates_text(api, body):
    accept(api)
    response = api.client.post(url(f"/{PSID}/messages"), headers=CSRF, json=body)
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)
    assert api.sender.calls == []


def test_send_accepts_exactly_4000_chars(api):
    accept(api)
    assert api.client.post(url(f"/{PSID}/messages"), headers=CSRF, json={"text": "x" * 4000}).status_code == 200


def test_send_rejects_non_json_body(api):
    accept(api)
    response = api.client.post(url(f"/{PSID}/messages"), headers={**CSRF, "Content-Type": "application/json"}, content=b"{not json")
    assert response.status_code == 422


# --- resume -------------------------------------------------------------------------------------------------------

def test_resume_with_greeting_inside_window(api):
    accept(api)
    response = api.client.post(url(f"/{PSID}/resume"), headers=CSRF, json={"send_greeting": True})
    body = response.json()
    assert response.status_code == 200 and "warning" not in body
    assert body["conversation"]["handoff_status"] == "none" and api.store.status[PSID] == "none"
    assert api.sender.calls == [(PSID, RESUME_MESSAGE)]
    texts = [(e["role"], e["text"]) for e in api.log.entries[PSID]]
    assert ("system", f"Thầy/cô {USERNAME} đã trả lại cho Mimi") in texts and ("mimi", RESUME_MESSAGE) in texts


def test_resume_defaults_to_greeting_and_empty_body_allowed(api):
    accept(api)
    assert api.client.post(url(f"/{PSID}/resume"), headers=CSRF).status_code == 200
    assert api.sender.calls == [(PSID, RESUME_MESSAGE)]


def test_resume_without_greeting_sends_nothing(api):
    accept(api)
    response = api.client.post(url(f"/{PSID}/resume"), headers=CSRF, json={"send_greeting": False})
    assert response.status_code == 200 and "warning" not in response.json()
    assert api.sender.calls == []


def test_resume_outside_window_skips_greeting_with_warning(api):
    api.log.seed("late", "accepted", last_student_ago=30 * HOUR)
    api.store.status["late"] = "accepted"
    response = api.client.post(url("/late/resume"), headers=CSRF, json={"send_greeting": True})
    assert response.status_code == 200 and response.json()["conversation"]["handoff_status"] == "none"
    assert response.json()["warning"] and api.sender.calls == []


def test_resume_greeting_failure_is_a_warning_not_an_error(api):
    accept(api)
    api.sender.fail = True
    response = api.client.post(url(f"/{PSID}/resume"), headers=CSRF, json={"send_greeting": True})
    assert response.status_code == 200
    assert response.json()["conversation"]["handoff_status"] == "none" and response.json()["warning"]
    assert api.store.status[PSID] == "none"


def test_resume_store_failure_is_503(api):
    api.store.fail = True
    assert api.client.post(url(f"/{PSID}/resume"), headers=CSRF, json={}).status_code == 503


# --- alias ----------------------------------------------------------------------------------------------------------

def test_alias_set_trim_and_limit(api):
    response = api.client.put(url(f"/{PSID}/alias"), headers=CSRF, json={"alias": "  Bạn Hoa 8A  "})
    assert response.status_code == 200 and response.json()["conversation"]["alias"] == "Bạn Hoa 8A"
    assert api.client.put(url(f"/{PSID}/alias"), headers=CSRF, json={"alias": "x" * 61}).status_code == 422
    assert api.client.put(url(f"/{PSID}/alias"), headers=CSRF, json={"alias": "x" * 60}).status_code == 200
    assert api.client.put(url(f"/{PSID}/alias"), headers=CSRF, json={}).status_code == 422


# --- SSE ----------------------------------------------------------------------------------------------------------

class FakeRequest:
    def __init__(self):
        self.disconnected = False

    async def is_disconnected(self):
        return self.disconnected


async def next_frame(stream, timeout=2.0):
    return await asyncio.wait_for(stream.__anext__(), timeout)


@pytest.mark.asyncio
async def test_event_stream_hello_events_resync_and_cleanup():
    bus, request = FakeBus(), FakeRequest()
    stream = console.event_stream(request, bus, heartbeat=30, poll=0.01)
    assert await next_frame(stream) == "event: hello\ndata: {}\n\n"
    assert len(bus.queues) == 1
    bus.publish({"type": "message", "psid": "abc"})
    assert await next_frame(stream) == 'event: conversation\ndata: {"psid":"abc","type":"message"}\n\n'
    bus.publish({"type": "resync"})
    assert await next_frame(stream) == "event: resync\ndata: {}\n\n"
    await stream.aclose()
    assert bus.queues == []


@pytest.mark.asyncio
async def test_event_stream_heartbeat_and_stops_on_disconnect():
    bus, request = FakeBus(), FakeRequest()
    stream = console.event_stream(request, bus, heartbeat=0.05, poll=0.01)
    await next_frame(stream)
    assert await next_frame(stream) == ": ping\n\n"
    request.disconnected = True
    with pytest.raises(StopAsyncIteration):
        await next_frame(stream)
    assert bus.queues == []


@pytest.mark.asyncio
async def test_event_stream_unsubscribes_when_cancelled():
    bus, request = FakeBus(), FakeRequest()
    stream = console.event_stream(request, bus, heartbeat=30, poll=0.01)
    await next_frame(stream)
    pending = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.03)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert bus.queues == []


@pytest.mark.asyncio
async def test_events_endpoint_headers_and_first_frame():
    bus = FakeBus()
    response = await console.events(FakeRequest(), user=USERNAME, bus=bus)
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache" and response.headers["x-accel-buffering"] == "no"
    assert await next_frame(response.body_iterator) == "event: hello\ndata: {}\n\n"
    await response.body_iterator.aclose()
    assert bus.queues == []


# --- integration with the real in-memory data layer -----------------------------------------------------------------

def test_full_flow_with_memory_console_log(env):
    from mimi_v2.console_log import EventBus, MemoryConsoleLog

    bus = EventBus()
    log, store, sender = MemoryConsoleLog(bus=bus), FakeStore(), FakeSender()
    app = FastAPI()
    app.include_router(console.router)
    app.dependency_overrides.update({
        console.get_log: lambda: log, console.get_store: lambda: store,
        console.get_sender: lambda: sender, console.get_bus: lambda: bus,
    })
    client = TestClient(app)
    client.cookies.set(auth.COOKIE_NAME, auth.sign_session(USERNAME, SECRET), path="/console")
    queue = bus.subscribe()

    async def seed():
        await log.append("stu1", "student", "Con cần được giúp")
        await log.set_status("stu1", "pending")
    asyncio.run(seed())
    store.status["stu1"] = "pending"

    listing = client.get(url()).json()
    assert listing["counts"]["pending"] == 1 and listing["conversations"][0]["unread"] == 1
    assert client.get(url("/stu1")).json()["conversation"]["unread"] == 0
    assert client.post(url("/stu1/accept"), headers=CSRF).json()["conversation"]["can_reply"] is True
    assert client.post(url("/stu1/messages"), headers=CSRF, json={"text": "Cô đây"}).json()["message"]["status"] == "sent"
    resumed = client.post(url("/stu1/resume"), headers=CSRF, json={"send_greeting": True}).json()
    assert resumed["conversation"]["handoff_status"] == "none"
    roles = [e["role"] for e in client.get(url("/stu1")).json()["messages"]]
    assert roles == ["student", "system", "teacher", "system", "mimi"]
    assert sender.calls == [("stu1", "Cô đây"), ("stu1", RESUME_MESSAGE)]
    kinds = []
    while not queue.empty():
        kinds.append(queue.get_nowait()["type"])
    assert "status" in kinds and "message" in kinds


# --- wiring in api.main and the demo script ----------------------------------------------------------------------------

def test_main_app_serves_console_page_and_routes(env, monkeypatch, tmp_path):
    import api.main as main

    client = TestClient(main.app)  # no lifespan: no database needed
    monkeypatch.setattr(main, "CONSOLE_STATIC_DIR", tmp_path)
    assert client.get("/console").status_code == 404
    (tmp_path / "index.html").write_text("<html>console</html>", encoding="utf-8")
    for path in ("/console", "/console/"):
        response = client.get(path)
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
        assert "console" in response.text
    assert client.get("/console/api/me").status_code == 401
    assert any(route.path == "/console/static" for route in main.app.routes)


def test_demo_app_end_to_end(env):
    from scripts.console_demo import DEMO_PASSWORD, DEMO_USER, build_app

    with TestClient(build_app()) as client:  # lifespan seeds the demo conversations
        assert client.post("/console/api/login", headers=CSRF, json={"username": DEMO_USER, "password": DEMO_PASSWORD}).status_code == 200
        body = client.get(url()).json()
        assert body["counts"]["pending"] == 1 and body["counts"]["all"] == 4
        assert body["conversations"][0]["handoff_status"] == "pending"
        psid = body["conversations"][0]["psid"]
        assert client.post("/demo/student", json={"psid": psid, "text": "Con đây"}).status_code == 200
        assert client.post(url(f"/{psid}/accept"), headers=CSRF).status_code == 200
        assert client.post(url(f"/{psid}/messages"), headers=CSRF, json={"text": "Thầy đây"}).status_code == 200


# --- typing indicator + fast detail ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_stream_forwards_typing_as_a_flag_without_text():
    bus, request = FakeBus(), FakeRequest()
    stream = console.event_stream(request, bus, heartbeat=30, poll=0.01)
    await next_frame(stream)
    bus.publish({"type": "typing", "psid": "abc", "active": True, "text": "secret"})
    assert await next_frame(stream) == 'event: typing\ndata: {"psid":"abc","active":true}\n\n'
    bus.publish({"type": "typing", "psid": "abc", "active": False})
    assert await next_frame(stream) == 'event: typing\ndata: {"psid":"abc","active":false}\n\n'
    await stream.aclose()


@pytest.mark.asyncio
async def test_event_stream_replays_conversations_already_typing_after_hello():
    from mimi_v2.console_log import EventBus
    bus = EventBus()
    bus.publish({"type": "typing", "psid": "b", "active": True})
    bus.publish({"type": "typing", "psid": "a", "active": True})
    stream = console.event_stream(FakeRequest(), bus, heartbeat=30, poll=0.01)
    assert await next_frame(stream) == "event: hello\ndata: {}\n\n"
    assert await next_frame(stream) == 'event: typing\ndata: {"psid":"a","active":true}\n\n'
    assert await next_frame(stream) == 'event: typing\ndata: {"psid":"b","active":true}\n\n'
    await stream.aclose()


def test_detail_reads_run_concurrently_and_skip_mark_read_when_nothing_unread(api):
    calls = []
    original = api.log.mark_read

    async def mark_read(psid):
        calls.append(psid)
        await original(psid)
    api.log.mark_read = mark_read
    assert api.client.get(url(f"/{PSID}")).json()["conversation"]["unread"] == 0
    assert calls == [PSID]                      # seeded with unread=2
    api.client.get(url(f"/{PSID}"))
    assert calls == [PSID]                      # already read: no extra Redis writes on every poll


@pytest.mark.asyncio
async def test_detail_overlaps_meta_history_and_store_reads():
    log, store = FakeLog(), FakeStore()
    log.seed(PSID, "none")
    inflight, peak = 0, 0

    def slow(fn):
        async def wrapper(*args, **kwargs):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0.02)
            try:
                return await fn(*args, **kwargs)
            finally:
                inflight -= 1
        return wrapper
    log.get_meta, log.history, store.load = slow(log.get_meta), slow(log.history), slow(store.load)
    result = await console.conversation_detail(user=USERNAME, psid=PSID, log=log, store=store)
    assert peak == 3 and result["conversation"]["psid"] == PSID and len(result["messages"]) == 1
