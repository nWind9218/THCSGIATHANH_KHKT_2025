# Teacher Console (MVP) — Implementation Contract

Website giúp thầy cô xem tin nhắn Messenger, **tiếp nhận** cuộc trò chuyện (Mimi im lặng),
trả lời trực tiếp, rồi **trả lại cho Mimi** — thay cho lệnh `#accept` / `#resume`.

Worktree: `D:\ManhProject\THCS_GIATHANH_PROJECT\THCSGIATHANH_KHKT_2025\.worktree\teacher-console`
Python: `..\..\.venv\Scripts\python.exe` (run from the worktree root).
Test command: `../../.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`
Baseline: 188 tests pass. Every agent must leave the full suite green.

## Principles (non-negotiable)

1. **Real-time channel never carries data.** SSE only says "conversation X changed";
   the browser re-fetches via REST. Missing an event = a few seconds late, never lost data.
2. **Console must never break the bot.** Every console call inside the webhook path is wrapped
   in `try/except Exception` + `logger.warning(...)`. Bot delivery behaviour stays identical.
3. **Source of truth for handoff** is `ProductionStore.load(psid)["handoff_status"]`
   (`none` | `pending` | `accepted`). The console keeps a *mirror* in its meta hash for cheap
   list rendering; the detail endpoint repairs the mirror from the store.
4. No new third-party dependencies. Use stdlib (`hashlib`, `hmac`, `secrets`, `base64`, `json`).
5. Never put user text into HTML via `innerHTML`; use `textContent`.
6. Facebook 24h rule: teachers may only send while `now < last_student_ts + 86400`.
7. Match the surrounding code style (compact, type-hinted, short docstrings, few comments).

## File ownership (only edit your own files)

| Agent | Owns |
|---|---|
| **A — data + webhook** | `mimi_v2/console_log.py` (new), `mimi_v2/production.py` (add `takeover`), `api/facebook_webhook.py`, `tests/test_console_log.py` (new), `tests/test_console_webhook.py` (new). May adjust existing webhook tests only if behaviour is intentionally unchanged but a mock signature changed. |
| **B — API + auth** | `api/console_auth.py` (new), `api/teacher_console.py` (new), `api/main.py` (include router + static mount), `scripts/hash_console_password.py` (new), `scripts/console_demo.py` (new), `.env.production.example` (add console vars), `tests/test_console_api.py` (new) |
| **C — UI** | `api/static/console/index.html`, `api/static/console/console.css`, `api/static/console/console.js` (all new) |

Nobody commits; the coordinator integrates and commits.

---

## 1. Data layer — `mimi_v2/console_log.py` (Agent A)

Redis client from `utils.database.get_redis_client()` has `decode_responses=True`.

### Redis keys
- `console:log:{psid}` — LIST of JSON entries (RPUSH, LTRIM to last 500, EXPIRE 30 days).
- `console:conversations` — ZSET `psid → last_ts` (EXPIRE refreshed 30 days).
- `console:meta:{psid}` — HASH: `alias`, `handoff_status`, `last_ts`, `last_student_ts`,
  `last_preview` (≤120 chars), `last_role`, `unread` (int). EXPIRE 30 days.

### Entry shape
```json
{"id": "1760700000123-a1b2c3", "ts": 1760700000.123, "role": "student|mimi|teacher|system",
 "text": "...", "author": "username or null", "status": "sent|failed|null"}
```

### Public API
```python
Role = Literal["student", "mimi", "teacher", "system"]
HandoffStatus = Literal["none", "pending", "accepted"]

class EventBus:
    """In-process fan-out (one uvicorn process today). Never blocks publishers."""
    def subscribe(self) -> asyncio.Queue: ...          # maxsize=100
    def unsubscribe(self, queue: asyncio.Queue) -> None: ...
    def publish(self, event: dict) -> None: ...         # on full queue: clear it, put {"type": "resync"}

class ConsoleLog:
    def __init__(self, redis_factory=None, *, bus: EventBus | None = None,
                 max_entries: int = 500, ttl_seconds: int = 2_592_000): ...
    async def append(self, psid: str, role: Role, text: str, *, author: str | None = None,
                     status: str | None = None) -> dict: ...
        # writes entry + meta (+ unread += 1 when role == "student", last_student_ts when student)
        # + zset, in one transaction pipeline; THEN publishes {"type": "message", "psid": psid}
    async def history(self, psid: str, limit: int = 200) -> list[dict]: ...   # oldest → newest
    async def list_conversations(self, limit: int = 50) -> list[dict]: ...
        # [{"psid", "alias", "handoff_status", "last_ts", "last_student_ts",
        #   "last_preview", "last_role", "unread"}]  newest first; numbers as float/int, missing → None/0
        # handoff_status missing → "none"
    async def get_meta(self, psid: str) -> dict: ...    # same shape as one list item
    async def set_status(self, psid: str, status: HandoffStatus) -> None: ...
        # only writes + publishes {"type": "status", "psid"} when the value changed
    async def set_alias(self, psid: str, alias: str) -> None: ...   # strip, ≤60 chars; publishes {"type": "alias"}
    async def mark_read(self, psid: str) -> None: ...               # unread = 0 (no publish)

class MemoryConsoleLog(ConsoleLog):
    """Dict-backed twin with identical behaviour, for tests and scripts/console_demo.py."""

def get_event_bus() -> EventBus: ...       # process singleton
def get_console_log() -> ConsoleLog: ...   # process singleton bound to get_event_bus()
```
Redis I/O bounded with the existing `bounded(IO_TIMEOUT)` pattern from `mimi_v2/production.py`.

### `ProductionStore.takeover(user_id)` (in `mimi_v2/production.py`)
Under `conversation_lock`: load, set `handoff_status = "accepted"` from **any** state
(idempotent), `_write(..., takeover=True)`. Keep `acknowledge` unchanged for `#accept`.

## 2. Webhook changes — `api/facebook_webhook.py` (Agent A)

- Public helpers for Agent B:
  - `send_messenger_text = _send_messenger_message` (alias, same signature `(psid, text)`).
  - `RESUME_MESSAGE = "Mimi đã quay trở lại rồi đây! Mình tiếp tục câu chuyện nhé?"` (reuse in `#resume`).
- **Echo events** (`message.is_echo == true`) must never be processed as a student message
  (today they would be, with sender = page id). If env `FACEBOOK_APP_ID` is set and the echo's
  `app_id` differs from it (= a human replied from Page Inbox), log it to the console as
  `role="teacher", author="Facebook Page Inbox"` for psid = `event["recipient"]["id"]`.
  Otherwise skip silently.
- In `handle_facebook_event` (skip all console work when `sender == E2E_PROBE_PSID`):
  - `#whoami`, `#accept`, `#resume` admin commands: not logged as messages, but successful
    `#accept` / `#resume` mirror status via `set_status(target, "accepted"/"none")` and append a
    `system` entry ("Đã tiếp nhận qua Messenger" / "Đã trả lại cho Mimi qua Messenger").
  - Before the engine: `append(sender, "student", text)`.
  - After the engine: if a reply was sent, `append(sender, "mimi", reply, status="sent"|"failed")`;
    then `set_status(sender, result["handoff_status"])` when it is a valid status.
    (`_process_message` currently returns only text — get the status without breaking
    existing tests, e.g. a new helper returning the full result.)

## 3. HTTP API — `api/teacher_console.py` (Agent B)

Router mounted in `api/main.py`. Static files: `app.mount("/console/static", StaticFiles(directory=api/static/console))`.
`GET /console` returns `api/static/console/index.html` (no auth; JS calls `/console/api/me`).

Dependencies (FastAPI `Depends`, overridable in tests): `get_log()` → `get_console_log()`,
`get_store()` → `get_production_engine().store`, `get_sender()` → `api.facebook_webhook.send_messenger_text`
(import lazily inside the function), `get_bus()` → `get_event_bus()`.

### Auth (`api/console_auth.py`)
- Env: `CONSOLE_ADMIN_USERNAME`, `CONSOLE_ADMIN_PASSWORD_HASH`, `CONSOLE_SESSION_SECRET` (≥32 chars).
  Any missing → console disabled: every `/console/api/*` returns 503 `{"detail": "Bàn hỗ trợ chưa được cấu hình"}`.
- Hash format: `pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>` (≥ 390000 iterations).
  `scripts/hash_console_password.py` reads password via `getpass` twice and prints the hash.
- Session cookie `mimi_console`: `base64url(json{"u","exp"}) + "." + hmac_sha256_hex`, HttpOnly,
  SameSite=Strict, Path=/console, Max-Age 12h, `Secure` unless `CONSOLE_COOKIE_SECURE=false`.
- Compare with `hmac.compare_digest`. Login rate limit (in-memory): 5 failures / IP / 5 min → 429.
- All mutating requests (POST/PUT) require header `X-Requested-With: mimi-console` → else 403.

### Endpoints (JSON; errors are `{"detail": "<Vietnamese message>"}`)
| Method & path | Body | Result |
|---|---|---|
| `POST /console/api/login` | `{username, password}` | 200 `{username}` + cookie; 401 "Sai tài khoản hoặc mật khẩu"; 429 |
| `POST /console/api/logout` | – | 200, cookie cleared |
| `GET /console/api/me` | – | 200 `{username}` / 401 |
| `GET /console/api/conversations?filter=all\|pending\|accepted\|none` | – | `{"conversations": [item...], "counts": {"pending","accepted","none","all"}, "server_time"}` — sort: pending, accepted, none; then `last_ts` desc |
| `GET /console/api/conversations/{psid}` | – | `{"conversation": Conv, "messages": [entry...]}`; marks read; repairs mirror from store |
| `POST /console/api/conversations/{psid}/accept` | – | `store.takeover`; `set_status accepted`; system entry "Thầy/cô {u} đã tiếp nhận"; returns `{"conversation": Conv}` |
| `POST /console/api/conversations/{psid}/messages` | `{text}` 1..4000 chars | 409 "Hãy bấm Tiếp nhận trước khi trả lời" unless accepted; 409 "Đã quá 24 giờ kể từ tin nhắn cuối của học sinh…" when window closed; send → append teacher entry (`author=u`, `status="sent"`); on send failure append `status="failed"` and 502 "Facebook chưa nhận tin nhắn, thầy/cô thử lại nhé" ; returns `{"message": entry}` |
| `POST /console/api/conversations/{psid}/resume` | `{send_greeting: bool = true}` | `store.resume`; `set_status none`; system entry "Thầy/cô {u} đã trả lại cho Mimi"; optional `RESUME_MESSAGE` (only inside 24h window; failure → `warning` field, not an error) |
| `PUT /console/api/conversations/{psid}/alias` | `{alias}` ≤60 | `{"conversation": Conv}` |
| `GET /console/api/events` | – | SSE (see below) |

`Conv` = list item fields + `"window_expires_at": last_student_ts + 86400 | null`, `"can_reply": bool`
(`handoff_status == "accepted"` and window open).
PSID path param must match `^[A-Za-z0-9_-]{1,64}$` → else 404.

### SSE `GET /console/api/events` (auth required)
- Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
- First: `event: hello\ndata: {}\n\n`. Then each bus event as `event: conversation\ndata: {"psid","type"}\n\n`;
  bus `{"type": "resync"}` → `event: resync\ndata: {}\n\n`. Heartbeat `: ping\n\n` every 20 s.
- Unsubscribe from the bus on disconnect (`finally`).

### `scripts/console_demo.py`
Runs uvicorn on `127.0.0.1:8765` with a FastAPI app containing only the console router + static mount,
`MemoryConsoleLog`, an in-memory fake store (none/pending/accepted transitions), a fake sender
that just logs, 3–4 seeded conversations (one pending), demo creds `demo` / `demo12345` (hash generated
at startup), `CONSOLE_COOKIE_SECURE=false`, plus `POST /demo/student {psid, text}` that appends a
student message (to watch real-time updates). No Redis/Postgres needed.

## 4. UI — `api/static/console/` (Agent C)

Vanilla HTML/CSS/JS, no build step, **no external CDN/fonts** (student privacy). Vietnamese copy.
Designed for teachers who are not technical: large clear buttons, no codes or commands.

- **Login screen**: Tên đăng nhập, Mật khẩu, nút "Đăng nhập", friendly error text.
- **Top bar**: "Mimi · Bàn hỗ trợ thầy cô", connection indicator (● Đã kết nối / Đang kết nối lại…),
  🔔 sound toggle, username, "Đăng xuất".
- **Left column**: tabs with counts — "Cần hỗ trợ" (pending, red), "Thầy cô đang trả lời" (accepted),
  "Mimi đang trả lời" (none), "Tất cả". Cards: alias or "Học sinh ••1234" (last 4 of PSID), status
  badge, preview, relative time ("2 phút trước"), unread dot.
- **Right column** (selected conversation):
  - Header: name + ✏️ "Đặt tên gợi nhớ" (inline edit), status badge, 24h countdown
    ("Còn 21 giờ 05 phút để trả lời" / "Đã hết 24 giờ, không thể nhắn qua Facebook").
  - State banner + primary action:
    - `pending`: red banner "Mimi đã chuyển cuộc trò chuyện này cho thầy cô" + big **"Tiếp nhận cuộc trò chuyện"**.
    - `none`: neutral banner "Mimi đang trả lời" + **"Tiếp nhận"** (secondary style).
    - `accepted`: blue banner "Thầy/cô đang trả lời — Mimi tạm im lặng" + **"Trả lại cho Mimi"**
      (confirm dialog with checkbox "Gửi lời nhắn ‘Mimi đã quay trở lại…’ cho học sinh", default checked).
  - Thread: student bubbles left; Mimi right/grey labelled "Mimi"; teacher right/blue labelled with author;
    system centered small grey; failed messages red "Gửi lỗi". Auto-scroll when already near bottom.
  - Composer: textarea + "Gửi" (Enter sends, Shift+Enter newline), counter /4000; disabled with
    a plain-language reason unless `can_reply`. Keep the draft on refresh of the same conversation.
- **Real-time**: `EventSource("/console/api/events")`. On `conversation`: debounce 300 ms, refetch list,
  refetch detail if that psid is open. On `hello` after a reconnect or `resync`: refetch everything.
  Fallback: refetch list (+ open detail) every 15 s regardless. Any 401 → login screen.
- **Alerts**: new pending conversation → `document.title` prefix "(n) ", short WebAudio beep if 🔔 on,
  browser Notification if permission granted (button to ask).
- All POST/PUT send `X-Requested-With: mimi-console`, `credentials: "same-origin"`.
- Responsive: < 768px shows list OR thread with a "← Danh sách" back button. Respect
  `prefers-color-scheme: dark`. Font ≥ 15px, visible focus outlines, buttons ≥ 40px tall.
