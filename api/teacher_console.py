"""Teacher console HTTP API: login, conversation list/detail, takeover, reply, hand back to Mimi, live events (SSE)."""

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from api import console_auth as auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/console/api", tags=["Teacher Console"])

WINDOW_SECONDS = 24 * 3600
MAX_TEXT = 4000
MAX_ALIAS = 60
HEARTBEAT_SECONDS = 20.0
PSID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VALID_STATUSES = ("none", "pending", "accepted")
STATUS_RANK = {"pending": 0, "accepted": 1, "none": 2}
FILTERS = ("all", *VALID_STATUSES)

NOT_FOUND = "Không tìm thấy cuộc trò chuyện"
BAD_REQUEST = "Dữ liệu gửi lên không hợp lệ"
NEED_ACCEPT = "Hãy bấm Tiếp nhận trước khi trả lời"
WINDOW_CLOSED = "Đã quá 24 giờ kể từ tin nhắn cuối của học sinh, Facebook không cho phép nhắn thêm"
SEND_FAILED = "Facebook chưa nhận tin nhắn, thầy/cô thử lại nhé"
STORE_FAILED = "Hệ thống chưa phản hồi, thầy/cô thử lại sau nhé"
GREETING_FAILED = "Đã trả lại cho Mimi nhưng chưa gửi được lời nhắn cho học sinh"
GREETING_EXPIRED = "Đã trả lại cho Mimi. Đã quá 24 giờ nên chưa gửi lời nhắn cho học sinh"


def _now() -> float:
    return time.time()


# --- dependencies (lazy imports so this module loads without the data layer) ---------------------------------

def get_log():
    from mimi_v2.console_log import get_console_log
    return get_console_log()


def get_store():
    from mimi_v2.production import get_production_engine
    return get_production_engine().store


def get_sender():
    from api.facebook_webhook import send_messenger_text
    return send_messenger_text


def get_bus():
    from mimi_v2.console_log import get_event_bus
    return get_event_bus()


def valid_psid(psid: str) -> str:
    if not PSID_PATTERN.match(psid):
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return psid


# --- request bodies ------------------------------------------------------------------------------------------

class LoginBody(BaseModel):
    username: str
    password: str


class MessageBody(BaseModel):
    text: str


class ResumeBody(BaseModel):
    send_greeting: bool = True


class AliasBody(BaseModel):
    alias: str


async def read_body(request: Request, model: type[BaseModel], *, allow_empty: bool = False) -> Any:
    """Parse the JSON body into `model`; any problem becomes a Vietnamese 422."""
    raw = await request.body()
    if not raw.strip():
        if allow_empty:
            return model()
        raise HTTPException(status_code=422, detail=BAD_REQUEST)
    try:
        return model.model_validate(json.loads(raw))
    except (ValueError, ValidationError):
        raise HTTPException(status_code=422, detail=BAD_REQUEST) from None


# --- helpers -------------------------------------------------------------------------------------------------

def to_conv(meta: dict, now: float | None = None) -> dict:
    """List item + `window_expires_at` and `can_reply`."""
    moment = _now() if now is None else now
    last_student = meta.get("last_student_ts")
    expires = last_student + WINDOW_SECONDS if last_student else None
    conv = dict(meta)
    conv["handoff_status"] = meta.get("handoff_status") if meta.get("handoff_status") in VALID_STATUSES else "none"
    conv["window_expires_at"] = expires
    conv["can_reply"] = conv["handoff_status"] == "accepted" and expires is not None and moment < expires
    return conv


def window_open(meta: dict, now: float | None = None) -> bool:
    last_student = meta.get("last_student_ts")
    return bool(last_student) and (_now() if now is None else now) < last_student + WINDOW_SECONDS


async def load_meta(log, psid: str) -> dict:
    """Meta of an existing conversation, else 404."""
    meta = await log.get_meta(psid)
    if not meta or meta.get("last_ts") is None:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return meta


async def sync_status(log, store, psid: str, meta: dict) -> dict:
    """Align the console mirror with the store (source of truth); on store trouble keep the mirror."""
    try:
        status = (await store.load(psid)).get("handoff_status")
    except Exception as exc:
        logger.warning("Console could not read handoff status psid=%s error=%s", psid, type(exc).__name__)
        return meta
    if status in VALID_STATUSES and status != meta.get("handoff_status"):
        await log.set_status(psid, status)
        meta = {**meta, "handoff_status": status}
    return meta


# --- auth endpoints ------------------------------------------------------------------------------------------

@router.post("/login")
async def login(request: Request, response: Response, config: auth.ConsoleConfig = Depends(auth.guard)):
    ip = auth.client_ip(request)
    if auth.login_limiter.blocked(ip):
        raise HTTPException(status_code=429, detail=auth.RATE_LIMIT_MESSAGE)
    body = await read_body(request, LoginBody)
    if not auth.check_credentials(config, body.username, body.password):
        auth.login_limiter.record_failure(ip)
        raise HTTPException(status_code=401, detail=auth.BAD_LOGIN_MESSAGE)
    auth.login_limiter.reset(ip)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.sign_session(config.username, config.secret),
        max_age=auth.SESSION_TTL,
        path=auth.COOKIE_PATH,
        httponly=True,
        samesite="strict",
        secure=config.cookie_secure,
    )
    return {"username": config.username}


@router.post("/logout")
async def logout(response: Response, _config: auth.ConsoleConfig = Depends(auth.guard)):
    response.delete_cookie(auth.COOKIE_NAME, path=auth.COOKIE_PATH)
    return {"ok": True}


@router.get("/me")
async def me(user: str = Depends(auth.current_user)):
    return {"username": user}


# --- conversations -------------------------------------------------------------------------------------------

@router.get("/conversations")
async def list_conversations(filter: str = "all", user: str = Depends(auth.current_user), log=Depends(get_log)):
    if filter not in FILTERS:
        raise HTTPException(status_code=422, detail=BAD_REQUEST)
    now = _now()
    items = [to_conv(meta, now) for meta in await log.list_conversations(limit=200)]
    counts = {status: sum(1 for item in items if item["handoff_status"] == status) for status in VALID_STATUSES}
    counts["all"] = len(items)
    if filter != "all":
        items = [item for item in items if item["handoff_status"] == filter]
    items.sort(key=lambda item: (STATUS_RANK[item["handoff_status"]], -(item.get("last_ts") or 0)))
    return {"conversations": items, "counts": counts, "server_time": now}


@router.get("/conversations/{psid}")
async def conversation_detail(
    user: str = Depends(auth.current_user), psid: str = Depends(valid_psid), log=Depends(get_log), store=Depends(get_store),
):
    meta = await sync_status(log, store, psid, await load_meta(log, psid))
    messages = await log.history(psid)
    await log.mark_read(psid)
    return {"conversation": to_conv({**meta, "unread": 0}), "messages": messages}


@router.post("/conversations/{psid}/accept")
async def accept(
    user: str = Depends(auth.current_user), psid: str = Depends(valid_psid), log=Depends(get_log), store=Depends(get_store),
):
    meta = await load_meta(log, psid)
    try:
        previous = (await store.load(psid)).get("handoff_status")
        await store.takeover(psid)
    except Exception as exc:
        logger.warning("Console takeover failed psid=%s error=%s", psid, type(exc).__name__)
        raise HTTPException(status_code=503, detail=STORE_FAILED) from None
    await log.set_status(psid, "accepted")
    if previous != "accepted":
        await log.append(psid, "system", f"Thầy/cô {user} đã tiếp nhận", author=user)
    return {"conversation": to_conv(await log.get_meta(psid) or {**meta, "handoff_status": "accepted"})}


@router.post("/conversations/{psid}/messages")
async def send_message(
    request: Request, user: str = Depends(auth.current_user), psid: str = Depends(valid_psid),
    log=Depends(get_log), store=Depends(get_store), sender=Depends(get_sender),
):
    body = await read_body(request, MessageBody)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Tin nhắn không được để trống")
    if len(text) > MAX_TEXT:
        raise HTTPException(status_code=422, detail=f"Tin nhắn dài tối đa {MAX_TEXT} ký tự")
    meta = await sync_status(log, store, psid, await load_meta(log, psid))
    if meta.get("handoff_status") != "accepted":
        raise HTTPException(status_code=409, detail=NEED_ACCEPT)
    if not window_open(meta):
        raise HTTPException(status_code=409, detail=WINDOW_CLOSED)
    try:
        await sender(psid, text)
    except Exception as exc:
        logger.warning("Console send failed psid=%s error=%s", psid, type(exc).__name__)
        await log.append(psid, "teacher", text, author=user, status="failed")
        raise HTTPException(status_code=502, detail=SEND_FAILED) from None
    return {"message": await log.append(psid, "teacher", text, author=user, status="sent")}


@router.post("/conversations/{psid}/resume")
async def resume(
    request: Request, user: str = Depends(auth.current_user), psid: str = Depends(valid_psid),
    log=Depends(get_log), store=Depends(get_store), sender=Depends(get_sender),
):
    body = await read_body(request, ResumeBody, allow_empty=True)
    meta = await load_meta(log, psid)
    try:
        await store.resume(psid)
    except Exception as exc:
        logger.warning("Console resume failed psid=%s error=%s", psid, type(exc).__name__)
        raise HTTPException(status_code=503, detail=STORE_FAILED) from None
    await log.set_status(psid, "none")
    await log.append(psid, "system", f"Thầy/cô {user} đã trả lại cho Mimi", author=user)
    warning = None
    if body.send_greeting:
        if not window_open(meta):
            warning = GREETING_EXPIRED
        else:
            from api.facebook_webhook import RESUME_MESSAGE
            try:
                await sender(psid, RESUME_MESSAGE)
            except Exception as exc:
                logger.warning("Console resume greeting failed psid=%s error=%s", psid, type(exc).__name__)
                warning = GREETING_FAILED
            else:
                await log.append(psid, "mimi", RESUME_MESSAGE, status="sent")
    result: dict = {"conversation": to_conv(await log.get_meta(psid) or {**meta, "handoff_status": "none"})}
    if warning:
        result["warning"] = warning
    return result


@router.put("/conversations/{psid}/alias")
async def set_alias(
    request: Request, user: str = Depends(auth.current_user), psid: str = Depends(valid_psid), log=Depends(get_log),
):
    body = await read_body(request, AliasBody)
    alias = body.alias.strip()
    if len(alias) > MAX_ALIAS:
        raise HTTPException(status_code=422, detail=f"Tên gợi nhớ dài tối đa {MAX_ALIAS} ký tự")
    meta = await load_meta(log, psid)
    await log.set_alias(psid, alias)
    return {"conversation": to_conv(await log.get_meta(psid) or {**meta, "alias": alias})}


# --- live events (SSE) ---------------------------------------------------------------------------------------

def sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def event_stream(
    request: Request, bus, *, heartbeat: float = HEARTBEAT_SECONDS, poll: float = 1.0,
) -> AsyncIterator[str]:
    """hello, then one frame per bus event, `: ping` every `heartbeat` s; ends when the client leaves."""
    queue = bus.subscribe()
    try:
        yield sse_frame("hello", {})
        last_sent = time.monotonic()
        while not await request.is_disconnected():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=min(poll, heartbeat))
            except asyncio.TimeoutError:
                if time.monotonic() - last_sent >= heartbeat:
                    last_sent = time.monotonic()
                    yield ": ping\n\n"
                continue
            last_sent = time.monotonic()
            if event.get("type") == "resync" or not event.get("psid"):
                yield sse_frame("resync", {})
            else:
                yield sse_frame("conversation", {"psid": event["psid"], "type": event.get("type")})
    finally:
        bus.unsubscribe(queue)


@router.get("/events")
async def events(request: Request, user: str = Depends(auth.current_user), bus=Depends(get_bus)):
    return StreamingResponse(
        event_stream(request, bus),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
