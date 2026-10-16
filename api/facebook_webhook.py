"""Facebook Messenger webhook integration for the counseling bot."""

from __future__ import annotations

import hashlib
import hmac
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response, HTTPException, BackgroundTasks

from mimi_v2.messenger_format import to_messenger
from mimi_v2.stacker import MessageStacker, StackItem

logger = logging.getLogger(__name__)
router = APIRouter()

FACEBOOK_GRAPH_API_VERSION = os.getenv("FACEBOOK_GRAPH_API_VERSION", "v20.0")
FACEBOOK_VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET", "")
E2E_PROBE_PSID = "codex_e2e_probe"
E2E_PROBE_HEADER = "x-mimi-e2e-probe"
RESUME_MESSAGE = "Mimi đã quay trở lại rồi đây! Mình tiếp tục câu chuyện nhé?"
_HANDOFF_STATUSES = {"none", "pending", "accepted"}

# List of authorized Facebook PSIDs for admin commands
ADMIN_FACEBOOK_PSIDS = [
    x.strip() for x in os.getenv("ADMIN_FACEBOOK_PSIDS", "").split(",") if x.strip()
]


def _runtime_context() -> dict[str, str]:
    """Collect runtime metadata without assuming a specific hosting provider."""
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
    return {
        "platform": "railway" if os.getenv("RAILWAY_ENVIRONMENT_NAME") else "render" if os.getenv("RENDER") else "local",
        "service": os.getenv("RAILWAY_SERVICE_NAME", "") or os.getenv("RENDER_SERVICE_NAME", ""),
        "instance": os.getenv("RAILWAY_REPLICA_ID", "") or os.getenv("RENDER_INSTANCE_ID", ""),
        "commit": os.getenv("RAILWAY_GIT_COMMIT_SHA", "") or os.getenv("RENDER_GIT_COMMIT", ""),
        "external_url": f"https://{railway_domain}" if railway_domain else os.getenv("RENDER_EXTERNAL_URL", ""),
    }


def _is_production_env() -> bool:
    return bool(os.getenv("RAILWAY_ENVIRONMENT_NAME") or os.getenv("RENDER"))


def _verify_signature(raw_body: bytes, signature_header: str | None) -> bool:
# ... (rest of helper functions _verify_signature, _chunk_text, _send_messenger_message stay the same)
    if not FACEBOOK_APP_SECRET:
        return True
    if not signature_header:
        return False

    try:
        algorithm, signature = signature_header.split("=", 1)
    except ValueError:
        return False

    if algorithm != "sha256":
        return False

    digest = hmac.new(
        FACEBOOK_APP_SECRET.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(digest, signature)


_MESSENGER_MAX_CHARS = 2000


def _chunk_text(text: str, size: int = _MESSENGER_MAX_CHARS) -> list[str]:
    """Split text into chunks that fit within Messenger's character limit."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= size:
            chunks.append(text)
            break
        # Try to split at last newline or space within limit
        split_at = text.rfind("\n", 0, size)
        if split_at <= 0:
            split_at = text.rfind(" ", 0, size)
        if split_at <= 0:
            split_at = size
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    return chunks


async def _send_messenger_message(psid: str, text: str) -> None:
    if not FACEBOOK_PAGE_ACCESS_TOKEN:
        raise RuntimeError("Missing PAGE_ACCESS_TOKEN for Messenger replies")

    url = f"https://graph.facebook.com/{FACEBOOK_GRAPH_API_VERSION}/me/messages"
    headers = {"Authorization": f"Bearer {FACEBOOK_PAGE_ACCESS_TOKEN}"}

    # Safety net: Messenger shows text verbatim, so strip any LaTeX/Markdown that slipped through (idempotent).
    text = to_messenger(text) or text

    async with httpx.AsyncClient(timeout=15.0) as client:
        for chunk in _chunk_text(text):
            payload = {
                "recipient": {"id": psid},
                "message": {"text": chunk},
                "messaging_type": "RESPONSE",
            }
            response = await client.post(url, headers=headers, json=payload)
            if not response.is_success:
                try:
                    error = response.json().get("error", {})
                except (ValueError, AttributeError):
                    error = {}
                logger.error(
                    "Facebook API error status=%s code=%s type=%s psid=%s message=%s",
                    response.status_code,
                    error.get("code", ""),
                    error.get("type", ""),
                    psid,
                    error.get("message", "Unknown Facebook API error"),
                )
                raise RuntimeError(
                    f"Messenger delivery failed with status {response.status_code}"
                ) from None


send_messenger_text = _send_messenger_message  # public alias used by the teacher console API

TYPING_TIMEOUT = 3.0


async def _send_sender_actions(psid: str, *actions: str) -> None:
    """Best-effort Graph sender actions (mark_seen, typing_on); a failure only logs a warning."""
    if not FACEBOOK_PAGE_ACCESS_TOKEN or not actions:
        return
    url = f"https://graph.facebook.com/{FACEBOOK_GRAPH_API_VERSION}/me/messages"
    headers = {"Authorization": f"Bearer {FACEBOOK_PAGE_ACCESS_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=TYPING_TIMEOUT) as client:
            for action in actions:
                response = await client.post(
                    url, headers=headers, json={"recipient": {"id": psid}, "sender_action": action})
                if not response.is_success:
                    logger.warning("Messenger sender action %s failed status=%s", action, response.status_code)
                    return
    except Exception as exc:
        logger.warning("Messenger sender action failed: %r", exc)


def _start_typing(psid: str) -> "asyncio.Task":
    """Show 'seen' + typing dots to the student without delaying the engine."""
    return asyncio.ensure_future(_send_sender_actions(psid, "mark_seen", "typing_on"))


async def _stop_typing(psid: str, task: "asyncio.Task | None", *, replying: bool) -> None:
    """Reply on its way: Facebook clears the dots itself. Nothing to say (takeover, error): switch them off."""
    if task is None:
        return
    if replying:
        if not task.done():
            task.cancel()
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), TYPING_TIMEOUT)
    except Exception:
        pass
    await _send_sender_actions(psid, "typing_off")


class _Parts(list):
    """Messages to deliver; also carries the engine's handoff status for the console mirror."""
    handoff_status: str | None = None
    handoff_reason: str | None = None


_HANDOFF_REASON_LABELS = {
    "model:urgent": "mô hình đánh giá khẩn cấp",
    "model:uncertain_repeat": "nghi ngờ lặp lại nhiều lần",
    "accumulation": "nhiều tín hiệu tích luỹ",
}


def handoff_reason_text(code: str) -> str:
    """Vietnamese console note for a text-free handoff reason code (never contains student text)."""
    if code.startswith("keyword:"):
        return f"Lý do chuyển thầy cô: từ khoá nguy hiểm ({code[len('keyword:'):]})"
    return f"Lý do chuyển thầy cô: {_HANDOFF_REASON_LABELS.get(code, code)}"


async def _process_message(psid: str, text: str) -> list[str]:
    """Messages to deliver for one turn, in order (e.g. crisis reply, then the one-time AI greeting)."""
    from mimi_v2.production import get_production_engine
    result = await get_production_engine().respond(psid, text)
    parts = result.get("response_parts")
    if not isinstance(parts, list):
        parts = [result.get("response_text") or ""]
    out = _Parts(part.strip() for part in parts if isinstance(part, str) and part.strip())
    out.handoff_status = result.get("handoff_status")
    reason = result.get("handoff_reason")
    out.handoff_reason = reason if isinstance(reason, str) and reason else None
    return out


async def _console(op: str, psid: str, *args, **kwargs) -> None:
    """Teacher console side effects must never change bot behaviour: log and move on."""
    if psid == E2E_PROBE_PSID:
        return
    try:
        from mimi_v2.console_log import get_console_log
        await getattr(get_console_log(), op)(psid, *args, **kwargs)
    except Exception as exc:
        logger.warning("Teacher console %s failed: %r", op, exc)


def _console_typing(psid: str, active: bool) -> None:
    """Tell open consoles that Mimi is (not) composing a reply; carries no text."""
    if psid == E2E_PROBE_PSID:
        return
    try:
        from mimi_v2.console_log import get_console_log
        get_console_log().set_typing(psid, active)
    except Exception as exc:
        logger.warning("Teacher console typing failed: %r", exc)


async def _log_page_inbox_echo(recipient: str, text: str) -> None:
    """A human replied from Facebook Page Inbox: show it in the console as a teacher message."""
    await _console("append", recipient, "teacher", text, author="Facebook Page Inbox")


async def handle_facebook_event(
    sender: str,
    message_text: str,
    suppress_delivery: bool = False,
    received_at: float | None = None,
    *,
    prelogged: bool = False,
    typing_task: "asyncio.Future | None" = None,
    allow_commands: bool = True,
    stacked: int = 1,
):
    """Run ONE engine turn for a student message (or several joined ones) and deliver the replies.

    The stacker calls this with the joined text and ``prelogged=True``: each message was already logged to the console
    and had its typing indicator started on arrival, and admin commands were handled before stacking
    (``allow_commands=False``). Called plainly it does all of that itself (``stacked`` only feeds the latency log).
    """
    try:
        if _is_production_env():
            logger.info(
                "[RUNTIME_DIAG] Processing Facebook event sender=%s text_len=%s service=%s instance=%s",
                sender,
                len(message_text or ""),
                _runtime_context().get("service"),
                _runtime_context().get("instance"),
            )

        # Check for discovery commands
        clean_text = message_text.strip().lower() if allow_commands else ""

        if clean_text == "#whoami":
            await _send_messenger_message(sender, f"ID Facebook (PSID) của bạn là:\n{sender}")
            return

        # Check for #resume command from authorized admins
        if clean_text.split()[:1] == ["#accept"]:
            if sender in ADMIN_FACEBOOK_PSIDS:
                parts = message_text.split()
                if len(parts) == 2:
                    from mimi_v2.production import get_production_engine
                    try:
                        await get_production_engine().store.acknowledge(parts[1])
                    except ValueError:
                        await _send_messenger_message(sender, "Cuộc hội thoại này chưa có yêu cầu tiếp nhận đang chờ.")
                        return
                    await _console("set_status", parts[1], "accepted")
                    await _console("append", parts[1], "system", "Đã tiếp nhận qua Messenger")
                    await _send_messenger_message(sender, "Đã ghi nhận thầy/cô tiếp nhận cuộc hội thoại.")
                    return
                await _send_messenger_message(sender, "Cú pháp: #accept [PSID]")
                return
        if clean_text.startswith("#resume"):
            if sender in ADMIN_FACEBOOK_PSIDS:
                parts = message_text.split()
                if len(parts) >= 2:
                    target_psid = parts[1].strip()
                    from mimi_v2.production import get_production_engine
                    await get_production_engine().store.resume(target_psid)
                    await _console("set_status", target_psid, "none")
                    await _console("append", target_psid, "system", "Đã trả lại cho Mimi qua Messenger")

                    # Notify admin
                    await _send_messenger_message(sender, f"✅ Đã kích hoạt lại Bot cho học sinh {target_psid}.")
                    
                    # Notify student
                    await _send_messenger_message(target_psid, RESUME_MESSAGE)
                    return
                else:
                    await _send_messenger_message(sender, "Cú pháp không đúng. Vui lòng dùng: #resume [PSID]")
                    return
            else:
                # If sender is not an admin, we just treat it as a normal message.
                logger.warning(f"Unauthorized #resume attempt from {sender}")

        if not prelogged:
            typing_task = None if suppress_delivery else _start_typing(sender)  # overlaps the log write and engine
            _console_typing(sender, True)
            await _console("append", sender, "student", message_text)
        handling_started = time.monotonic()
        try:
            response_parts = await _process_message(sender, message_text)
        except BaseException:
            _console_typing(sender, False)
            await _stop_typing(sender, typing_task, replying=False)
            raise
        await _stop_typing(sender, typing_task, replying=bool(response_parts) and not suppress_delivery)
        engine_done = time.monotonic()
        handoff_status = getattr(response_parts, "handoff_status", None)
        handoff_reason = getattr(response_parts, "handoff_reason", None)
        try:
            if handoff_reason:
                await _console("append", sender, "system", handoff_reason_text(handoff_reason))
            if response_parts:
                if suppress_delivery:
                    logger.info(
                        "[RUNTIME_DIAG] E2E probe completed sender=%s delivery_suppressed=true",
                        sender,
                    )
                else:
                    for part in response_parts:
                        # Send and log the exact same text: what the student sees is what teachers see.
                        text = to_messenger(part) or part
                        delivered = False
                        try:
                            await _send_messenger_message(sender, text)
                            delivered = True
                        finally:
                            await _console("append", sender, "mimi", text,
                                           status="sent" if delivered else "failed")
        finally:
            _console_typing(sender, False)
            if handoff_status in _HANDOFF_STATUSES:
                await _console("set_status", sender, handoff_status)
        # Latency only; no ids or message text. engine_s includes lock, storage and model calls.
        done = time.monotonic()
        logger.info("[LATENCY] %s", json.dumps({
            "queue_wait_s": round(handling_started - received_at, 3) if received_at is not None else None,
            "engine_s": round(engine_done - handling_started, 3),
            "send_s": round(done - engine_done, 3),
            "end_to_end_s": round(done - received_at, 3) if received_at is not None else None,
            "replied": bool(response_parts) and not suppress_delivery,
            "stacked": stacked,
        }))
    except Exception:
        logger.exception(
            "Failed to process Messenger event for %s runtime_ctx=%s",
            sender,
            _runtime_context(),
        )
        # Optional: Send a fallback only if it's not a takeover situation
        # For now, we rely on logging for debugging background tasks.


# --- message stacking ----------------------------------------------------------------------------------------------
# Several quick messages from one student become ONE engine turn (see mimi_v2/stacker.py). In-process state: this
# assumes a single replica; with more, stacking is per replica and nothing else breaks.

def _is_admin_command(sender: str, message_text: str) -> bool:
    """Commands handle_facebook_event runs itself (never stacked); mirrors its parsing."""
    clean_text = message_text.strip().lower()
    if clean_text == "#whoami":
        return True
    if clean_text.split()[:1] == ["#accept"] and sender in ADMIN_FACEBOOK_PSIDS:
        return True
    return clean_text.startswith("#resume") and sender in ADMIN_FACEBOOK_PSIDS


async def _on_stack_arrival(psid: str, item: StackItem) -> None:
    """Per-message side effects on arrival: seen/typing dots, console typing flag, console log entry."""
    if not item.suppress:
        item.typing.append(_start_typing(psid))
    _console_typing(psid, True)
    await _console("append", psid, "student", item.text)


async def _process_stacked_batch(psid: str, items: list[StackItem]) -> None:
    typing = [task for item in items for task in item.typing]
    typing_task = asyncio.gather(*typing, return_exceptions=True) if typing else None
    suppress = any(item.suppress for item in items)
    await handle_facebook_event(
        psid,
        "\n".join(item.text for item in items),
        suppress,
        items[0].received_at,
        prelogged=True,
        typing_task=typing_task,
        allow_commands=False,
        stacked=len(items),
    )
    waiting = get_stacker().pending(psid)
    if waiting:      # more messages arrived meanwhile: they are the next turn, so keep the dots going
        _console_typing(psid, True)
        if not suppress:
            waiting[-1].typing.append(_start_typing(psid))


_stacker: MessageStacker | None = None


def get_stacker() -> MessageStacker:
    global _stacker
    if _stacker is None:
        _stacker = MessageStacker(_process_stacked_batch, on_arrival=_on_stack_arrival)
    return _stacker


def reset_stacker() -> None:
    """Forget the process-wide stacker (tests)."""
    global _stacker
    _stacker = None


async def enqueue_facebook_event(
    sender: str,
    message_text: str,
    suppress_delivery: bool = False,
    received_at: float | None = None,
):
    """Entry point for webhook deliveries: stack the message, or handle it right away when stacking does not apply."""
    stacker = get_stacker()
    if sender == E2E_PROBE_PSID or not stacker.enabled or _is_admin_command(sender, message_text):
        await handle_facebook_event(sender, message_text, suppress_delivery, received_at)
        return
    await stacker.submit(sender, StackItem(text=message_text, suppress=suppress_delivery, received_at=received_at))


# Facebook redelivers a webhook when it thinks we were slow; the same message id must not become a second turn.
_SEEN_MID_TTL = 600.0
_SEEN_MID_MAX = 5000
_seen_mids: dict[str, float] = {}


def _is_duplicate_mid(mid: Any) -> bool:
    """True if this message id was already accepted by this process within the TTL; remembers it otherwise."""
    if not mid or not isinstance(mid, str):
        return False
    now = time.monotonic()
    if len(_seen_mids) >= _SEEN_MID_MAX:
        for key in [k for k, expires in _seen_mids.items() if expires <= now]:
            del _seen_mids[key]
        while len(_seen_mids) >= _SEEN_MID_MAX:      # still full: drop the oldest (dicts keep insertion order)
            del _seen_mids[next(iter(_seen_mids))]
    expires = _seen_mids.get(mid)
    if expires is not None and expires > now:
        return True
    _seen_mids.pop(mid, None)
    _seen_mids[mid] = now + _SEEN_MID_TTL
    return False


@router.get("/webhook/facebook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token and token == FACEBOOK_VERIFY_TOKEN:
        if _is_production_env():
            logger.info(
                "[RUNTIME_DIAG] Facebook webhook verification success service=%s external_url=%s",
                _runtime_context().get("service"),
                _runtime_context().get("external_url"),
            )
        return Response(content=challenge or "", media_type="text/plain")

    if _is_production_env():
        logger.warning(
            "[RUNTIME_DIAG] Facebook webhook verification failed mode=%s token_present=%s service=%s",
            mode,
            bool(token),
            _runtime_context().get("service"),
        )
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook/facebook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    received_at = time.monotonic()
    raw_body = await request.body()
    signature = request.headers.get("x-hub-signature-256")
    probe_requested = request.headers.get(E2E_PROBE_HEADER) == "1"
    request_id = (
        request.headers.get("x-request-id")
        or request.headers.get("x-render-request-id")
        or uuid.uuid4().hex[:12]
    )

    if _is_production_env():
        logger.info(
            "[RUNTIME_DIAG] Facebook webhook received request_id=%s body_bytes=%s signature_present=%s",
            request_id,
            len(raw_body),
            bool(signature),
        )

    if not _verify_signature(raw_body, signature):
        logger.warning(
            "[RUNTIME_DIAG] Invalid Facebook signature request_id=%s service=%s",
            request_id,
            _runtime_context().get("service"),
        )
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = json.loads(raw_body.decode("utf-8"))

    if payload.get("object") != "page":
        logger.info("[RUNTIME_DIAG] Ignored Facebook payload object=%s request_id=%s", payload.get("object"), request_id)
        return {"status": "ignored"}

    if _is_production_env():
        logger.info(
            "[RUNTIME_DIAG] Facebook payload accepted request_id=%s entries=%s",
            request_id,
            len(payload.get("entry", [])),
        )

    suppressed_deliveries = 0
    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            sender = event.get("sender", {}).get("id")
            if not sender:
                continue

            message_text = ""
            message = event.get("message") or {}
            postback = event.get("postback") or {}

            if message.get("is_echo"):
                # Our own outgoing messages (or a human's Page Inbox reply) are never student input.
                app_id = os.getenv("FACEBOOK_APP_ID", "").strip()
                echo_app = str(message.get("app_id") or "")
                recipient = (event.get("recipient") or {}).get("id")
                if app_id and echo_app and echo_app != app_id and recipient and message.get("text"):
                    background_tasks.add_task(_log_page_inbox_echo, recipient, message["text"].strip())
                continue

            if message.get("text"):
                message_text = message.get("text", "").strip()
            elif postback.get("payload"):
                message_text = postback.get("payload", "").strip()
            else:
                continue

            if not message_text:
                continue

            if _is_duplicate_mid(message.get("mid") or postback.get("mid")):
                logger.info("[RUNTIME_DIAG] Ignored duplicate Facebook delivery request_id=%s sender=%s", request_id, sender)
                continue

            suppress_delivery = probe_requested and sender == E2E_PROBE_PSID
            if suppress_delivery:
                suppressed_deliveries += 1
            # Offload processing to background task to respond to Facebook immediately (within 20s)
            background_tasks.add_task(
                enqueue_facebook_event,
                sender,
                message_text,
                suppress_delivery,
                received_at=received_at,
            )
            if _is_production_env():
                logger.info(
                    "[RUNTIME_DIAG] Enqueued Facebook event request_id=%s sender=%s text_len=%s",
                    request_id,
                    sender,
                    len(message_text),
                )

    result = {"status": "ok"}
    if probe_requested:
        result["probe_delivery_suppressed"] = suppressed_deliveries > 0
    return result
