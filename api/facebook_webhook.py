"""Facebook Messenger webhook integration for the counseling bot."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response, HTTPException, BackgroundTasks

from graph.state import CounselingState
from graph.workflow import build_counseling_graph
from memory import load_history, load_topic

logger = logging.getLogger(__name__)
router = APIRouter()

FACEBOOK_GRAPH_API_VERSION = os.getenv("FACEBOOK_GRAPH_API_VERSION", "v20.0")
FACEBOOK_VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET", "")


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
    params = {"access_token": FACEBOOK_PAGE_ACCESS_TOKEN}

    async with httpx.AsyncClient(timeout=15.0) as client:
        for chunk in _chunk_text(text):
            payload = {
                "recipient": {"id": psid},
                "message": {"text": chunk},
                "messaging_type": "RESPONSE",
            }
            response = await client.post(url, params=params, json=payload)
            if not response.is_success:
                logger.error(
                    "Facebook API error %s for psid=%s: %s",
                    response.status_code,
                    psid,
                    response.text,
                )
                response.raise_for_status()


async def _process_message(psid: str, text: str) -> str:
    # We pass only the new message. load_memory_node will load history from Redis 
    # and merge it, avoiding the duplication bug.
    messages = [{"role": "user", "content": text}]

    graph = build_counseling_graph()
    state: CounselingState = {
        "messages": messages,
        "user_id": psid,
        "intent_category": "simple", # Defaults
        "info_gap_status": "missing",
        "is_emergency": False,
        "human_takeover": False,
    }

    result = await graph.ainvoke(state)
    
    # Check if human takeover is active
    if result.get("human_takeover") and not result.get("is_emergency"):
        # If takeover is active and it wasn't JUST triggered (which would have a comfort message),
        # then we should remain silent.
        if not result.get("response_text"):
            return ""

    response_text = (result.get("response_text") or "").strip()

    if not response_text and not result.get("human_takeover"):
        response_text = "Mình đã nhận được tin nhắn của bạn, bạn nói rõ hơn một chút nhé."

    return response_text


async def handle_facebook_event(sender: str, message_text: str):
    """Asynchronous handler for Facebook events to avoid webhook timeouts."""
    try:
        response_text = await _process_message(sender, message_text)
        if response_text:
            await _send_messenger_message(sender, response_text)
    except Exception:
        logger.exception("Failed to process Messenger event for %s", sender)
        # Optional: Send a fallback only if it's not a takeover situation
        # For now, we rely on logging for debugging background tasks.


@router.get("/webhook/facebook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token and token == FACEBOOK_VERIFY_TOKEN:
        return Response(content=challenge or "", media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook/facebook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()
    signature = request.headers.get("x-hub-signature-256")

    if not _verify_signature(raw_body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    payload = json.loads(raw_body.decode("utf-8"))

    if payload.get("object") != "page":
        return {"status": "ignored"}

    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            sender = event.get("sender", {}).get("id")
            if not sender:
                continue

            message_text = ""
            message = event.get("message") or {}
            postback = event.get("postback") or {}

            if message.get("text"):
                message_text = message.get("text", "").strip()
            elif postback.get("payload"):
                message_text = postback.get("payload", "").strip()
            else:
                continue

            if not message_text:
                continue

            # Offload processing to background task to respond to Facebook immediately (within 20s)
            background_tasks.add_task(handle_facebook_event, sender, message_text)

    return {"status": "ok"}
