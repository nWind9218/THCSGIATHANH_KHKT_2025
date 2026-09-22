"""Probe the deployed Messenger webhook without contacting a real user.

The probe submits a correctly signed webhook event with a deliberately invalid,
non-numeric PSID and requests delivery suppression. It then observes the
configured Redis history key to confirm that the deployed background task
completed the counseling graph without contacting a real Messenger user.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid

import httpx
import redis.asyncio as redis
from dotenv import load_dotenv


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="https://mimi-hello-server.up.railway.app",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    load_dotenv()
    app_secret = os.getenv("FACEBOOK_APP_SECRET", "")
    redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL", "")
    if not app_secret or not redis_url:
        raise RuntimeError("FACEBOOK_APP_SECRET and a Redis URL are required")

    probe_id = "codex_e2e_probe"
    marker = f"CODEX_E2E_{uuid.uuid4().hex[:10]}"
    request_id = marker.lower()
    message = f"{marker} Xin chao Mimi, day la kiem tra he thong."
    payload = {
        "object": "page",
        "entry": [
            {
                "messaging": [
                    {
                        "sender": {"id": probe_id},
                        "message": {"text": message},
                    }
                ]
            }
        ],
    }
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(
        app_secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{args.base_url.rstrip('/')}/webhook/facebook",
            content=raw_body,
            headers={
                "content-type": "application/json",
                "x-hub-signature-256": f"sha256={signature}",
                "x-request-id": request_id,
                "x-mimi-e2e-probe": "1",
            },
        )
    print(
        json.dumps(
            {
                "phase": "webhook_ack",
                "request_id": request_id,
                "status": response.status_code,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "body": response.text[:300],
            }
        ),
        flush=True,
    )
    response.raise_for_status()

    redis_client = redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    history_key = f"chat:{probe_id}:history"
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            history = await redis_client.get(history_key)
            if history and marker in history:
                decoded = json.loads(history)
                print(
                    json.dumps(
                        {
                            "phase": "graph_completed",
                            "request_id": request_id,
                            "elapsed_ms": round(
                                (time.perf_counter() - started) * 1000
                            ),
                            "history_messages": len(decoded),
                            "assistant_response_saved": any(
                                item.get("role") == "assistant"
                                for item in decoded
                                if isinstance(item, dict)
                            ),
                        }
                    ),
                    flush=True,
                )
                return 0
            await asyncio.sleep(5)
    finally:
        await redis_client.aclose()

    print(
        json.dumps(
            {
                "phase": "timeout",
                "request_id": request_id,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        ),
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
