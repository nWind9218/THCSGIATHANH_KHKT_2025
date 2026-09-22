"""Reproduce a server-closed idle Redis connection with a disposable client."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle-seconds", type=float, default=31.0)
    parser.add_argument("--redis-url")
    parser.add_argument(
        "--drop-mode",
        choices=("external", "client-kill"),
        default="external",
    )
    parser.add_argument("--external-drop-wait", type=float, default=5.0)
    parser.add_argument(
        "--app-retry",
        action="store_true",
        help="Use the same bounded retry policy as the application client.",
    )
    args = parser.parse_args()

    load_dotenv()
    redis_url = (
        args.redis_url
        or os.getenv("UPSTASH_REDIS_URL")
        or os.getenv("REDIS_URL")
    )
    if not redis_url:
        raise RuntimeError("A Redis URL is required")

    victim_options = {
        "encoding": "utf-8",
        "decode_responses": True,
        "max_connections": 1,
        "socket_connect_timeout": 5,
        "socket_timeout": 10,
        "socket_keepalive": True,
        "health_check_interval": 30,
    }
    if args.app_retry:
        from utils.database import _redis_retry

        victim_options["retry"] = _redis_retry()

    victim = aioredis.from_url(
        redis_url,
        **victim_options,
    )
    controller = aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    try:
        await victim.ping()
        victim_id = await victim.client_id()
        print(
            json.dumps(
                {
                    "phase": "connected",
                    "redis_py": redis.__version__,
                    "victim_client_id": victim_id,
                    "app_retry": args.app_retry,
                }
            ),
            flush=True,
        )
        await asyncio.sleep(args.idle_seconds)
        killed = None
        if args.drop_mode == "client-kill":
            killed = await controller.execute_command("CLIENT", "KILL", "ID", victim_id)
            await asyncio.sleep(1)
        else:
            print(
                json.dumps(
                    {
                        "phase": "awaiting_external_drop",
                        "redis_py": redis.__version__,
                        "wait_seconds": args.external_drop_wait,
                    }
                ),
                flush=True,
            )
            await asyncio.sleep(args.external_drop_wait)

        started = time.perf_counter()
        try:
            result = await victim.get("codex:e2e:nonexistent")
            print(
                json.dumps(
                    {
                        "phase": "reconnect_result",
                        "redis_py": redis.__version__,
                        "client_kill_result": killed,
                        "success": True,
                        "value": result,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    }
                ),
                flush=True,
            )
            return 0
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "phase": "reconnect_result",
                        "redis_py": redis.__version__,
                        "client_kill_result": killed,
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    }
                ),
                flush=True,
            )
            return 1
    finally:
        await victim.aclose()
        await controller.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
