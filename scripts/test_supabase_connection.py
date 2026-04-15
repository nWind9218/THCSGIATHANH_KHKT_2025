from __future__ import annotations

import asyncio
import json

from utils.database import check_postgres_health, close_db_pools, fetch_one, start_pooling


async def main() -> None:
    await start_pooling()
    try:
        health = await check_postgres_health()

        table_check = await fetch_one(
            """
            SELECT
                EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'user_longterm_memory') AS has_user_longterm_memory,
                EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'psychology_kb') AS has_psychology_kb,
                EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'user_memory_kb') AS has_user_memory_kb,
                EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'emergency_logs') AS has_emergency_logs
            """
        )

        function_check = await fetch_one(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'public' AND p.proname = 'sync_user_style'
            ) AS has_sync_user_style
            """
        )

        payload = {
            "postgres_health": health,
            "table_check": dict(table_check) if table_check else {},
            "function_check": dict(function_check) if function_check else {},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        await close_db_pools()


if __name__ == "__main__":
    asyncio.run(main())
