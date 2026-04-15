from __future__ import annotations

import asyncio

from utils.database import close_db_pools, exec_query, start_pooling


async def main() -> None:
    await start_pooling()
    try:
        await exec_query("CREATE EXTENSION IF NOT EXISTS vector")
        await exec_query("CREATE EXTENSION IF NOT EXISTS pgcrypto")

        await exec_query(
            """
            CREATE TABLE IF NOT EXISTS psychology_kb (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                topic TEXT,
                content TEXT,
                embedding VECTOR(1536),
                source_url TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

        await exec_query(
            """
            CREATE TABLE IF NOT EXISTS user_memory_kb (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                content TEXT,
                embedding VECTOR(1536),
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

        await exec_query(
            """
            CREATE TABLE IF NOT EXISTS emergency_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                summary TEXT,
                handled_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

        await exec_query(
            """
            CREATE INDEX IF NOT EXISTS idx_psychology_kb_embedding
            ON psychology_kb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
        )
        await exec_query(
            """
            CREATE INDEX IF NOT EXISTS idx_user_memory_kb_embedding
            ON user_memory_kb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
        )
        await exec_query(
            """
            CREATE INDEX IF NOT EXISTS idx_user_memory_kb_user_id
            ON user_memory_kb (user_id)
            """
        )

        print("Schema bootstrap completed")
    finally:
        await close_db_pools()


if __name__ == "__main__":
    asyncio.run(main())
