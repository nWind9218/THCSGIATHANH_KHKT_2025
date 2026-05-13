import asyncpg
import redis.asyncio as aioredis
import os
import asyncio
import logging
from dotenv import load_dotenv
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
DATABASE_URL = os.getenv("DATABASE_URL")
NEO_URL = os.getenv("NEO_URL")
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")

logger = logging.getLogger(__name__)

pg_pool: asyncpg.Pool = None
redis_client: aioredis.Redis = None
redis_checkpoint_client = None
_lock = asyncio.Lock()


def resolve_postgres_url() -> str:
    """
    Supabase-first resolution:
    1) SUPABASE_DB_URL
    2) DATABASE_URL fallback
    """
    postgres_url = (SUPABASE_DB_URL or DATABASE_URL or "").strip()
    if not postgres_url:
        raise RuntimeError("Missing PostgreSQL URL. Set SUPABASE_DB_URL (preferred) or DATABASE_URL")
    return postgres_url


def resolve_redis_url() -> str:
    """
    Upstash-first resolution for Redis-compatible workloads:
    1) UPSTASH_REDIS_URL (recommended, rediss://...)
    2) REDIS_URL fallback (local/dev or other Redis providers)
    """
    redis_url = (os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL") or "").strip()
    if not redis_url:
        raise RuntimeError(
            "Missing Redis URL. Set UPSTASH_REDIS_URL (preferred) or REDIS_URL"
        )
    return redis_url


async def start_pooling():
    """Khởi tạo connection pools"""
    global pg_pool, redis_client, redis_checkpoint_client
    async with _lock: # Chỉ cho phép một task được chạy trong một thời điểm
        if pg_pool is not None:
            logger.warning("⚠️ Pool đã được cài đặt!")
            return

        postgres_url = resolve_postgres_url()
        redis_url = resolve_redis_url()

        pg_pool = await asyncpg.create_pool(
            postgres_url,
            min_size=5,
            max_size=20,
            command_timeout=60,
            timeout=20,
            max_inactive_connection_lifetime=60,
            statement_cache_size=0,
            server_settings={
                'jit': 'off',
                'application_name': 'mimi_langgraph'
            }
        )
        if redis_client is None:
            redis_client = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
        if redis_checkpoint_client is None:
            redis_checkpoint_client = aioredis.from_url(
                redis_url,
                decode_responses=False, # Để LangGraph tự xử lý binary dữ liệu
                max_connections=10
            )
        # Fail fast with explicit health checks for Supabase Postgres and local Redis.
        async with pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        await redis_client.ping()
        logger.info("✅ Pooling successfully (Supabase PostgreSQL + Redis)")


async def get_redis_checkpointer():
    """
    PHƯƠNG PHÁP THỦ CÔNG: Tránh hoàn toàn lỗi 'decode' và 'surrogates'
    """
    # 1. Tạo client thô (Binary) - KHÔNG decode_responses
    redis_url = resolve_redis_url()
    storage_client = aioredis.from_url(redis_url, decode_responses=False)

    # 2. Khởi tạo Saver mà không gọi __init__ tiêu chuẩn để tránh parse URL lại
    saver = AsyncRedisSaver.__new__(AsyncRedisSaver)
    
    # 3. Gán trực tiếp client đã cấu hình vào thuộc tính nội bộ
    saver._redis = storage_client
    
    # 4. Kích hoạt bộ Serializer (serde) của LangGraph thủ công
    BaseCheckpointSaver.__init__(saver)
    
    return saver
async def close_db_pools():
    """Đóng connection pools"""
    global pg_pool, redis_client, redis_checkpoint_client
    
    if pg_pool:
        await pg_pool.close()
        pg_pool = None
        logger.info("✅ PostgreSQL pool closed")
    
    if redis_client:
        await redis_client.close()
        redis_client = None
        logger.info("✅ Redis pool closed")

    if redis_checkpoint_client:
        await redis_checkpoint_client.close()
        redis_checkpoint_client = None
        logger.info("✅ Redis checkpoint pool closed")


async def check_postgres_health() -> dict:
    """Runtime health information for the active Postgres connection."""
    await get_pg_connection()
    async with pg_pool.acquire() as conn:
        version = await conn.fetchval("SELECT version()")
        vector_installed = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        return {
            "ok": True,
            "version": version,
            "vector_extension": bool(vector_installed),
        }


async def check_redis_health() -> dict:
    """Runtime health information for the active Redis connection."""
    client = await get_redis_client()
    pong = await client.ping()
    return {
        "ok": bool(pong),
        "ping": bool(pong),
    }

async def get_pg_connection():
    """Lấy PostgreSQL pool"""
    if pg_pool is None:
        logger.info("❌ PG Pool isn't initiated successfully!")
        logger.info("✅ PG Pool starting initiate!...")
        await start_pooling()
    return pg_pool
async def get_redis_client():
    """Lấy Redis client cho các Tools (Đã sửa tên biến từ redis_pool thành redis_client)"""
    if redis_client is None:
        await start_pooling()
    return redis_client
async def exec_query(query: str, *params):
    """Execute query với parameters"""
    async with pg_pool.acquire() as conn:
        await conn.execute(query, *params)

async def fetchall(query: str, *params):
    """Fetch nhiều rows"""
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return rows

async def fetch_one(query: str, *params):
    """Fetch một row"""
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)
        return row

async def fetch_val(query: str, *params):
    """Fetch một giá trị duy nhất"""
    async with pg_pool.acquire() as conn:
        value = await conn.fetchval(query, *params)
        return value
