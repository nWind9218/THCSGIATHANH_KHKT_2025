import asyncpg
import redis.asyncio as aioredis
import os
import asyncio
import logging
from dotenv import load_dotenv
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
NEO_URL = os.getenv("NEO_URL")
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")

logger = logging.getLogger(__name__)

pg_pool: asyncpg.Pool = None
redis_client: aioredis.Redis = None
redis_checkpoint_client = None
_lock = asyncio.Lock()
async def start_pooling():
    """Khởi tạo connection pools"""
    global pg_pool, redis_client, redis_checkpoint_client
    async with _lock: # Chỉ cho phép một task được chạy trong một thời điểm
        if pg_pool is not None:
            logger.warning("⚠️ Pool đã được cài đặt!")
            return

        pg_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60,
            server_settings={
                'jit': 'off'
            }
        )
        if redis_client is None:
            redis_client = aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
        if redis_checkpoint_client is None:
            redis_checkpoint_client = aioredis.from_url(
                REDIS_URL, 
                decode_responses=False, # Để LangGraph tự xử lý binary dữ liệu
                max_connections=10
            )
        logger.info("✅ Pooling successfully (asyncpg + Redis )")
async def get_redis_checkpointer():
    """
    PHƯƠNG PHÁP THỦ CÔNG: Tránh hoàn toàn lỗi 'decode' và 'surrogates'
    """
    # 1. Tạo client thô (Binary) - KHÔNG decode_responses
    storage_client = aioredis.from_url(REDIS_URL, decode_responses=False)

    # 2. Khởi tạo Saver mà không gọi __init__ tiêu chuẩn để tránh parse URL lại
    saver = AsyncRedisSaver.__new__(AsyncRedisSaver)
    
    # 3. Gán trực tiếp client đã cấu hình vào thuộc tính nội bộ
    saver._redis = storage_client
    
    # 4. Kích hoạt bộ Serializer (serde) của LangGraph thủ công
    BaseCheckpointSaver.__init__(saver)
    
    return saver
async def close_db_pools():
    """Đóng connection pools"""
    global pg_pool, redis_client
    
    if pg_pool:
        await pg_pool.close()
        logger.info("✅ PostgreSQL pool closed")
    
    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis pool closed")

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

# ✅ asyncpg syntax: $1, $2 thay vì %s
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
