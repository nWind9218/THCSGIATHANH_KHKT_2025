import asyncpg
import redis.asyncio as aioredis
import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
NEO_URL = os.getenv("NEO_URL")
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")

logger = logging.getLogger(__name__)

pg_pool: asyncpg.Pool = None
redis_pool: aioredis.Redis = None
driver: GraphDatabase.driver = None 
async def start_pooling():
    """Khởi tạo connection pools"""
    global pg_pool, redis_pool
    global driver
    if driver is not None:
        logger.info("✅ Neo4j has been initialized!")
        return
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

    # Redis Pool
    redis_pool = aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=10,
        socket_connect_timeout=5,
        socket_keepalive=True,
        health_check_interval=30
    )
    
    # Neo4J 
    driver = GraphDatabase.driver(
        NEO_URL,
        auth=(NEO_USERNAME, NEO_PASSWORD),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=30
    )
    
    logger.info("✅ Pooling successfully (asyncpg + Redis + Neo4J)")

async def close_db_pools():
    """Đóng connection pools"""
    global pg_pool, redis_pool
    
    if pg_pool:
        await pg_pool.close()
        logger.info("✅ PostgreSQL pool closed")
    
    if redis_pool:
        await redis_pool.close()
        logger.info("✅ Redis pool closed")

async def get_pg_connection():
    """Lấy PostgreSQL pool"""
    if pg_pool is None:
        raise RuntimeError("❌ PG Pool isn't initiated successfully!")
    return pg_pool

async def get_redis_client():
    """Lấy Redis client từ pool"""
    if redis_pool is None:
        raise RuntimeError("❌ Redis isn't initiated successfully")
    return redis_pool

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
