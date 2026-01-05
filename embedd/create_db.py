import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
import asyncpg
from dotenv import load_dotenv
import os

# 1. Load environment variables
load_dotenv()

# Connection pool global
db_pool: Optional[asyncpg.Pool] = None

# ==========================================
# 2. SQL Queries để tạo bảng
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL")
CREATE_EXTENSION_QUERY = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

CREATE_BOT_KNOWLEDGE_TABLE = """
CREATE TABLE IF NOT EXISTS bot_knowledge (
    problem TEXT,
    solution TEXT,
    tone VARCHAR(255),
    must_not_do TEXT,
    level INTEGER,
    language_signals TEXT,
    embedding vector(1024),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    self_harm BOOLEAN DEFAULT FALSE,
    violence BOOLEAN DEFAULT FALSE,
    urgency TEXT
);
"""

CREATE_USER_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS user_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    preferenes TEXT,
    hates TEXT,
    source TEXT,
    confidence_score INTEGER,
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

CREATE_KNOWLEDGE_NEED_VERIFY_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_need_verify (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem TEXT,
    solution TEXT,
    tone VARCHAR(255),
    must_not_do TEXT,
    level INTEGER,
    language_signals TEXT,
    confidence_score FLOAT,
    is_verified BOOLEAN DEFAULT FALSE,
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    self_harm BOOLEAN DEFAULT FALSE,
    violence BOOLEAN DEFAULT FALSE,
    urgency TEXT
);
"""

CREATE_UPDATED_AT_TRIGGER = """
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW."updatedAt" = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_knowledge_need_verify_updated_at ON knowledge_need_verify;
CREATE TRIGGER update_knowledge_need_verify_updated_at 
    BEFORE UPDATE ON knowledge_need_verify 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_bot_knowledge_updated_at ON bot_knowledge;
CREATE TRIGGER update_bot_knowledge_updated_at 
    BEFORE UPDATE ON bot_knowledge 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""


# ==========================================
# 3. Khởi tạo Connection Pool
# ==========================================

async def init_db_pool():
    """Khởi tạo connection pool"""
    global db_pool
    db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60,
        server_settings={
            'jit': 'off'
        }
    )
    print("✅ Database connection pool initialized")
    return db_pool

async def close_db_pool():
    """Đóng connection pool"""
    global db_pool
    if db_pool:
        await db_pool.close()
        print("✅ Database connection pool closed")

async def get_connection():
    """Lấy connection từ pool"""
    if not db_pool:
        await init_db_pool()
    return await db_pool.acquire()

# ==========================================
# 4. Tạo bảng trong Database
# ==========================================

async def create_tables():
    """Tạo các bảng trong database"""
    conn = await get_connection()
    try:
        # Tạo extension vector
        await conn.execute(CREATE_EXTENSION_QUERY)
        print("✅ Vector extension created")
        
        # Tạo các bảng
        await conn.execute(CREATE_BOT_KNOWLEDGE_TABLE)
        print("✅ bot_knowledge table created")
        
        await conn.execute(CREATE_USER_MEMORY_TABLE)
        print("✅ user_memory table created")
        
        await conn.execute(CREATE_KNOWLEDGE_NEED_VERIFY_TABLE)
        print("✅ knowledge_need_verify table created")
        
        # Tạo triggers
        await conn.execute(CREATE_UPDATED_AT_TRIGGER)
        print("✅ Triggers created")
        
    finally:
        await db_pool.release(conn)

# ==========================================
# 5. Các hàm Query
# ==========================================

async def get_user_preferences(user_id: str) -> List[Dict[str, Any]]:
    """Lấy preferences của user"""
    query = """
        SELECT id, user_id, preferenes, hates, source, confidence_score, "updatedAt"
        FROM user_memory
        WHERE user_id = $1
    """
    conn = await get_connection()
    try:
        rows = await conn.fetch(query, user_id)
        return [dict(row) for row in rows]
    finally:
        await db_pool.release(conn)

async def add_new_knowledge_verify(data: dict) -> Dict[str, Any]:
    """Thêm knowledge mới cần verify"""
    query = """
        INSERT INTO knowledge_need_verify 
        (problem, solution, tone, must_not_do, level, language_signals, 
         confidence_score, self_harm, violence, urgency)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id, problem, solution, tone, must_not_do, level, 
                  language_signals, confidence_score, is_verified, 
                  "createdAt", "updatedAt", self_harm, violence, urgency
    """
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            query,
            data.get("problem"),
            data.get("solution"),
            data.get("tone"),
            data.get("must_not_do"),
            data.get("level"),
            data.get("language_signals"),
            data.get("confidence_score"),
            data.get("self_harm", False),
            data.get("violence", False),
            data.get("urgency")
        )
        return dict(row)
    finally:
        await db_pool.release(conn)

async def add_bot_knowledge(data: dict) -> Dict[str, Any]:
    """Thêm bot knowledge mới với embedding"""
    query = """
        INSERT INTO bot_knowledge 
        (problem, solution, tone, must_not_do, level, language_signals, 
         embedding, self_harm, violence, urgency)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id, problem, solution, tone, must_not_do, level, 
                  language_signals, updated_at, self_harm, violence, urgency
    """
    conn = await get_connection()
    try:
        row = await conn.fetchrow(
            query,
            data.get("problem"),
            data.get("solution"),
            data.get("tone"),
            data.get("must_not_do"),
            data.get("level"),
            data.get("language_signals"),
            data.get("embedding"),  # Vector embedding
            data.get("self_harm", False),
            data.get("violence", False),
            data.get("urgency")
        )
        return dict(row)
    finally:
        await db_pool.release(conn)

async def search_similar_knowledge(embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Tìm kiếm knowledge tương tự bằng vector similarity"""
    query = """
        SELECT id, problem, solution, tone, must_not_do, level, 
               language_signals, self_harm, violence, urgency,
               embedding <-> $1::vector as distance
        FROM bot_knowledge
        ORDER BY embedding <-> $1::vector
        LIMIT $2
    """
    conn = await get_connection()
    try:
        rows = await conn.fetch(query, embedding, limit)
        return [dict(row) for row in rows]
    finally:
        await db_pool.release(conn)

async def update_user_memory(user_id: str, data: dict) -> Dict[str, Any]:
    """Cập nhật hoặc tạo mới user memory"""
    query = """
        INSERT INTO user_memory (user_id, preferenes, hates, source, confidence_score)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (user_id) 
        DO UPDATE SET 
            preferenes = EXCLUDED.preferenes,
            hates = EXCLUDED.hates,
            source = EXCLUDED.source,
            confidence_score = EXCLUDED.confidence_score,
            "updatedAt" = NOW()
        RETURNING id, user_id, preferenes, hates, source, confidence_score, "updatedAt"
    """
    conn = await get_connection()
    try:
        # Note: Cần thêm UNIQUE constraint cho user_id nếu muốn dùng ON CONFLICT
        row = await conn.fetchrow(
            query,
            user_id,
            data.get("preferenes"),
            data.get("hates"),
            data.get("source"),
            data.get("confidence_score")
        )
        return dict(row)
    finally:
        await db_pool.release(conn)

# ==========================================
# 6. Main Script
# ==========================================

async def main():
    """Main function - Tạo database và tables"""
    try:
        print("🚀 Starting database setup...")
        
        # Khởi tạo connection pool
        await init_db_pool()
        
        # Tạo các bảng
        await create_tables()
        
        print("\n✅ Database setup completed successfully!")
        print("\n📊 Available functions:")
        print("  - get_user_preferences(user_id)")
        print("  - add_new_knowledge_verify(data)")
        print("  - add_bot_knowledge(data)")
        print("  - search_similar_knowledge(embedding, limit)")
        print("  - update_user_memory(user_id, data)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Đóng connection pool
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())