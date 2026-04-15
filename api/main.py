"""FastAPI main application with WebSocket support for real-time counseling chatbot."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from utils.database import start_pooling, close_db_pools, check_postgres_health

load_dotenv()
logger = logging.getLogger(__name__)

# WebSocket connection managers
student_connections: dict[str, WebSocket] = {}  # user_id -> WebSocket
teacher_connections: dict[str, WebSocket] = {}  # teacher_id -> WebSocket
teacher_subscriptions: dict[str, set[str]] = {}  # teacher_id -> set of user_ids


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    logger.info("🚀 Starting Mimi Counseling Bot API...")
    try:
        await start_pooling()
        logger.info("✅ Database pools initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pools: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down Mimi API...")
    await close_db_pools()
    logger.info("✅ Database pools closed")


app = FastAPI(
    title="Mimi Counseling Bot",
    description="Real-time teen counseling chatbot with teacher supervision",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Check API and database health."""
    try:
        health = await check_postgres_health()
        return {
            "status": "healthy",
            "database": health,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": "Mimi Counseling Bot",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# WebSocket endpoints imported from chat_ws and admin_ws
from api.chat_ws import router as chat_router
from api.admin_ws import router as admin_router

app.include_router(chat_router, prefix="/ws", tags=["WebSocket"])
app.include_router(admin_router, prefix="/ws/admin", tags=["Admin WebSocket"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
