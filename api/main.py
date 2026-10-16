"""FastAPI application serving the Facebook Messenger webhook for the Mimi counseling bot."""

import logging
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

try:
    from utils.database import (
        start_pooling,
        close_db_pools,
        check_postgres_health,
        check_redis_health,
    )
except ModuleNotFoundError:
    # Support direct execution from the api folder: python main.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.database import (
        start_pooling,
        close_db_pools,
        check_postgres_health,
        check_redis_health,
    )

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

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
    description="Teen counseling chatbot on Facebook Messenger with teacher handoff",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", tags=["Health"])
async def health_check():
    """Check all stateful services required to process a message."""
    try:
        database_health, redis_health = await asyncio.gather(
            check_postgres_health(),
            check_redis_health(),
        )
        return {
            "status": "healthy",
            "database": database_health,
            "redis": redis_health,
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


from api.facebook_webhook import router as facebook_router

app.include_router(facebook_router, tags=["Facebook Webhook"])

from api.teacher_console import router as console_router

CONSOLE_STATIC_DIR = Path(__file__).resolve().parent / "static" / "console"

app.include_router(console_router)
app.mount("/console/static", StaticFiles(directory=CONSOLE_STATIC_DIR, check_dir=False), name="console-static")


@app.get("/console", include_in_schema=False)
@app.get("/console/", include_in_schema=False)
async def console_page():
    """Teacher console single page; the page itself calls /console/api/me to decide whether to show login."""
    index = CONSOLE_STATIC_DIR / "index.html"
    if not index.is_file():
        return JSONResponse(status_code=404, content={"detail": "Bàn hỗ trợ chưa có giao diện"})
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
