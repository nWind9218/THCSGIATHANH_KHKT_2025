"""Run the teacher console against in-memory fakes (no Redis, Postgres or Facebook needed).

Usage: python scripts/console_demo.py   then open http://127.0.0.1:8765/console  (demo / demo12345)
Simulate a student:  curl -X POST localhost:8765/demo/student -H "Content-Type: application/json" \
  -d '{"psid": "demo_student_1", "text": "Con cần giúp ạ", "handoff": true}'
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO_USER, DEMO_PASSWORD = "demo", "demo12345"
HOST, PORT = "127.0.0.1", 8765
STATIC_DIR = ROOT / "api" / "static" / "console"

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from api import console_auth, teacher_console  # noqa: E402
from mimi_v2.console_log import EventBus, MemoryConsoleLog  # noqa: E402

logger = logging.getLogger("console_demo")


class FakeStore:
    """Just the handoff transitions of ProductionStore."""

    def __init__(self):
        self.status: dict[str, str] = {}

    async def load(self, psid: str) -> dict:
        return {"handoff_status": self.status.get(psid, "none")}

    async def takeover(self, psid: str) -> None:
        self.status[psid] = "accepted"

    async def resume(self, psid: str) -> None:
        self.status[psid] = "none"


class StudentMessage(BaseModel):
    psid: str
    text: str
    handoff: bool = False


async def fake_sender(psid: str, text: str) -> None:
    logger.info("[fake Facebook] -> %s: %s", psid, text)


async def seed(log: MemoryConsoleLog, store: FakeStore) -> None:
    async def student(psid, text, status="none", alias=None):
        await log.append(psid, "student", text)
        store.status[psid] = status
        await log.set_status(psid, status)
        if alias:
            await log.set_alias(psid, alias)

    await student("demo_student_1", "Con thấy rất áp lực vì điểm thi, con không ngủ được.", "pending")
    await log.append("demo_student_1", "mimi", "Mimi nghe con đây. Con kể thêm cho Mimi nhé?", status="sent")
    await log.append("demo_student_1", "student", "Con muốn nói chuyện với thầy cô.")
    await log.append("demo_student_1", "system", "Mimi đã chuyển cuộc trò chuyện này cho thầy cô")

    await student("demo_student_2", "Em bị bạn trong lớp trêu chọc hoài.", "accepted", alias="Bạn lớp 8A")
    await log.append("demo_student_2", "teacher", "Thầy/cô nghe em, em kể rõ hơn được không?", author=DEMO_USER, status="sent")

    await student("demo_student_3", "Cuối tuần này con nên ôn bài thế nào ạ?")
    await log.append("demo_student_3", "mimi", "Con thử chia nhỏ môn học theo từng ngày nhé.", status="sent")

    await student("demo_old_4", "Cảm ơn Mimi.")
    old = time.time() - 30 * 3600  # window already closed
    log.memory._hset("console:meta:demo_old_4", mapping={"last_ts": old, "last_student_ts": old})
    log.memory._zadd("console:conversations", {"demo_old_4": old})


def build_app() -> FastAPI:
    os.environ.update(
        CONSOLE_ADMIN_USERNAME=DEMO_USER,
        CONSOLE_ADMIN_PASSWORD_HASH=console_auth.hash_password(DEMO_PASSWORD),
        CONSOLE_SESSION_SECRET="demo-secret-demo-secret-demo-secret-0123456789",
        CONSOLE_COOKIE_SECURE="false",
    )
    bus, store = EventBus(), FakeStore()
    log = MemoryConsoleLog(bus=bus)

    @asynccontextmanager
    async def lifespan(_app):
        await seed(log, store)
        yield

    app = FastAPI(title="Teacher console demo", lifespan=lifespan)
    app.dependency_overrides.update({
        teacher_console.get_log: lambda: log,
        teacher_console.get_store: lambda: store,
        teacher_console.get_sender: lambda: fake_sender,
        teacher_console.get_bus: lambda: bus,
    })
    app.include_router(teacher_console.router)
    app.mount("/console/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="console-static")

    @app.get("/console", include_in_schema=False)
    @app.get("/console/", include_in_schema=False)
    async def page():
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            return JSONResponse(status_code=404, content={"detail": "Chưa có giao diện"})
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.post("/demo/student")
    async def student_message(body: StudentMessage):
        await log.append(body.psid, "student", body.text)
        if body.handoff:
            store.status[body.psid] = "pending"
            await log.set_status(body.psid, "pending")
        return {"ok": True}

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"Console demo: http://{HOST}:{PORT}/console  (user: {DEMO_USER} / password: {DEMO_PASSWORD})")
    uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning")
