"""Supabase PostgreSQL client for long-term memory and knowledge base operations."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from utils.embeddings import embed_text, vector_literal
from utils.database import exec_query, fetchall, get_pg_connection

logger = logging.getLogger(__name__)


async def search_psychology_kb(query: str, top_k: int = 3) -> str:
    """Search psychology knowledge base for general guidelines."""
    if not query:
        return ""

    await get_pg_connection()
    embedding = await embed_text(query)
    rows = await fetchall(
        """
        SELECT content, 1 - (embedding <=> $1::vector) AS similarity
        FROM psychology_kb
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        vector_literal(embedding),
        top_k,
    )
    if not rows:
        return ""

    chunks: list[str] = []
    for row in rows:
        content = row.get("content") if isinstance(row, dict) else row[0]
        if content:
            chunks.append(content)
    return "\n\n".join(chunks)


async def search_user_memory_kb(user_id: str, query: str, top_k: int = 3) -> str:
    """Search user-specific long-term memory."""
    if not user_id or not query:
        return ""

    await get_pg_connection()
    embedding = await embed_text(query)
    rows = await fetchall(
        """
        SELECT content, 1 - (embedding <=> $1::vector) AS similarity
        FROM user_memory_kb
        WHERE user_id = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        """,
        vector_literal(embedding),
        user_id,
        top_k,
    )
    if not rows:
        return ""

    chunks: list[str] = []
    for row in rows:
        content = row.get("content") if isinstance(row, dict) else row[0]
        if content:
            chunks.append(content)
    return "\n\n".join(chunks)


async def upsert_user_memory_chunk(user_id: str, content: str) -> None:
    """Add or update long-term memory chunk for user."""
    if not user_id or not content:
        return

    await get_pg_connection()
    embedding = await embed_text(content)
    await exec_query(
        """
        INSERT INTO user_memory_kb (user_id, content, embedding)
        VALUES ($1, $2, $3::vector)
        """,
        user_id,
        content,
        vector_literal(embedding),
    )
    logger.info(f"Upserted memory chunk for {user_id}")


async def log_emergency(user_id: str, summary: str) -> None:
    """Log emergency incident to database."""
    await get_pg_connection()
    await exec_query(
        """
        INSERT INTO emergency_logs (user_id, summary, created_at)
        VALUES ($1, $2, NOW())
        """,
        user_id,
        summary,
    )
    logger.warning(f"Emergency logged for {user_id}: {summary}")


async def send_emergency_email(user_id: str, summary: str, raw_message: str) -> bool:
    """Send emergency notification email to admin."""
    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    recipients = os.getenv("EMERGENCY_EMAIL_RECIPIENTS", "").strip()

    if not smtp_user or not smtp_password or not recipients:
        logger.debug("Email not configured, skipping emergency email")
        return True

    target_list = [x.strip() for x in recipients.split(",") if x.strip()]
    if not target_list:
        return True

    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = "[Mimi] Emergency handoff required"
        message["From"] = smtp_user
        message["To"] = ", ".join(target_list)

        body = (
            f"User: {user_id}\n"
            f"Summary: {summary}\n"
            f"Last message: {raw_message}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
        message.attach(MIMEText(body, "plain", "utf-8"))

        await aiosmtplib.send(
            message,
            hostname=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.getenv("SMTP_PORT", "587")),
            username=smtp_user,
            password=smtp_password,
            start_tls=True,
        )
        logger.info(f"Emergency email sent for {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to send emergency email: {e}")
        return False


async def notify_human_admin(user_id: str, summary: str, raw_message: str) -> bool:
    """
    Comprehensive emergency notification: log + email + pub/sub.
    Triggers when emergency is detected.
    """
    # 1. Log to database
    await log_emergency(user_id, summary)

    # 2. Send email
    await send_emergency_email(user_id, summary, raw_message)

    # 3. Publish to admin channel (real-time)
    from memory.redis_client import publish_admin_alert
    await publish_admin_alert(user_id, {
        "type": "emergency",
        "summary": summary,
        "raw_message": raw_message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return True


def infer_ocean_increment(text: str) -> dict[str, int]:
    """Infer OCEAN personality increment from user message."""
    msg = (text or "").lower()
    inc = {"o": 0, "c": 0, "e": 0, "a": 0, "n": 0}

    if any(k in msg for k in ["muon thu", "sang tao", "y tuong", "kham pha"]):
        inc["o"] += 1
    if any(k in msg for k in ["ke hoach", "ky luat", "deadline", "hoc bai"]):
        inc["c"] += 1
    if any(k in msg for k in ["ban be", "noi chuyen", "ngoai khoa", "dong nguoi"]):
        inc["e"] += 1
    if any(k in msg for k in ["xin loi", "hoa giai", "quan tam", "thong cam"]):
        inc["a"] += 1
    if any(k in msg for k in ["lo", "so", "cang thang", "mat ngu", "ap luc"]):
        inc["n"] += 1

    return inc


async def update_user_longterm_style(ip_or_user_id: str, latest_message: str) -> None:
    """Update OCEAN personality scores and dominant style."""
    await get_pg_connection()
    inc = infer_ocean_increment(latest_message)

    await exec_query(
        """
        INSERT INTO user_longterm_memory (ip_address, o_quest, c_quest, e_quest, a_quest, n_quest)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (ip_address)
        DO UPDATE SET
            o_quest = user_longterm_memory.o_quest + EXCLUDED.o_quest,
            c_quest = user_longterm_memory.c_quest + EXCLUDED.c_quest,
            e_quest = user_longterm_memory.e_quest + EXCLUDED.e_quest,
            a_quest = user_longterm_memory.a_quest + EXCLUDED.a_quest,
            n_quest = user_longterm_memory.n_quest + EXCLUDED.n_quest
        """,
        ip_or_user_id,
        inc["o"],
        inc["c"],
        inc["e"],
        inc["a"],
        inc["n"],
    )

    await exec_query("SELECT sync_user_style($1)", ip_or_user_id)
    logger.debug(f"Updated OCEAN scores for {ip_or_user_id}")
