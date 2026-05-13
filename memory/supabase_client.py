"""Supabase PostgreSQL client for long-term memory and knowledge base operations."""

import asyncio
import json
import logging
import os
import socket
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from utils.embeddings import embed_text, vector_literal
from utils.database import exec_query, fetchall, get_pg_connection

logger = logging.getLogger(__name__)


def _mask_email(email: str) -> str:
    """Mask email in logs to avoid leaking full address."""
    if not email or "@" not in email:
        return "<empty>"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = local[:2] + "*" * (len(local) - 2)
    return f"{masked_local}@{domain}"


async def _probe_smtp_outbound(host: str, port: int, timeout_seconds: float = 5.0) -> None:
    """Run DNS and raw TCP probe before SMTP auth/send for outbound diagnostics."""
    started_at = time.perf_counter()
    try:
        addr_infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
        resolved = sorted({addr[4][0] for addr in addr_infos if addr and len(addr) > 4 and addr[4]})
        logger.info(
            "[SMTP_DIAG] DNS resolve success host=%s port=%s resolved_ips=%s",
            host,
            port,
            resolved,
        )
    except Exception as exc:
        logger.error(
            "[SMTP_DIAG] DNS resolve FAILED host=%s port=%s error=%s: %s",
            host,
            port,
            type(exc).__name__,
            exc,
        )
        raise

    writer = None
    try:
        connect_started = time.perf_counter()
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout_seconds)
        _ = reader
        connect_ms = int((time.perf_counter() - connect_started) * 1000)
        total_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "[SMTP_DIAG] TCP connect success host=%s port=%s connect_ms=%s total_probe_ms=%s",
            host,
            port,
            connect_ms,
            total_ms,
        )
    except Exception as exc:
        logger.error(
            "[SMTP_DIAG] TCP connect FAILED host=%s port=%s timeout_s=%s error=%s: %s",
            host,
            port,
            timeout_seconds,
            type(exc).__name__,
            exc,
        )
        raise
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()


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
    # Try both common names for SMTP user
    smtp_user = (os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER") or "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    recipients = os.getenv("EMERGENCY_EMAIL_RECIPIENTS", "").strip()
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = os.getenv("SMTP_PORT", "587")
    smtp_diag_enabled = os.getenv("SMTP_OUTBOUND_DIAG", "true").strip().lower() in {"1", "true", "yes", "on"}

    logger.info(f"Attempting to send emergency email for {user_id}")
    logger.info(
        "[SMTP_DIAG] Config host=%s port=%s user=%s password=%s recipients_count=%s diag_enabled=%s",
        smtp_host,
        smtp_port,
        _mask_email(smtp_user),
        "SET" if smtp_password else "MISSING",
        len([x for x in recipients.split(",") if x.strip()]),
        smtp_diag_enabled,
    )

    if not smtp_user or not smtp_password or not recipients:
        logger.warning(
            f"Email NOT sent: Configuration missing for {user_id}. "
            f"SMTP_USER: {'set' if smtp_user else 'MISSING'}, "
            f"SMTP_PASSWORD: {'set' if smtp_password else 'MISSING'}, "
            f"RECIPIENTS: {'set' if recipients else 'MISSING'}"
        )
        return False

    target_list = [x.strip() for x in recipients.split(",") if x.strip()]
    if not target_list:
        logger.warning(f"Email NOT sent for {user_id}: Recipients list is empty after parsing.")
        return True

    try:
        smtp_port_num = int(smtp_port)
    except ValueError:
        logger.error("[SMTP_DIAG] Invalid SMTP_PORT value: %s", smtp_port)
        return False

    try:
        if smtp_diag_enabled:
            await _probe_smtp_outbound(smtp_host, smtp_port_num)

        message = MIMEMultipart("alternative")
        # Tiêu đề khẩn cấp, viết hoa và có biểu tượng để tránh bị coi là rác
        message["Subject"] = f"🔴 [KHẨN CẤP] HỌC SINH CẦN TRỢ GIÚP - ID: {user_id}"
        message["From"] = smtp_user
        message["To"] = ", ".join(target_list)

        # Nội dung văn bản thuần (Fallback)
        text_body = (
            f"⚠️ CẢNH BÁO KHẨN CẤP ⚠️\n\n"
            f"User ID: {user_id}\n"
            f"Nội dung: {raw_message}\n"
            f"Thời gian: {datetime.now(timezone.utc).isoformat()}\n\n"
            f"Lệnh kích hoạt lại Bot: #resume {user_id}"
        )

        # Nội dung HTML (Rich Format)
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; border: 2px solid #e74c3c; border-radius: 10px; overflow: hidden;">
                <div style="background-color: #e74c3c; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0;">⚠️ CẢNH BÁO KHẨN CẤP</h1>
                </div>
                <div style="padding: 20px;">
                    <p>Chào thầy/cô, hệ thống Mimi vừa phát hiện một tình huống khẩn cấp cần sự can thiệp của con người.</p>
                    
                    <div style="background-color: #f9f9f9; border-left: 5px solid #e74c3c; padding: 15px; margin: 20px 0;">
                        <strong>Thông tin học sinh:</strong><br>
                        • <b>User ID (PSID):</b> <code style="background: #eee; padding: 2px 5px;">{user_id}</code><br>
                        • <b>Thời gian:</b> {datetime.now(timezone.utc).strftime('%H:%M:%S %d/%m/%Y')} (UTC)
                    </div>

                    <div style="margin: 20px 0;">
                        <strong>Nội dung tin nhắn cuối của học sinh:</strong><br>
                        <blockquote style="font-style: italic; color: #555; border-left: 3px solid #ccc; padding-left: 15px; margin-left: 0;">
                            "{raw_message}"
                        </blockquote>
                    </div>

                    <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">

                    <div style="background-color: #e8f4fd; border-radius: 5px; padding: 15px;">
                        <h3 style="margin-top: 0; color: #2980b9;">🛠 Hướng dẫn xử lý</h3>
                        <p>1. Thầy/cô vui lòng kiểm tra ngay hộp thư Fanpage để hỗ trợ học sinh.</p>
                        <p>2. Khi đã hỗ trợ xong và muốn <b>Bot quay lại hoạt động</b>, hãy nhắn tin cho Fanpage cú pháp sau:</p>
                        <div style="text-align: center; margin: 15px 0;">
                            <code style="display: inline-block; background-color: #2c3e50; color: #ecf0f1; padding: 10px 20px; border-radius: 5px; font-size: 1.2em; font-weight: bold;">
                                #resume {user_id}
                            </code>
                        </div>
                    </div>
                </div>
                <div style="background-color: #f4f4f4; color: #888; padding: 10px; text-align: center; font-size: 0.8em;">
                    Đây là email tự động từ hệ thống Mimi Counseling Bot. Vui lòng không trả lời email này.
                </div>
            </div>
        </body>
        </html>
        """
        
        message.attach(MIMEText(text_body, "plain", "utf-8"))
        message.attach(MIMEText(html_body, "html", "utf-8"))

        logger.info("[SMTP_DIAG] Starting SMTP send host=%s port=%s start_tls=%s timeout=%s", smtp_host, smtp_port_num, True, 15)
        send_response = await aiosmtplib.send(
            message,
            hostname=smtp_host,
            port=smtp_port_num,
            username=smtp_user,
            password=smtp_password,
            start_tls=True,
            timeout=15,
        )
        logger.info("[SMTP_DIAG] SMTP send completed host=%s port=%s", smtp_host, smtp_port_num)
        logger.debug("[SMTP_DIAG] SMTP provider response: %s", send_response)
        logger.info(f"Emergency email SUCCESS for {user_id}")
        return True
    except Exception as e:
        logger.error(
            "[SMTP_DIAG] Failed to send emergency email for %s host=%s port=%s error=%s: %s",
            user_id,
            smtp_host,
            smtp_port,
            type(e).__name__,
            e,
        )
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
