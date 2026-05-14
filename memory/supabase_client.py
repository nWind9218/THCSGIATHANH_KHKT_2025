"""Supabase PostgreSQL client for long-term memory and knowledge base operations."""

import json
import logging
import os
from datetime import datetime, timezone

import httpx

from utils.embeddings import embed_text, vector_literal
from utils.database import exec_query, fetchall, get_pg_connection

logger = logging.getLogger(__name__)


def _is_render_env() -> bool:
    return bool(os.getenv("RENDER"))


def _render_context() -> dict[str, str]:
    return {
        "service": os.getenv("RENDER_SERVICE_NAME", ""),
        "instance": os.getenv("RENDER_INSTANCE_ID", ""),
        "commit": os.getenv("RENDER_GIT_COMMIT", ""),
    }


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


def _mask_api_key(api_key: str) -> str:
    """Mask API key in logs while still allowing quick validation."""
    if not api_key:
        return "<empty>"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}***{api_key[-4:]}"


def _mask_email_list(emails: list[str]) -> list[str]:
    return [_mask_email(email) for email in emails]


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


async def search_student_knowledge_kb(query: str, top_k: int = 3) -> str:
    """Search student counseling response knowledge by user intent."""
    if not query:
        return ""

    await get_pg_connection()
    embedding = await embed_text(query)
    try:
        rows = await fetchall(
            """
            SELECT user_intent, response_mode, should_do, should_not_do, sample_responses,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM student_knowledge
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vector_literal(embedding),
            top_k,
        )
    except Exception as exc:
        logger.warning("student_knowledge lookup unavailable: %s", exc)
        return ""
    if not rows:
        return ""

    chunks: list[str] = []
    for row in rows:
        row_dict = dict(row) if not isinstance(row, dict) else row
        chunks.append(
            (
                f"intent: {row_dict.get('user_intent', '')}\n"
                f"response_mode: {row_dict.get('response_mode', '')}\n"
                f"should_do: {row_dict.get('should_do', '')}\n"
                f"should_not_do: {row_dict.get('should_not_do', '')}\n"
                f"sample_responses: {row_dict.get('sample_responses', '')}"
            ).strip()
        )
    return "\n\n---\n\n".join(chunks)


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
    resend_api_key = os.getenv("RESEND_API_KEY", "").strip()
    resend_from_email = os.getenv("RESEND_FROM_EMAIL", "").strip()
    resend_base_url = os.getenv("RESEND_API_BASE_URL", "https://api.resend.com").strip().rstrip("/")
    resend_timeout = os.getenv("RESEND_REQUEST_TIMEOUT", "15").strip()
    recipients = os.getenv("EMERGENCY_EMAIL_RECIPIENTS", "").strip()
    resend_diag_enabled = os.getenv("RESEND_DIAG", "true").strip().lower() in {"1", "true", "yes", "on"}

    logger.info(f"Attempting to send emergency email for {user_id}")
    logger.info(
        "[RESEND_DIAG] Config base_url=%s from_email=%s api_key=%s recipients_count=%s diag_enabled=%s",
        resend_base_url,
        _mask_email(resend_from_email),
        _mask_api_key(resend_api_key),
        len([x for x in recipients.split(",") if x.strip()]),
        resend_diag_enabled,
    )

    if not resend_api_key or not resend_from_email or not recipients:
        logger.warning(
            f"Email NOT sent: Configuration missing for {user_id}. "
            f"RESEND_API_KEY: {'set' if resend_api_key else 'MISSING'}, "
            f"RESEND_FROM_EMAIL: {'set' if resend_from_email else 'MISSING'}, "
            f"RECIPIENTS: {'set' if recipients else 'MISSING'}"
        )
        return False

    target_list = [x.strip() for x in recipients.split(",") if x.strip()]
    if not target_list:
        logger.warning(f"Email NOT sent for {user_id}: Recipients list is empty after parsing.")
        return True

    logger.info(
        "[RESEND_DIAG] Parsed recipients user_id=%s recipients=%s",
        user_id,
        _mask_email_list(target_list),
    )

    if _is_render_env():
        logger.info("[RENDER_DIAG] Resend send on Render context=%s", _render_context())

    try:
        resend_timeout_num = float(resend_timeout)
    except ValueError:
        logger.error("[RESEND_DIAG] Invalid RESEND_REQUEST_TIMEOUT value: %s", resend_timeout)
        return False

    try:
        subject = f"🔴 [KHẨN CẤP] HỌC SINH CẦN TRỢ GIÚP - ID: {user_id}"

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

        endpoint = f"{resend_base_url}/emails"
        payload = {
            "from": resend_from_email,
            "to": target_list,
            "subject": subject,
            "text": text_body,
            "html": html_body,
        }
        headers = {
            "Authorization": f"Bearer {resend_api_key}",
            "Content-Type": "application/json",
        }

        if resend_diag_enabled:
            logger.info(
                "[RESEND_DIAG] Starting API send endpoint=%s timeout_s=%s recipients=%s",
                endpoint,
                resend_timeout_num,
                len(target_list),
            )

        async with httpx.AsyncClient(timeout=resend_timeout_num) as client:
            response = await client.post(endpoint, json=payload, headers=headers)

            if response.status_code in {200, 201, 202}:
                response_json = response.json() if response.content else {}
                logger.info(
                    "[RESEND_DIAG] API send success status=%s email_id=%s mode=batch",
                    response.status_code,
                    response_json.get("id"),
                )
                logger.info(f"Emergency email SUCCESS for {user_id}")
                return True

            logger.error(
                "[RESEND_DIAG] API send FAILED status=%s body=%s mode=batch",
                response.status_code,
                response.text[:500],
            )

            if len(target_list) == 1:
                return False

            logger.warning(
                "[RESEND_DIAG] Retrying as per-recipient sends user_id=%s recipients=%s",
                user_id,
                _mask_email_list(target_list),
            )

            success_recipients: list[str] = []
            failed_recipients: list[str] = []

            for recipient in target_list:
                single_payload = dict(payload)
                single_payload["to"] = [recipient]
                try:
                    single_response = await client.post(endpoint, json=single_payload, headers=headers)
                    if single_response.status_code in {200, 201, 202}:
                        success_recipients.append(recipient)
                        logger.info(
                            "[RESEND_DIAG] Per-recipient send success recipient=%s status=%s",
                            _mask_email(recipient),
                            single_response.status_code,
                        )
                    else:
                        failed_recipients.append(recipient)
                        logger.error(
                            "[RESEND_DIAG] Per-recipient send failed recipient=%s status=%s body=%s",
                            _mask_email(recipient),
                            single_response.status_code,
                            single_response.text[:300],
                        )
                except Exception as per_recipient_error:
                    failed_recipients.append(recipient)
                    logger.error(
                        "[RESEND_DIAG] Per-recipient send exception recipient=%s error=%s: %s",
                        _mask_email(recipient),
                        type(per_recipient_error).__name__,
                        per_recipient_error,
                    )

            logger.info(
                "[RESEND_DIAG] Per-recipient retry result success=%s failed=%s",
                _mask_email_list(success_recipients),
                _mask_email_list(failed_recipients),
            )

            if success_recipients:
                logger.info("Emergency email PARTIAL SUCCESS for %s", user_id)
                return True
            return False
    except Exception as e:
        logger.error(
            "[RESEND_DIAG] Failed to send emergency email for %s endpoint=%s error=%s: %s",
            user_id,
            f"{resend_base_url}/emails",
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
