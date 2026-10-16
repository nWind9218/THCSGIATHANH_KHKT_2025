"""Emergency logging and email notification used by the Facebook/v2 runtime."""

import json
import logging
import os
from datetime import datetime, timezone

import httpx

from utils.database import exec_query, get_pg_connection

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
