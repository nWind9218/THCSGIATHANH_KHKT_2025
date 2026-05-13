import asyncio
import os
import logging
from dotenv import load_dotenv
from memory.supabase_client import send_emergency_email

# Thiết lập logging để thấy lỗi nếu có
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_emergency_email():
    load_dotenv(override=True)
    
    print("--- KIỂM TRA CẤU HÌNH SMTP ---")
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_host = os.getenv("SMTP_HOST")
    recipients = os.getenv("EMERGENCY_EMAIL_RECIPIENTS")
    
    if not all([smtp_user, smtp_host, recipients]):
        print("❌ LỖI: Thiếu cấu hình SMTP trong file .env")
        print(f"SMTP_USERNAME: {'OK' if smtp_user else 'Missing'}")
        print(f"SMTP_HOST: {'OK' if smtp_host else 'Missing'}")
        print(f"EMERGENCY_EMAIL_RECIPIENTS: {'OK' if recipients else 'Missing'}")
        return

    print(f"Đang gửi thử email từ: {smtp_user}")
    print(f"Đến danh sách: {recipients}")
    
    user_id = "test_user_123"
    summary = "CẢNH BÁO GIẢ LẬP: Phát hiện dấu hiệu tự hại trong bài kiểm tra hệ thống."
    raw_message = "Mình cảm thấy rất bế tắc và không muốn tiếp tục nữa..."
    
    success = await send_emergency_email(user_id, summary, raw_message)
    
    if success:
        print("\n✅ THÀNH CÔNG: Email đã được gửi đi!")
        print("Hãy kiểm tra hộp thư (bao gồm cả thư rác/spam) của người nhận.")
    else:
        print("\n❌ THẤT BẠI: Không thể gửi email. Vui lòng kiểm tra log để biết chi tiết.")

if __name__ == "__main__":
    asyncio.run(test_emergency_email())
