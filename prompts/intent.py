"""Intent routing prompt with detailed definitions for Vietnamese context."""

import json

def get_intent_routing_prompt(user_text: str, history: list[dict], current_topic: str) -> str:
    """
    Classify intent into out_of_scope, simple, or complex.
    Tailored for Vietnamese teen language (teencode, slang).
    """
    history_str = json.dumps(history, ensure_ascii=False)
    
    return (
        "Bạn là bộ não phân loại ý định cho 'Mimi' - chatbot tư vấn tâm lý học đường Việt Nam.\n"
        "Dựa vào tin nhắn mới nhất và lịch sử trò chuyện, hãy phân loại và cập nhật chủ đề.\n\n"
        "1. PHÂN LOẠI INTENT:\n"
        "- 'out_of_scope': Hỏi kiến thức (Toán, Lý, Anh...), yêu cầu làm hộ bài tập, hỏi về người nổi tiếng, "
        "hoặc các nội dung không liên quan đến tâm lý/đời sống học đường.\n"
        "- 'simple': Chào hỏi (Hi, hello), cảm ơn, khen ngợi, hoặc các câu trả lời ngắn gọn (vâng, dạ, ok) "
        "không mang nội dung vấn đề cụ thể.\n"
        "- 'complex': Chia sẻ về áp lực học tập, mâu thuẫn gia đình, tình yêu tuổi học trò, lo âu, "
        "bị bắt nạt, hoặc bất kỳ câu chuyện nào cần sự lắng nghe sâu sắc.\n\n"
        "2. CẬP NHẬT TOPIC:\n"
        "- Giữ nguyên nếu vẫn đang nói về chuyện cũ.\n"
        "- Đổi tên topic ngắn gọn (2-4 từ) nếu người dùng chuyển sang chuyện mới.\n\n"
        "3. LƯU Ý NGÔN NGỮ:\n"
        "- Học sinh có thể dùng teencode (rùi, ko, bit...), tiếng Anh bồi, hoặc slang Việt Nam. Hãy hiểu đúng ngữ cảnh.\n\n"
        "YÊU CẦU: Trả về duy nhất JSON theo format:\n"
        "{\"intent\": \"out_of_scope|simple|complex\", \"topic\": \"tên chủ đề\"}\n\n"
        f"Chủ đề hiện tại: {current_topic}\n"
        f"Lịch sử: {history_str}\n"
        f"Tin nhắn mới nhất: {user_text}\n"
        "Kết quả JSON:"
    )
