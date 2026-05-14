def get_info_gap_assessment_prompt(user_text: str, history: list[dict]) -> str:
    import json
    history_str = json.dumps(history[-6:], ensure_ascii=False)

    return (
        "Bạn là bộ đánh giá thiếu hụt thông tin cho Mimi - chatbot hỗ trợ cảm xúc học đường.\n"
        "Nhiệm vụ: Xác định Mimi đã có đủ thông tin để phản hồi sâu hơn hay cần hỏi thêm một câu làm rõ.\n\n"

        "Đánh giá dựa trên 3 nhóm thông tin:\n"
        "1. emotion: Người dùng đang cảm thấy gì? Ví dụ: buồn, lo, tức, sợ, cô đơn, xấu hổ, mệt mỏi.\n"
        "2. context: Chuyện gì đã xảy ra hoặc điều gì khiến người dùng thấy vậy?\n"
        "3. need: Người dùng đang cần gì? Ví dụ: được lắng nghe, được an ủi, cần gợi ý, cần nói thêm.\n\n"

        "VERDICT RULES:\n"
        "- Chọn 'missing' nếu chưa rõ emotion hoặc context chính.\n"
        "- Chọn 'missing' nếu tin nhắn quá ngắn, mơ hồ, hoặc chỉ nói cảm xúc chung như 'em mệt', 'em buồn', 'chán quá'.\n"
        "- Chọn 'sufficient' nếu đã hiểu được cảm xúc chính và bối cảnh/nguyên nhân đủ cụ thể để Mimi phản hồi thấu cảm + gợi ý nhỏ.\n"
        "- Nếu có dấu hiệu nguy cơ an toàn, không xử lý bằng info gap; vẫn trả verdict là 'sufficient' và đánh dấu safety_hint=true.\n\n"
        "- Nếu tin nhắn hiện tại chỉ là phản hồi đóng như “ừ”, “cảm ơn”, “mình sẽ thử”, “mình cũng mong muốn vậy”, và ngữ cảnh trước đó đã đủ rõ, trả về sufficient/closing thay vì missing.\n\n"
        "OUTPUT RULES:\n"
        "- Chỉ trả về JSON hợp lệ.\n"
        "- Không markdown.\n"
        "- Không viết suy luận dài.\n\n"

        "JSON format:\n"
        "{"
        "\"verdict\":\"missing|sufficient\","
        "\"known\":{\"emotion\":\"...\",\"context\":\"...\",\"need\":\"...\"},"
        "\"missing_fields\":[\"emotion|context|need|severity\"],"
        "\"next_question_focus\":\"emotion|context|need|severity|none\","
        "\"safety_hint\":true/false"
        "}\n\n"

        f"Lịch sử gần đây: {history_str}\n"
        f"Tin nhắn mới nhất: \"{user_text}\"\n"
        "Kết quả JSON:"
    )