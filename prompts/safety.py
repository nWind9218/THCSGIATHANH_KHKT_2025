"""Safety gateway prompt for detecting emergency situations in Vietnamese context."""

def get_safety_check_prompt(user_text: str) -> str:
    """
    Binary safety classifier for Vietnamese students.
    Detects: Self-harm, suicide, violence, severe abuse, or intent to hurt others.
    """
    return (
        "Bạn là chuyên gia lọc nội dung an toàn cho chatbot tâm lý học đường tại Việt Nam. "
        "Nhiệm vụ: Xác định xem tin nhắn của học sinh có chứa dấu hiệu nguy hiểm khẩn cấp không.\n\n"
        "Tiêu chí 'YES' (Nguy hiểm):\n"
        "- Có ý định hoặc kế hoạch tự làm hại bản thân, tự tử.\n"
        "- Có ý định gây bạo lực, làm hại người khác.\n"
        "- Đang ở trong tình huống bị lạm dụng, bạo hành nghiêm trọng cần can thiệp ngay.\n"
        "- Sử dụng các từ ngữ tuyệt vọng cực độ (muốn biến mất, không muốn sống nữa...).\n\n"
        "Tiêu chí 'NO': Các vấn đề buồn bã, áp lực học tập, mâu thuẫn bạn bè thông thường.\n\n"
        "QUY TẮC: Chỉ trả lời duy nhất một từ 'YES' hoặc 'NO'. Không giải thích gì thêm.\n"
        f"Tin nhắn: \"{user_text}\"\n"
        "Kết quả:"
    )
