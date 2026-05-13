"""Save memory and gating prompts for Vietnamese context."""

def get_memory_gate_prompt(user_text: str) -> str:
    """
    Identify durable information to store in long-term memory.
    Focus: Family context, recurring issues, personality traits, specific interests.
    """
    return (
        "Bạn là trợ lý quản lý trí nhớ cho chatbot 'Mimi'.\n"
        "Nhiệm vụ: Chắt lọc những thông tin quan trọng về học sinh từ tin nhắn mới nhất để ghi nhớ lâu dài.\n\n"
        "THÔNG TIN CẦN LƯU:\n"
        "- Hoàn cảnh gia đình, tên bạn bè/người thân.\n"
        "- Những vấn đề lặp đi lặp lại (VD: luôn sợ bị điểm kém, hay cãi nhau với mẹ).\n"
        "- Sở thích, thế mạnh hoặc điểm yếu của học sinh.\n"
        "- Các sự kiện quan trọng vừa xảy ra.\n\n"
        "YÊU CẦU: Trả về duy nhất JSON format:\n"
        "{\"should_store\": true|false, \"memory\": \"Tóm tắt thông tin cần nhớ bằng tiếng Việt\"}\n\n"
        f"Tin nhắn: \"{user_text}\"\n"
        "Kết quả JSON:"
    )
