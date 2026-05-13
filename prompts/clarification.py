"""Clarification question prompt for natural Vietnamese dialogue."""

import json

def get_clarification_prompt(scratchpad: list[str], user_text: str) -> str:
    """
    Generate a single, warm clarification question in Vietnamese.
    """
    return (
        "Dựa vào phân tích (scratchpad) và tin nhắn mới nhất, hãy đặt MỘT câu hỏi để học sinh chia sẻ thêm.\n\n"
        "QUY TẮC:\n"
        "- Chỉ 01 câu hỏi duy nhất.\n"
        "- Không giảng giải, không khuyên bảo ở bước này.\n"
        "- Tông giọng: Quan tâm, không giống đang thẩm vấn. Ưu tiên câu hỏi mở (Điều gì..., Như thế nào..., Từ khi nào...).\n"
        "- Xưng hô: Mimi - bạn.\n\n"
        f" Scratchpad: {json.dumps(scratchpad, ensure_ascii=False)}\n"
        f"Tin nhắn cuối: {user_text}\n"
        "Câu hỏi của Mimi:"
    )
