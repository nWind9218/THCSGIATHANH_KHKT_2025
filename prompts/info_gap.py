"""Info gap assessment prompt to detect if more context is needed."""

def get_info_gap_assessment_prompt(user_text: str, history: list[dict]) -> str:
    """
    Chain-of-Thought assessment in Vietnamese.
    Checks if 'Mimi' understands: 1. Core emotion + 2. Context/Cause.
    """
    import json
    history_str = json.dumps(history[-5:], ensure_ascii=False)
    
    return (
        "Bạn là chuyên gia phân tích hội thoại tâm lý. "
        "Nhiệm vụ: Đánh giá xem thông tin hiện tại đã đủ để đưa ra lời khuyên/hỗ trợ sâu sắc chưa.\n\n"
        "TIÊU CHÍ 'SUFFICIENT' (Đủ):\n"
        "1. Đã xác định được cảm xúc chủ đạo của học sinh (buồn, lo lắng, tức giận...).\n"
        "2. Đã biết ít nhất một bối cảnh hoặc nguyên nhân cụ thể (do bài kiểm tra, do cãi nhau với mẹ...).\n\n"
        "Nếu thiếu 1 trong 2, hãy đánh giá là 'missing'.\n\n"
        "YÊU CẦU: Trả về duy nhất JSON:\n"
        "{\"scratchpad\": [\"Phân tích ngắn gọn về những gì đã biết và chưa biết\"], \"verdict\": \"missing|sufficient\"}\n\n"
        f"Lịch sử gần đây: {history_str}\n"
        f"Tin nhắn mới nhất: {user_text}\n"
        "Kết quả JSON:"
    )
