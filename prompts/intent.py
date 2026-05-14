import json

def get_intent_routing_prompt(user_text: str, history: list[dict], current_topic: str) -> str:
    """
    Classify student message intent for Mimi.
    Output stable JSON for routing.
    """
    history_str = json.dumps(history[-6:], ensure_ascii=False)

    return (
        "Bạn là bộ phân loại ý định cho Mimi - chatbot hỗ trợ cảm xúc học đường cho học sinh Việt Nam.\n"
        "Nhiệm vụ của bạn là xác định mục đích chính của tin nhắn mới nhất, dựa trên cả lịch sử trò chuyện.\n\n"

        "QUAN TRỌNG:\n"
        "- Ưu tiên hiểu 'core emotional intent' hơn là bề mặt câu chữ.\n"
        "- Nếu tin nhắn có yếu tố cảm xúc, áp lực, tổn thương, né tránh, cô đơn, tự ti, bị bắt nạt, gia đình, tình yêu, trường lớp → chọn complex.\n"
        "- Nếu người dùng hỏi kiến thức nhưng có kèm cảm xúc cá nhân, vẫn chọn complex.\n"
        "- Các câu ngắn như 'dạ', 'ừm', 'không biết nữa', 'mệt' phải được hiểu theo ngữ cảnh lịch sử.\n"
        "- Nếu tin nhắn mới nhất chỉ là phản hồi đồng thuận nhẹ sau khi Mimi đã tư vấn xong như 'ừm', 'yub', 'mình cũng hi vọng vậy', 'mình sẽ thử', 'cảm ơn Mimi', 'nghe cũng ổn', thì chọn simple, không chọn complex, trừ khi có thêm vấn đề cảm xúc mới hoặc dấu hiệu nguy hiểm.\n\n"

        "INTENT DEFINITIONS:\n"
        "1. out_of_scope:\n"
        "- Hỏi kiến thức, bài tập, người nổi tiếng, lập trình, thông tin chung.\n"
        "- Yêu cầu làm hộ bài hoặc giải bài.\n"
        "- Không có dấu hiệu chia sẻ cảm xúc cá nhân.\n\n"

        "2. simple:\n"
        "- Chào hỏi, cảm ơn, tạm biệt, khen Mimi.\n"
        "- Tán gẫu nhẹ không có vấn đề cảm xúc rõ ràng.\n"
        "- Câu trả lời ngắn chỉ mang tính xác nhận, khi lịch sử trước đó cũng không có vấn đề tâm lý cụ thể.\n\n"
        "- Phản hồi đóng sau tư vấn: người dùng đồng thuận nhẹ, cảm ơn, nói sẽ thử, hoặc thể hiện đã tiếp nhận lời khuyên mà không mở thêm vấn đề mới.\n"
        "3. complex:\n"
        "- Áp lực học tập, điểm kém, sợ đi học, không muốn đi học.\n"
        "- Bị bắt nạt, bị trêu, bị cô lập, toxic friendship.\n"
        "- Mâu thuẫn gia đình, bị so sánh, không được lắng nghe.\n"
        "- Lo âu, buồn bã, cô đơn, tự ti, xấu hổ, thất vọng, kiệt sức.\n"
        "- Tình yêu tuổi học trò, chia tay, bị từ chối.\n"
        "- Bất kỳ câu chuyện nào cần Mimi lắng nghe sâu hơn.\n\n"

        "TOPIC RULES:\n"
        "- Giữ current_topic nếu tin nhắn vẫn tiếp nối chuyện cũ.\n"
        "- Đổi topic nếu người dùng chuyển sang vấn đề mới.\n"
        "- Topic phải ngắn, 2-4 từ, bằng tiếng Việt không dấu hoặc snake_case.\n"
        "- Ưu tiên một trong các topic sau nếu phù hợp:\n"
        "  school_stress, school_avoidance, bad_grades, bullying, loneliness, friendship, body_image, family_conflict, comparison, love, anxiety, sadness, casual, out_of_scope.\n\n"

        "OUTPUT RULES:\n"
        "- Chỉ trả về JSON hợp lệ.\n"
        "- Không giải thích.\n"
        "- Không thêm markdown.\n\n"

        "JSON format:\n"
        "{\"intent\":\"out_of_scope|simple|complex\",\"topic\":\"topic_name\",\"core_issue\":\"short Vietnamese summary\"}\n\n"

        f"Chủ đề hiện tại: {current_topic}\n"
        f"Lịch sử gần đây: {history_str}\n"
        f"Tin nhắn mới nhất: \"{user_text}\"\n"
        "Kết quả JSON:"
    )