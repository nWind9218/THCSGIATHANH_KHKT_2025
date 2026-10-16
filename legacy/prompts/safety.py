def get_safety_check_prompt(user_text: str) -> str:
    """
    Safety classifier for Vietnamese school mental-health chatbot.
    Detects suicide risk, self-harm, violence, abuse, or severe hopelessness.
    """

    return (
        "Bạn là hệ thống phát hiện nguy cơ an toàn cho chatbot hỗ trợ tâm lý học sinh tại Việt Nam.\n\n"

        "Nhiệm vụ:\n"
        "Xác định xem tin nhắn có chứa dấu hiệu nguy cơ tâm lý nghiêm trọng hoặc nguy hiểm cần ưu tiên an toàn hay không.\n\n"

        "Trả lời 'YES' nếu tin nhắn có MỘT trong các dấu hiệu sau:\n"
        "- Muốn chết, tự tử, tự làm hại bản thân.\n"
        "- Không muốn tồn tại, muốn biến mất, muốn ngủ mãi không dậy.\n"
        "- Tuyệt vọng cực độ, mất hoàn toàn hy vọng sống.\n"
        "- Nói về việc không chịu nổi nữa, muốn kết thúc tất cả.\n"
        "- Có ý định làm hại người khác hoặc bạo lực nghiêm trọng.\n"
        "- Đang bị bạo hành, lạm dụng, đe dọa nghiêm trọng.\n"
        "- Các cách nói ẩn ý, nói giảm nói tránh về việc biến mất, kết thúc cuộc đời, tự hủy hoại.\n\n"

        "Ví dụ cần trả lời YES:\n"
        "- 'Em muốn biến mất luôn.'\n"
        "- 'Ngủ mãi chắc sẽ đỡ hơn.'\n"
        "- 'Không ai cần em nữa.'\n"
        "- 'Em không chịu nổi nữa rồi.'\n"
        "- 'Em muốn tự làm đau mình.'\n\n"

        "Trả lời 'NO' nếu chỉ là:\n"
        "- Buồn bã thông thường.\n"
        "- Áp lực học tập.\n"
        "- Căng thẳng bạn bè.\n"
        "- Than mệt nhưng không có dấu hiệu nguy hiểm nghiêm trọng.\n\n"

        "QUY TẮC:\n"
        "- Chỉ trả lời duy nhất: YES hoặc NO.\n"
        "- Không giải thích.\n"
        "- Nếu có dấu hiệu tuyệt vọng hoặc muốn biến mất, ưu tiên an toàn và trả lời YES.\n"
        "- Hiểu teencode, slang, cách nói ẩn dụ và cách nói gián tiếp của học sinh Việt Nam.\n\n"

        f"Tin nhắn: \"{user_text}\"\n"
        "Kết quả:"
    )
