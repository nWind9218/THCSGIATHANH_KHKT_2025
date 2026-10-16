def get_memory_gate_prompt(user_text: str) -> str:
    """
    Decide whether user information is durable and useful enough
    for long-term memory in a school mental-health chatbot.
    """

    return (
        "Bạn là bộ lọc trí nhớ dài hạn cho Mimi - chatbot hỗ trợ cảm xúc học đường.\n\n"

        "Nhiệm vụ:\n"
        "Chỉ lưu những thông tin ổn định, lặp lại hoặc thật sự hữu ích cho việc hỗ trợ người dùng trong tương lai.\n\n"

        "NÊN LƯU:\n"
        "- Khó khăn lặp đi lặp lại: thường xuyên bị áp lực học tập, hay bị bắt nạt, luôn tự ti ngoại hình.\n"
        "- Hoàn cảnh ổn định: bố mẹ hay cãi nhau, sống xa gia đình, thường bị so sánh.\n"
        "- Sở thích hoặc điều khiến người dùng cảm thấy an toàn/vui vẻ.\n"
        "- Điểm mạnh hoặc điều người dùng tự hào.\n"
        "- Thói quen hoặc nỗi lo xuất hiện nhiều lần.\n\n"

        "KHÔNG NÊN LƯU:\n"
        "- Cảm xúc nhất thời: buồn hôm nay, mệt hôm nay.\n"
        "- Sự kiện nhỏ chỉ xảy ra một lần.\n"
        "- Chi tiết quá riêng tư hoặc nhạy cảm không cần thiết.\n"
        "- Tên đầy đủ, địa chỉ, thông tin nhận diện cá nhân.\n"
        "- Nội dung self-harm, tự tử hoặc trauma chi tiết.\n\n"

        "NGUYÊN TẮC:\n"
        "- Ưu tiên tối thiểu hóa dữ liệu.\n"
        "- Chỉ lưu nếu nó có ích cho việc hỗ trợ lâu dài.\n"
        "- Tóm tắt ngắn gọn, trung tính, không phán xét.\n"
        "- Không copy nguyên văn tin nhắn.\n"
        "- Nếu không chắc có nên lưu hay không, chọn false.\n\n"

        "OUTPUT RULES:\n"
        "- Chỉ trả về JSON hợp lệ.\n"
        "- Không markdown.\n"
        "- Không giải thích.\n\n"

        "JSON format:\n"
        "{"
        "\"should_store\": true|false,"
        "\"memory_type\":\"recurring_issue|family_context|preference|strength|other|none\","
        "\"memory\":\"Tóm tắt ngắn gọn bằng tiếng Việt\""
        "}\n\n"

        f"Tin nhắn mới nhất: \"{user_text}\"\n"
        "Kết quả JSON:"
    )