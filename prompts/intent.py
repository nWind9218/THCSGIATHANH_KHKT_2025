"""Intent routing prompt for classifying user messages."""

import json


def get_intent_routing_prompt(user_text: str, history: list[dict], current_topic: str) -> str:
    """
    Classify intent as: out_of_scope, simple, or complex
    Also update topic based on recent messages
    Returns: JSON with {"intent": "...", "topic": "..."}
    """
    return (
        "Phan loai message hoc sinh theo 3 nhan: out_of_scope, simple, complex. "
        "Dong thoi cap nhat topic hien tai dua tren 3-5 tin gan nhat. "
        "Tra ve JSON duy nhat theo schema: "
        '{"intent":"out_of_scope|simple|complex","topic":"..."}.\n'
        f"Current topic: {current_topic}\n"
        f"History: {json.dumps(history, ensure_ascii=False)}\n"
        f"Latest: {user_text}"
    )
