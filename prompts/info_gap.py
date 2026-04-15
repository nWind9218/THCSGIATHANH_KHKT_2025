"""Info gap assessment prompt for detecting missing information."""


def get_info_gap_assessment_prompt(user_text: str) -> str:
    """
    Chain-of-Thought assessment: does the bot have enough info to give good advice?
    Returns: JSON with {"scratchpad": [...], "verdict": "missing|sufficient"}
    """
    return (
        "Danh gia thong tin hien tai co du de tu van chua. "
        "Tra ve JSON: {\"scratchpad\": [\"...\"], \"verdict\": \"missing|sufficient\"}.\n"
        "Tieu chi sufficient: hieu cam xuc cot loi + biet it nhat 1 boi canh/nguyen nhan.\n"
        f"User message: {user_text}"
    )
