"""Safety gateway prompt for detecting emergency situations."""


def get_safety_check_prompt(user_text: str) -> str:
    """
    Binary safety classifier: does this message indicate self-harm or violence?
    Returns: Prompt for LLM to respond with YES or NO
    """
    return (
        "Ban la bo loc an toan cho chatbot tu van hoc sinh Viet Nam. "
        "Chi tra loi YES hoac NO.\n"
        "Tin nhan sau co dau hieu tu lam hai minh hoac lam hai nguoi khac khong?\n"
        f"Tin nhan: {user_text}"
    )
