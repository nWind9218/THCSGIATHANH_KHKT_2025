"""System and user prompts for simple response node."""


def get_simple_response_system_prompt() -> str:
    """System prompt for handling simple intent issues."""
    return (
        "Ban la ban dong hanh tam ly hoc duong cho teen. "
        "Tra loi ngan gon, am ap, hoi mo 1 cau de tiep tuc doi thoai."
    )


def get_simple_response_user_prompt(user_text: str, user_memory: str) -> str:
    """User prompt for simple response (includes personalization from memory)."""
    return f"User message: {user_text}\nUser memory: {user_memory}"


def get_out_of_scope_response() -> str:
    """Hardcoded response for out-of-scope messages."""
    return (
        "Chuyen do minh khong gioi lam, nhung neu ban dang co dieu gi can noi ra "
        "thi minh luon san sang lang nghe nhe."
    )
