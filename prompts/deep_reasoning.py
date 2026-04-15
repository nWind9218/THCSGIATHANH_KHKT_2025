"""Deep reasoning prompts for synthesizing KB + user memory into counseling response."""


def get_deep_reasoning_system_prompt() -> str:
    """System prompt for deep reasoning with KB integration."""
    return (
        "Ban la tro ly tam ly hoc duong cho teen Viet Nam. "
        "Tra loi dang hoi thoai tu nhien, khong dung bullet list, "
        "khong loanh quanh, khong bịa thong tin ngoai context."
    )


def get_deep_reasoning_user_prompt(user_text: str, kb_guidelines: str, user_memory: str) -> str:
    """User prompt for deep reasoning (includes KB + memory context)."""
    return (
        f"User message: {user_text}\n"
        f"KB guidelines: {kb_guidelines}\n"
        f"User memory: {user_memory}\n"
        "Viet 1 doan phan hoi vua dong cam vua co huong dan thuc te."
    )
