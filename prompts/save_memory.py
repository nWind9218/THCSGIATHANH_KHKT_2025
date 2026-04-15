"""Save memory and gating prompts for long-term memory persistence."""


def get_memory_gate_prompt(user_text: str) -> str:
    """
    Determine if this message contains info worth storing in long-term memory.
    Returns: JSON with {"should_store": true|false, "memory": "..."}
    """
    return (
        "Kiem tra user message co thong tin ben vung ve hoan canh/tinh cach/van de lap lai hay khong. "
        "Tra JSON {\"should_store\": true|false, \"memory\": \"...\"}.\n"
        f"Message: {user_text}"
    )
