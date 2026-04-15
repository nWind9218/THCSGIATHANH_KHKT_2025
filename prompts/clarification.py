"""Clarification question prompt for gathering missing information."""

import json


def get_clarification_prompt(scratchpad: list[str], user_text: str) -> str:
    """
    Generate a single, open-ended clarification question.
    Rules:
    - Exactly 1 question per response
    - Open-ended, not leading
    - Friendly tone, not clinical
    """
    return (
        "Dat 1 cau hoi mo duy nhat de bo sung thong tin con thieu. "
        "Giong dieu than thien, khong dan dat, khong giang giai dai dong.\n"
        f"Scratchpad: {json.dumps(scratchpad, ensure_ascii=False)}\n"
        f"Latest: {user_text}"
    )
