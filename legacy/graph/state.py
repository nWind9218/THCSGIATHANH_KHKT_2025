from __future__ import annotations

from typing import Annotated, Literal, TypedDict


def merge_last_twenty(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    combined = (old or []) + (new or [])
    return combined[-20:]


class CounselingState(TypedDict, total=False):
    messages: Annotated[list[dict], merge_last_twenty]
    user_id: str

    current_topic: str

    intent_category: Literal["out_of_scope", "simple", "complex"]
    info_gap_status: Literal["missing", "sufficient"]

    kb_guidelines: str
    student_knowledge: str
    user_memory: str

    reasoning_scratchpad: list[str]

    is_emergency: bool
    human_takeover: bool

    response_text: str
    error: str
