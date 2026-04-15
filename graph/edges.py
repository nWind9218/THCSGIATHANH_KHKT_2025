from __future__ import annotations

from typing import Literal

from graph.state import CounselingState


def route_after_load_memory(state: CounselingState) -> Literal["safety_gateway", "end"]:
    if state.get("human_takeover"):
        return "end"
    return "safety_gateway"


def route_after_safety(state: CounselingState) -> Literal["handoff_human", "intent_routing"]:
    if state.get("is_emergency"):
        return "handoff_human"
    return "intent_routing"


def route_after_intent(state: CounselingState) -> Literal["out_of_scope", "simple_response", "info_gap_assessor"]:
    intent = state.get("intent_category", "simple")
    if intent == "out_of_scope":
        return "out_of_scope"
    if intent == "complex":
        return "info_gap_assessor"
    return "simple_response"


def route_after_info_gap(state: CounselingState) -> Literal["clarification", "deep_reasoning"]:
    if state.get("info_gap_status") == "missing":
        return "clarification"
    return "deep_reasoning"
