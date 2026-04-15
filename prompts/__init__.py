"""Prompt templates for counseling chatbot - centralized management."""

from .safety import get_safety_check_prompt
from .intent import get_intent_routing_prompt
from .simple_response import (
    get_simple_response_system_prompt,
    get_simple_response_user_prompt,
    get_out_of_scope_response,
)
from .clarification import get_clarification_prompt
from .deep_reasoning import (
    get_deep_reasoning_system_prompt,
    get_deep_reasoning_user_prompt,
)
from .info_gap import get_info_gap_assessment_prompt
from .save_memory import get_memory_gate_prompt

__all__ = [
    "get_safety_check_prompt",
    "get_intent_routing_prompt",
    "get_simple_response_system_prompt",
    "get_simple_response_user_prompt",
    "get_out_of_scope_response",
    "get_clarification_prompt",
    "get_deep_reasoning_system_prompt",
    "get_deep_reasoning_user_prompt",
    "get_info_gap_assessment_prompt",
    "get_memory_gate_prompt",
]
