"""Provider-compatible decision schemas; public/session facts remain dictionaries."""
from __future__ import annotations

import json
from copy import deepcopy


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


SAFETY_SCHEMA = _object({
    "risk": {"type": "string", "enum": ["safe", "urgent", "uncertain"]},
    "confidence": {"type": "string", "enum": ["high", "low"]},
    "reason": {"type": "string"},
})

DELIBERATE_SCHEMA = _object({
    "cues": {"type": "array", "items": {"type": "string"}},
    "about_self": {"type": "boolean"},
    "negated_or_denied": {"type": "boolean"},
    "joking_or_hyperbole": {"type": "boolean"},
    "quoted_or_hypothetical": {"type": "boolean"},
    "intent": {"type": "string", "enum": ["none", "passive", "active"]},
    "plan_or_means": {"type": "boolean"},
    "abuse_or_danger_from_others": {"type": "boolean"},
    "context_trend": {"type": "string"},
    "risk": {"type": "string", "enum": ["safe", "uncertain", "urgent"]},
    "reason": {"type": "string"},
})

ASSESSMENT_SCHEMA = _object({
    "known": {"type": "array", "items": _object({"key": {"type": "string"}, "value": {"type": "string"}})},
    "missing_fields": {"type": "array", "items": {"type": "string"}},
    "next_question_focus": {"type": "string"},
    "need": {"type": "string", "enum": ["listen", "comfort", "clarify", "advice", "explain"]},
    "intent": {"type": "string", "enum": ["simple", "support", "closing", "out_of_scope"]},
    "topic": {"type": "string"},
    "core_issue": {"type": "string"},
    "safety_hint": {"type": "string", "enum": ["safe", "urgent", "uncertain"]},
    "user_role": {"type": "string", "enum": ["unknown", "student", "parent"]},
})

SCHEMAS = {
    "safety": {"name": "mimi_safety", "schema": SAFETY_SCHEMA, "strict": True},
    "deliberate": {"name": "mimi_safety_deliberation", "schema": DELIBERATE_SCHEMA, "strict": True},
    "assessment": {"name": "mimi_assessment", "schema": ASSESSMENT_SCHEMA, "strict": True},
}


def schema_for_purpose(purpose: str) -> dict | None:
    return deepcopy(SCHEMAS.get(purpose))


class ContractError(ValueError):
    """Only predefined codes/fields, never raw model text or user fact keys."""
    def __init__(self, code: str, field: str = "known"):
        self.code, self.field = code, field
        super().__init__(code)


def parse_decision(text: str) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("duplicate_json_property", "payload")
            result[key] = value
        return result
    parsed = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(parsed, dict):
        raise ContractError("object_required", "payload")
    return parsed


def normalize_assessment(payload: dict) -> dict:
    """Normalize wire facts without coercion; allow trusted local legacy dict fixtures."""
    result = deepcopy(payload)
    facts = result.get("known")
    if isinstance(facts, list):
        normalized = {}
        for item in facts:
            if not isinstance(item, dict) or set(item) != {"key", "value"}:
                raise ContractError("invalid_fact_entry")
            key, value = item["key"], item["value"]
            if not isinstance(key, str) or not key.strip() or key != key.strip():
                raise ContractError("invalid_fact_key")
            if not isinstance(value, str):
                raise ContractError("invalid_fact_value")
            # A repeated key is the model correcting itself within one answer: the last
            # value is its newest statement. Never fatal (a bad fact must not block a turn).
            normalized[key] = value
        result["known"] = normalized
    elif isinstance(facts, dict):
        if any(not isinstance(key, str) or not key.strip() or key != key.strip() for key in facts):
            raise ContractError("invalid_fact_key")
    # Other values remain untouched for strict domain validation to reject.
    return result
