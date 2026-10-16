"""Strict JSON shape and cross-field validation; never resolve semantics here."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent
DIMENSIONS = tuple(f"Q{i}" for i in range(1, 7))
LABELS = ("explicit_listen_only", "explicit_no_questions", "advice_requested",
          "unsolicited_advice", "response_mode_match")


class ContractError(ValueError):
    """An invalid structure or source identity, distinct from missing evidence."""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: object) -> str:
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False))


def schema(name: str) -> dict:
    path = ROOT / "contracts" / f"mimi-judge-{name}-v0.2.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def validate_shape(value: object, name: str) -> None:
    errors = list(Draft202012Validator(schema(name)).iter_errors(value))
    # Do not echo student text or arbitrary model output in exception messages.
    require(not errors, f"{name}_schema_invalid")


def sources(case: dict) -> list[dict]:
    return [*case["transcript"], case["candidate_response"],
            *case["retrieval_evidence"], *case["memory_evidence"]]


def validate_input(case: dict) -> None:
    validate_shape(case, "input")
    all_sources = sources(case)
    ids = [s["message_id"] for s in all_sources]
    require(len(ids) == len(set(ids)), "duplicate_source_id")
    for source in all_sources:
        require(source["sha256"] == text_hash(source["text"]), "source_hash_mismatch")
    require(all(s["role"] in {"student", "assistant"} for s in case["transcript"]),
            "transcript_role")
    require(case["transcript"][-1]["role"] == "student", "target_user_missing")
    require(case["candidate_response"]["role"] == "assistant", "candidate_role")
    require(all(s["role"] == "tool" for s in
                case["retrieval_evidence"] + case["memory_evidence"]), "reference_role")
    scope = case["evaluation_scope"]
    required = set(case["required_dimensions"])
    if scope == "scoped_q2_q3":
        require(required == {"Q2", "Q3"}, "scoped_dimensions")
    elif scope == "full_rubric":
        require(required == set(DIMENSIONS), "full_dimensions")
    else:
        require(not required, "silence_dimensions")
    event_ids = [e["event_id"] for e in case["tool_events"]]
    require(len(event_ids) == len(set(event_ids)), "duplicate_event_id")
    for event in case["tool_events"]:
        try:
            parsed = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            require(parsed.tzinfo is not None, "event_timestamp_timezone")
        except ValueError as exc:
            raise ContractError("event_timestamp") from exc
        require(event["type"] != "human_ack" or event["actor_kind"] == "human",
                "ack_actor_not_human")
    router = case["router_check"]
    require(router["required"] or (router["expected"] is None and router["actual"] is None),
            "unused_router_values")
    require((case["runtime"]["status"] == "technical_failure") ==
            (case["runtime"]["error_code"] is not None), "runtime_error_contract")
    proof = case["safety_exception"]
    if proof["status"] == "not_claimed":
        require(all(proof[k] is None for k in ("rule_id", "policy_version", "reviewer",
                                              "proportionate", "preserves_autonomy"))
                and not proof["risk_evidence"] and not proof["intervention_evidence"],
                "unclaimed_exception_has_proof")


def validate_observation(case: dict, observation: dict) -> None:
    validate_shape(observation, "contract")
    require(observation["case_id"] == case["case_id"], "case_identity")
    require(observation["input_sha256"] == fingerprint(case), "input_identity")
    require(observation["rubric_version"] == case["rubric_version"], "rubric_identity")
    label_rows = observation["label_evidence"]
    require(len(label_rows) == len(LABELS) and
            {r["label"] for r in label_rows} == set(LABELS), "label_coverage")
    codes = [d["code"] for d in observation["dimensions"]]
    require(len(codes) == len(set(codes)), "duplicate_dimension")
    require(set(codes) == set(case["required_dimensions"]), "dimension_coverage")
    for dimension in observation["dimensions"]:
        require(dimension["applicable"] == (dimension["score"] is not None),
                "applicability_score")
        require(dimension["applicable"] or bool(dimension["reason"].strip()), "na_reason")
        if dimension["code"] in {"Q2", "Q3"}:
            require(dimension["applicable"], "required_interaction_dimension_na")
    # All references, including harness safety proof, are checked by evidence.py.
