"""Literal source/role/hash/span integrity. This is not a semantic verifier."""
from __future__ import annotations

from .contracts import require, sources


def check_ref(case: dict, ref: dict, allowed_ids: set[str] | None = None) -> None:
    index = {s["message_id"]: s for s in sources(case)}
    require(ref["message_id"] in index, "evidence_source_missing")
    source = index[ref["message_id"]]
    require(allowed_ids is None or ref["message_id"] in allowed_ids, "evidence_wrong_source")
    require(ref["source_sha256"] == source["sha256"], "evidence_hash_mismatch")
    require(ref["role"] == source["role"], "evidence_role_mismatch")
    start, end = ref["start"], ref["end"]
    require(0 <= start < end <= len(source["text"]), "evidence_range")
    # Python indexes Unicode code points. Do not normalize text after freezing it.
    require(source["text"][start:end] == ref["quote"], "evidence_quote_mismatch")


def verify_evidence(case: dict, observation: dict) -> None:
    candidate_ids = {case["candidate_response"]["message_id"]}
    transcript_ids = {s["message_id"] for s in case["transcript"]}
    student_ids = {s["message_id"] for s in case["transcript"] if s["role"] == "student"}
    for action in observation["observed_actions"]:
        for ref in action["evidence_refs"]:
            check_ref(case, ref, candidate_ids)
    for row in observation["label_evidence"]:
        allowed = transcript_ids | candidate_ids
        if row["label"] in {"explicit_listen_only", "explicit_no_questions", "advice_requested"}:
            allowed = student_ids
        for ref in row["evidence_refs"]:
            check_ref(case, ref, allowed)
    for row in observation["dimensions"] + observation["violation_candidates"]:
        for ref in row["evidence_refs"]:
            check_ref(case, ref)
    for ref in observation["claims_human_ack"]["evidence_refs"]:
        check_ref(case, ref, candidate_ids)
    for ref in observation["safety_assessment"]["evidence_refs"]:
        check_ref(case, ref, transcript_ids | candidate_ids)
    for ref in case["safety_exception"]["risk_evidence"]:
        check_ref(case, ref, student_ids)
    for ref in case["safety_exception"]["intervention_evidence"]:
        check_ref(case, ref, candidate_ids)
