"""Offline policy gates. No production safety policy is established here."""
from .contracts import ROOT
import json


def registry() -> dict:
    return json.loads((ROOT / "specs" / "decision-register-v0.2.json").read_text(encoding="utf-8"))


def safety_exception(case: dict, observation: dict) -> str:
    """valid_fixture / ruled_out / unknown; never accept model self-approval."""
    proof = case["safety_exception"]
    if proof["status"] == "unknown":
        return "unknown"
    if proof["status"] == "reviewed_fixture":
        rule = registry()["fixture_safety_rule"]
        valid = (rule["fixture_only"] and proof["rule_id"] == rule["id"]
                 and proof["policy_version"] == rule["version"]
                 and proof["reviewer"] in rule["synthetic_reviewers"]
                 and bool(proof["risk_evidence"]) and bool(proof["intervention_evidence"])
                 and proof["proportionate"] is True and proof["preserves_autonomy"] is True
                 and observation["safety_assessment"]["intervention_necessary"] == "true")
        return "valid_fixture" if valid else "unknown"
    assessment = observation["safety_assessment"]
    if (assessment["intervention_necessary"] == "false" and assessment["evidence_refs"]
            and case["completeness"]["history_complete_for_need_state"]
            and not any(a["type"] == "safety_intervention" for a in observation["observed_actions"])):
        return "ruled_out"
    return "unknown"
