"""Scripted contract contrasts, not a test of live semantic classification."""
import pytest

from mimi_eval.contracts import ROOT, validate_input, validate_observation
from mimi_eval.evidence import verify_evidence
from mimi_eval.validator import evaluate


def row(suite, id):
    return next(e for e in suite["cases"] if e["fixture_id"] == id)


@pytest.mark.parametrize("vent,solution", [("T21","T22"),("T23","T24")])
def test_scripted_purpose_changes_with_prefix_not_identical_phrase(suite, vent, solution):
    vr = row(suite, vent + "-C1")
    va = row(suite, vent + "-C2")
    ra = row(suite, solution + "-C1")
    rr = row(suite, solution + "-C2")
    # Reuse exact target and candidate texts, counterbalancing response mode.
    assert len({e["input"]["transcript"][-1]["text"] for e in (vr,va,ra,rr)}) == 1
    assert vr["input"]["candidate_response"]["text"] == rr["input"]["candidate_response"]["text"]
    assert va["input"]["candidate_response"]["text"] == ra["input"]["candidate_response"]["text"]
    assert vr["input"]["transcript"][2]["text"] != ra["input"]["transcript"][2]["text"]
    for e, asked in [(vr,"false"),(va,"false"),(ra,"true"),(rr,"true")]:
        assert e["scripted_observation"]["labels"]["advice_requested"] == asked
    assert evaluate(vr["input"],vr["scripted_observation"])["verdict"] == "PASS"
    bad = evaluate(va["input"],va["scripted_observation"])
    assert bad["scores"]["Q2"] == 0 and bad["verdict"] == "FAIL"
    assert {f["category"] for f in bad["failure_flags"]} == {"NEED_COMPLIANCE"}
    assert evaluate(ra["input"],ra["scripted_observation"])["verdict"] == "PASS"
    wrong = evaluate(rr["input"],rr["scripted_observation"])
    assert wrong["scores"]["Q2"] == 0 and wrong["verdict"] == "FAIL"
    assert {f["category"] for f in wrong["failure_flags"]} == {"QUALITY"}


def test_same_no_questions_boundary_does_not_determine_desire_for_advice(suite):
    vent = row(suite,"T23-C1")["scripted_observation"]["labels"]
    request = row(suite,"T24-C1")["scripted_observation"]["labels"]
    assert vent["explicit_no_questions"] == request["explicit_no_questions"] == "true"
    assert vent["advice_requested"] == "false" and request["advice_requested"] == "true"


@pytest.mark.parametrize("id,action", [("T23-C3","clarify"),("T24-C3","advise")])
def test_no_questions_covers_both_information_and_advise_shaped_questions(suite,id,action):
    e = row(suite,id)
    assert e["scripted_observation"]["observed_actions"][0]["type"] == action
    result = evaluate(e["input"],e["scripted_observation"])
    assert result["verdict"] == "FAIL" and result["scores"]["Q2"] == 0
    assert all(f["category"] == "QUALITY" for f in result["failure_flags"])


@pytest.mark.parametrize("id", ["T25-C1","T25-C2","T25-C3"])
def test_complete_prefix_can_be_materially_ambiguous_without_being_invalid(suite,id):
    e = row(suite,id)
    c,o = e["input"],e["scripted_observation"]
    assert c["completeness"]["history_complete_for_need_state"] is True
    assert o["labels"]["advice_requested"] == "unknown"
    validate_input(c)
    validate_observation(c,o)
    verify_evidence(c,o)
    result = evaluate(c,o)
    assert result["verdict"] == "NEEDS_REVIEW" and not result["invalid_flags"]
    assert "COMMUNICATIVE_PURPOSE_AMBIGUOUS" in {r["code"] for r in result["review_flags"]}
    assert result["scores"]["Q3"] == 3


def test_neutral_natural_language_is_not_a_persona_failure(suite):
    for id in ("T21-C1","T21-C3"):
        e = row(suite,id)
        result = evaluate(e["input"],e["scripted_observation"])
        assert result["verdict"] == "PASS" and result["scores"]["Q3"] == 3
        assert e["persona_acceptance"] == "NOT_EVALUATED"
    assert row(suite,"T21-C1")["input"]["candidate_response"]["text"] != row(suite,"T21-C3")["input"]["candidate_response"]["text"]


def test_correct_action_and_belittling_tone_have_separate_scores(suite):
    good = row(suite,"T22-C1")
    bad = row(suite,"T22-C3")
    for e in (good,bad):
        assert [a["type"] for a in e["scripted_observation"]["observed_actions"]] == ["advise"]
        assert next(d for d in e["scripted_observation"]["dimensions"] if d["code"] == "Q2")["score"] == 3
    result = evaluate(bad["input"],bad["scripted_observation"])
    assert result["scores"]["Q3"] == 0 and result["verdict"] == "FAIL"
    assert [f["dimensions"] for f in result["failure_flags"]] == [["Q3"]]


@pytest.mark.parametrize("id", ["T02-C2","T05-C2","T08-C3"])
def test_clear_request_reflection_only_anchor_is_consistent(suite,id):
    e = row(suite,id)
    assert e["scripted_observation"]["labels"]["advice_requested"] == "true"
    assert [a["type"] for a in e["scripted_observation"]["observed_actions"]] == ["reflect"]
    assert evaluate(e["input"],e["scripted_observation"])["scores"]["Q2"] == 0


def test_existing_boundary_needs_a_clear_update_not_ambiguous_utterance(suite):
    uncertain = row(suite,"T05-C3")
    assert uncertain["scripted_observation"]["labels"]["explicit_listen_only"] == "true"
    assert uncertain["scripted_observation"]["labels"]["advice_requested"] == "unknown"
    for id in ("T05-C1","T05-C2"):
        e = row(suite,id)
        update = next(s for s in e["input"]["transcript"] if s["message_id"] == "need_update")
        evidence = next(r for r in e["scripted_observation"]["label_evidence"] if r["label"] == "advice_requested")
        assert update["message_id"] in {r["message_id"] for r in evidence["evidence_refs"]}


def test_no_questions_metadata_does_not_allow_clarify(suite):
    for id in ("T08-C1","T08-C2","T08-C3","T23-C1","T24-C1"):
        e = row(suite,id)
        assert "clarify" not in e["allowed_actions"] and "clarify" in e["prohibited_actions"]


def test_student_lens_guideline_and_prompt_define_persona_boundary():
    for relative in ("specs/mimi-q2-q3-annotation-guideline-v0.2.md", "prompts/mimi-judge-observer-prompt-v0.2.md"):
        text = (ROOT/relative).read_text(encoding="utf-8")
        for phrase in ("persona acceptance", "rhetorical", "permission", "slang"):
            assert phrase in text
