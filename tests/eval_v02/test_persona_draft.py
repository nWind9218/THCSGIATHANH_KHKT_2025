"""Data-quality checks for persona drafts; no model or persona scoring here."""
from collections import defaultdict
import json

import pytest
from jsonschema import Draft202012Validator

from mimi_eval.contracts import ROOT, ContractError, fingerprint, schema, text_hash
from mimi_eval.evidence import check_ref
from mimi_eval.replay import run_suite


@pytest.fixture
def persona_suite():
    return json.loads((ROOT/"fixtures/mimi-persona-suite-draft-v0.2.json").read_text(encoding="utf-8"))


@pytest.fixture
def persona_schema():
    return json.loads((ROOT/"contracts/mimi-persona-suite-draft-v0.2.schema.json").read_text(encoding="utf-8"))


@pytest.fixture
def persona_rubric():
    return json.loads((ROOT/"specs/mimi-persona-rubric-v0.2.json").read_text(encoding="utf-8"))


def test_persona_schema_and_fixture_shape(persona_suite,persona_schema):
    Draft202012Validator.check_schema(persona_schema)
    Draft202012Validator(persona_schema).validate(persona_suite)
    assert len(persona_suite["scenarios"]) == 17
    assert sum(len(s["candidates"]) for s in persona_suite["scenarios"]) == 51
    assert persona_suite["persona_acceptance"] == "NOT_EVALUATED"


def test_six_facets_are_approved_design_not_operational_acceptance(persona_rubric):
    assert persona_rubric["status"] == "APPROVED_BY_PRODUCT_OWNER"
    assert {d["code"] for d in persona_rubric["dimensions"]} == {f"P{i}" for i in range(1,7)}
    for d in persona_rubric["dimensions"]:
        assert set(d["anchors"]) == {"0","1","2","3"}
        assert all(d["anchors"].values())
    assert persona_rubric["approval"]["detailed_anchors"] == "APPROVED_BY_PRODUCT_OWNER"
    assert persona_rubric["proposed_scoring"]["status"] == "APPROVED_DESIGN_NOT_EXECUTABLE_GATE"
    assert persona_rubric["proposed_scoring"]["no_full_persona_pass_from_scoped_cases"] is True
    assert persona_rubric["approval"]["human_gold"] == "UNREVIEWED"
    assert persona_rubric["approval"]["qualified_policy_review"] == "NOT_COMPLETED"
    assert persona_rubric["approval"]["persona_acceptance"] == "NOT_EVALUATED"
    register=json.loads((ROOT/"specs/decision-register-v0.2.json").read_text(encoding="utf-8"))
    assert register["persona_design"]["detailed_rubric_status"] == persona_rubric["status"]
    assert register["persona_design"]["fixture_status"] == "UNREVIEWED"


def test_four_flexible_modes_with_positive_examples(persona_suite,persona_rubric):
    modes={m["id"] for m in persona_rubric["temperament"]["modes"]}
    assert modes == {"subtle_playful","mischievous_playful","gentle_warm","energetic_encouraging"}
    assert persona_rubric["temperament"]["direction_status"] == "USER_CONFIRMED_FLEXIBLE_FOUR_MODES"
    examples={"MP01":"subtle_playful","MP15":"mischievous_playful","MP06":"gentle_warm","MP16":"energetic_encouraging"}
    for id,mode in examples.items():
        s=next(s for s in persona_suite["scenarios"] if s["scenario_id"]==id)
        assert mode in s["tone_context"]["appropriate_modes"]
        assert all(r["range"][0]>=2 for r in s["candidates"][0]["draft_scores"])
    assert any(len(s["tone_context"]["appropriate_modes"])>1 for s in persona_suite["scenarios"])


def test_focus_ranges_and_sources_are_bound_to_current_candidate(persona_suite):
    ids=[]
    for s in persona_suite["scenarios"]:
        assert s["target_user_message"]["role"]=="student"
        all_sources=[*s["context"],s["target_user_message"],*[c["response"] for c in s["candidates"]]]
        source_ids=[x["message_id"] for x in all_sources]
        assert len(source_ids)==len(set(source_ids))
        for source in all_sources:
            assert source["sha256"]==text_hash(source["text"])
        for c in s["candidates"]:
            ids.append(c["case_id"])
            assert c["response"]["role"]=="assistant"
            codes=[r["code"] for r in c["draft_scores"]]
            assert len(codes)==len(set(codes)) and set(codes)==set(s["persona_focus"])
            case={"transcript":[*s["context"],s["target_user_message"]],"candidate_response":c["response"],"retrieval_evidence":[],"memory_evidence":[]}
            allowed={x["message_id"] for x in case["transcript"]}|{c["response"]["message_id"]}
            for score in c["draft_scores"]:
                lo,hi=score["range"]
                assert 0<=lo<=hi<=3
                assert any(r["message_id"]==c["response"]["message_id"] for r in score["evidence_refs"])
                for ref in score["evidence_refs"]:
                    check_ref(case,ref,allowed)
    assert len(ids)==len(set(ids))==51


def test_no_peer_candidate_source_can_stand_in_for_response(persona_suite):
    s=persona_suite["scenarios"][0];c=s["candidates"][0]
    case={"transcript":[*s["context"],s["target_user_message"]],"candidate_response":c["response"],"retrieval_evidence":[],"memory_evidence":[]}
    foreign=next(r for r in s["candidates"][1]["draft_scores"][0]["evidence_refs"] if r["message_id"]==s["candidates"][1]["response"]["message_id"])
    with pytest.raises(ContractError):check_ref(case,foreign)


def test_turn_pacing_and_voice_claims_use_available_history(persona_suite):
    repeated=next(s for s in persona_suite["scenarios"] if s["scenario_id"]=="MP08")
    switch=next(s for s in persona_suite["scenarios"] if s["scenario_id"]=="MP17")
    assert len([m for m in repeated["context"] if m["role"]=="assistant"])==2
    assert repeated["context"][1]["text"]==repeated["context"][3]["text"]
    assert switch["tone_context"]["appropriate_modes"]==["gentle_warm"]
    assert switch["behavior_controls"]["no_questions"] is True
    for s in (repeated,switch):
        history_ids={x["message_id"] for x in s["context"]}
        assert history_ids
        for c in s["candidates"]:
            assert any(r["message_id"] in history_ids for d in c["draft_scores"] for r in d["evidence_refs"])


def test_multiple_natural_responses_not_one_canonical_string(persona_suite):
    for id in ("MP02","MP08"):
        s=next(s for s in persona_suite["scenarios"] if s["scenario_id"]==id)
        assert s["candidates"][0]["response"]["text"]!=s["candidates"][2]["response"]["text"]
        for c in (s["candidates"][0],s["candidates"][2]):
            assert all(d["range"][0]>=2 for d in c["draft_scores"])


def test_candidate_honesty_and_relation_risks_are_not_excused_by_style(persona_suite):
    for id,tag in (("MP10","HONESTY_REVIEW"),("MP11","RELATIONSHIP_POLICY_REVIEW")):
        s=next(s for s in persona_suite["scenarios"] if s["scenario_id"]==id)
        c=s["candidates"][2]
        assert tag in c["policy_review_tags"]
        assert next(d for d in c["draft_scores"] if d["code"]=="P6")["range"]==[0,0]
        assert c["human_adjudication"]["status"]=="UNREVIEWED"


def test_gold_template_is_empty_and_bound_to_unscored_input(persona_suite,persona_rubric):
    gold=json.loads((ROOT/"fixtures/human-gold-persona-template.v0.2.json").read_text(encoding="utf-8"))
    assert gold["status"]=="UNREVIEWED" and gold["rubric_sha256"]==fingerprint(persona_rubric)
    byid={r["case_id"]:r for r in gold["annotations"]}
    assert len(byid)==51
    for s in persona_suite["scenarios"]:
        for c in s["candidates"]:
            row=byid[c["case_id"]]
            basis={"scenario_id":s["scenario_id"],"persona_focus":s["persona_focus"],"tone_context":s["tone_context"],"context":s["context"],"target_user_message":s["target_user_message"],"completeness":s["completeness"],"behavior_controls":s["behavior_controls"],"response":c["response"]}
            assert row["basis_sha256"]==fingerprint(basis)
            assert row["reviewers"]==[] and row["scores_or_ranges"] is None and row["observed_traits"] is None
            assert row["status"]=="UNREVIEWED" and row["adjudicator"] is None


def test_family_split_and_case_ids_are_distinct_from_behavior_suite(persona_suite,suite):
    split=defaultdict(set)
    for s in persona_suite["scenarios"]:split[s["family_id"]].add(s["split"])
    assert all(len(v)==1 for v in split.values())
    persona_ids={c["case_id"] for s in persona_suite["scenarios"] for c in s["candidates"]}
    assert persona_ids.isdisjoint({r["fixture_id"] for r in suite["cases"]})
    assert len(suite["cases"])==79


def test_persona_draft_is_not_a_behavior_replay_or_observer_contract(persona_suite):
    assert not Draft202012Validator(schema("contract")).is_valid(persona_suite)
    with pytest.raises(ValueError,match="invalid_scripted_suite"):run_suite(persona_suite)


@pytest.mark.parametrize("mutate", [
    lambda s:s.update(final_verdict="PASS"),
    lambda s:s.update(persona_acceptance="PASS"),
    lambda s:s.update(document_status="APPROVED"),
    lambda s:s["scenarios"][0]["persona_focus"].append("P7"),
    lambda s:s["scenarios"][0]["tone_context"]["appropriate_modes"].append("always_slang"),
    lambda s:s["scenarios"][0]["candidates"][0]["draft_scores"][0].update(range=[0,4]),
    lambda s:s["scenarios"][0]["candidates"][0]["draft_scores"][0].update(range=[True,3]),
    lambda s:s["scenarios"][0]["candidates"][0].update(effective_model="gpt-5.6-terra"),
    lambda s:s["scenarios"][0]["candidates"][0]["human_adjudication"].update(status="APPROVED"),
    lambda s:s["scenarios"][0]["candidates"][0]["human_adjudication"].update(reviewers=["not-an-actual-review"]),
])
def test_persona_draft_negative_schema(persona_suite,persona_schema,mutate):
    mutate(persona_suite)
    assert not Draft202012Validator(persona_schema).is_valid(persona_suite)


def test_markdown_rubric_renders_current_anchors(persona_rubric):
    text=(ROOT/"specs/mimi-persona-rubric-v0.2.md").read_text(encoding="utf-8")
    for d in persona_rubric["dimensions"]:
        assert d["name"] in text
        assert all(a in text for a in d["anchors"].values())
    for mode in persona_rubric["temperament"]["modes"]:assert mode["name"] in text
