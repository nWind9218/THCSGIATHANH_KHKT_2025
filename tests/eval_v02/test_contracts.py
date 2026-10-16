from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mimi_eval.contracts import ContractError, schema, validate_input, validate_observation
from mimi_eval.validator import evaluate
from conftest import rebind


@pytest.mark.parametrize("name", ["input", "contract"])
def test_schemas_are_closed_and_all_fields_required(name):
    s = schema(name)
    Draft202012Validator.check_schema(s)
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values(): walk(value)
        elif isinstance(node, list):
            for value in node: walk(value)
    walk(s)


def test_positive_envelopes(sample):
    c, o = sample
    validate_input(c)
    validate_observation(c, o)


@pytest.mark.parametrize("mutation", [
    lambda c,o: o.update(final_verdict="PASS"),
    lambda c,o: o.update(safety_exception_approved=True),
    lambda c,o: o.update(handoff_acknowledged=True),
    lambda c,o: o.update(effective_model="gpt-5.6-terra"),
    lambda c,o: o.update(effective_effort="high"),
    lambda c,o: o["labels"].update(explicit_listen_only=True),
    lambda c,o: o["labels"].update(explicit_listen_only="not_applicable"),
    lambda c,o: o.pop("missing_evidence"),
    lambda c,o: o["dimensions"][0].update(score=4),
    lambda c,o: o["dimensions"][0].update(score=True),
    lambda c,o: o["dimensions"][0].update(applicable=False,score=3),
    lambda c,o: o["dimensions"][0].update(applicable=True,score=None),
    lambda c,o: o["dimensions"][0].update(applicable=False,score=None),
    lambda c,o: o["dimensions"].append(deepcopy(o["dimensions"][0])),
    lambda c,o: o["label_evidence"].append(deepcopy(o["label_evidence"][0])),
    lambda c,o: o.update(case_id="different"),
    lambda c,o: o.update(input_sha256="0"*64),
    lambda c,o: o["observed_actions"][0].update(type="primary_action"),
])
def test_negative_judge_contract(sample, mutation):
    c,o=sample
    mutation(c,o)
    assert evaluate(c,o)["verdict"] == "INVALID"


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("transcript"),
    lambda c: c.update(transcript=[]),
    lambda c: c["transcript"][0].update(sha256="0"*64),
    lambda c: c["candidate_response"].update(message_id=c["transcript"][0]["message_id"]),
    lambda c: c.update(required_dimensions=["Q2"]),
    lambda c: c.update(required_dimensions=["Q2","Q3","Q3"]),
    lambda c: c["transcript"][-1].update(role="assistant"),
    lambda c: c["runtime"].update(error_code="oops"),
    lambda c: c.update(router_check="malformed"),
])
def test_negative_input_contract(sample, mutation):
    c,o=sample
    mutation(c)
    rebind(c,o)
    assert evaluate(c,o)["verdict"] == "INVALID"


@pytest.mark.parametrize("field", ["timestamp", "actor_kind", "event_id"])
def test_event_integrity(suite, field):
    e=next(e for e in suite["cases"] if e["fixture_id"]=="CONTROL-1")
    c,o=e["input"],e["scripted_observation"]
    if field=="timestamp": c["tool_events"][0][field]="not-a-time"
    if field=="actor_kind": c["tool_events"][0][field]="system"
    if field=="event_id": c["tool_events"].append(deepcopy(c["tool_events"][0]))
    rebind(c,o)
    assert evaluate(c,o)["verdict"]=="INVALID"


@pytest.mark.parametrize("case,observation", [(None,None),([],[]),({},""),({"router_check":"bad"},{})])
def test_invalid_container_never_crashes(case,observation):
    assert evaluate(case,observation)["verdict"] == "INVALID"
