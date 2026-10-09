import asyncio
import pytest
from mimi_v2.datasets import validate_dataset
from mimi_v2.evaluation import run_job
from mimi_v2.simulator import make_engine


def test_setup_acknowledge_resume_reset_are_observed_not_prompt_claims():
    dataset = {"version": 1, "cases": [{"id": "controls", "setup": {"handoff_status": "pending"}, "turns": [
        {"user": "Mình muốn được lắng nghe", "action": "acknowledge", "expect": {"handoff": "accepted", "max_chars": 0}},
        {"user": "Mình muốn được lắng nghe", "action": "resume", "expect": {"handoff": "none"}},
        {"user": "Xin chào", "action": "reset"},
    ]}]}
    job = asyncio.run(run_job(dataset, [{"engine": "v2"}], lambda c, case: make_engine(c)))
    turns = job["results"][0]["turns"]
    assert turns[0]["status"] == "pass"
    assert turns[0]["result"]["observed_session"]["handoff_status"] == "accepted"
    assert turns[1]["result"]["environment_action"]["action"] == "resume"
    assert len(turns[2]["result"]["messages"]) == 2


def test_faults_reset_between_turns_and_are_not_counted_as_pass():
    dataset = {"version": 1, "cases": [{"id": "fault", "turns": [
        {"user": "Xin chào", "fault": "model"}, {"user": "Xin chào", "action": "reset"},
    ]}]}
    job = asyncio.run(run_job(dataset, [{"engine": "v2"}], lambda c, case: make_engine(c)))
    turns = job["results"][0]["turns"]
    assert turns[0]["status"] == "error"
    assert turns[1]["status"] == "pending_review"


def test_controls_cannot_be_used_on_production_like_store():
    class Engine:
        store = object()
        async def respond(self, *args):
            pytest.fail("must reject before any response or write")
    dataset = {"version": 1, "cases": [{"id": "unsafe", "setup": {"handoff_status": "accepted"}, "turns": [{"user": "hello"}]}]}
    job = asyncio.run(run_job(dataset, [{"engine": "v2"}], lambda c, case: Engine()))
    assert job["summary"]["error"] == 1


@pytest.mark.parametrize("field,value", [("action", "delete"), ("fault", "production")])
def test_unknown_actions_and_faults_rejected(field, value):
    with pytest.raises(ValueError):
        validate_dataset({"version": 1, "cases": [{"id": "x", "turns": [{"user": "hi", field: value}]}]})


def test_simple_engine_errors_are_observed_after_respond():
    called = []
    class Engine:
        async def respond(self, *args):
            called.append(True)
            return {"response_text": "fallback", "diagnostics": ["reply_unavailable_or_invalid"]}
    job = asyncio.run(run_job({"version": 1, "cases": [{"id": "x", "turns": [{"user": "hi"}]}]},
                              [{"engine": "v2"}], lambda c, case: Engine()))
    assert called == [True]
    assert job["summary"]["error"] == 1
