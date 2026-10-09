import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimi_v2.datasets import load_dataset
from mimi_v2.models import ModelConfig, ModelError, OpenAIModel
from mimi_v2.pilot import MeteredModel, UsageLedger, choose_pilot, run_pilot


def test_deterministic_balanced_selection_is_development_only():
    development = load_dataset("eval_datasets/v2_acceptance_development.json")
    pilot = choose_pilot(development)
    assert pilot == choose_pilot(development)
    assert len(pilot["cases"]) == 20
    assert len({c["category"] for c in pilot["cases"]}) == 20
    assert {v: sum(c["variant"] == v for c in pilot["cases"]) for v in {c["variant"] for c in pilot["cases"]}} == {
        "continuation": 4, "correction": 4, "listen_only": 4, "multi_intent": 4, "deep_reflection": 4}
    assert all(c["split"] == "development" and not c.get("setup") for c in pilot["cases"])
    assert any(max(len(t["user"].split()) for t in c["turns"]) >= 150 for c in pilot["cases"])
    acute = next(c for c in pilot["cases"] if c["category"] == "acute_safety_and_trusted_support")
    assert acute["variant"] in {"listen_only", "multi_intent", "deep_reflection"}
    assert not any(t.get("action") or t.get("fault") for t in acute["turns"])


def test_cost_cached_input_and_reasoning_not_double_counted(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    ledger.append({"engine": "v2", "provider_response": True, "usage": {
        "input_tokens": 1000000, "input_tokens_details": {"cached_tokens": 200000},
        "output_tokens": 100000, "output_tokens_details": {"reasoning_tokens": 30000}}})
    ledger.append({"engine": "v2", "provider_response": False, "usage": {}, "error": "timeout"})
    cost = ledger.summary()
    assert cost["engines"]["v2"]["observed_usage_estimate_usd"] == "0.40500000"
    assert cost["unknown_usage_calls"] == 1
    assert cost["engines"]["v2"]["reasoning_tokens_included_in_output"] == 30000


def test_observer_records_billable_incomplete_response(tmp_path):
    async def scenario():
        async def create(**kwargs):
            return {"status": "incomplete", "model": "gpt-5-mini-2025-08-07", "usage": {"input_tokens": 20, "output_tokens": 50}, "output": []}
        ledger = UsageLedger(tmp_path / "usage.jsonl")
        provider = OpenAIModel(ModelConfig(reasoning_effort="minimal"), client=SimpleNamespace(responses=SimpleNamespace(create=create)))
        model = MeteredModel(provider, ledger, "v2", "safe-id")
        with pytest.raises(ModelError, match="incomplete_output"):
            await model.complete([{"role": "user", "content": "do-not-log-content"}], "reply")
        assert len(ledger.events) == 1
        assert ledger.events[0]["provider_status"] == "incomplete"
        assert ledger.summary()["unknown_usage_calls"] == 0
        assert "do-not-log-content" not in ledger.path.read_text()
    asyncio.run(scenario())


def test_fixed_scope_checkpoint_and_no_automatic_rerun(tmp_path):
    pilot = choose_pilot(load_dataset("eval_datasets/v2_acceptance_development.json"))
    seen = []
    class Engine:
        async def respond(self, user_id, text):
            return {"response_text": "synthetic test output", "assessment": {}, "diagnostics": []}
    def factory(config, case, ledger):
        seen.append((config["engine"], case["id"]))
        return Engine()
    result = asyncio.run(run_pilot(pilot, tmp_path, engine_factory=factory))
    assert len(seen) == 40 and len(set(seen)) == 40
    assert len(result["results"]) == 40
    assert len(list(tmp_path.glob("*.checkpoint.json"))) == 40
    assert json.loads((tmp_path / "pilot_dataset.json").read_text(encoding="utf-8"))["provenance"]["holdout_used"] is False
    with pytest.raises(ValueError, match="already started"):
        asyncio.run(run_pilot(pilot, tmp_path, engine_factory=factory))
    invalid = dict(pilot, cases=pilot["cases"][:-1])
    with pytest.raises(ValueError, match="20 cases"):
        asyncio.run(run_pilot(invalid, tmp_path / "bad", engine_factory=factory))
