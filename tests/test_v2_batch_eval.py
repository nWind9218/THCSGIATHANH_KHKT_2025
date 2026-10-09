import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimi_v2.batch_eval import Circuit, DurableModel, Journal, JobPause, RunLock, atomic_json, process_token, run_batch
from mimi_v2.core import CounselingEngine
from mimi_v2.models import ModelConfig, ModelError, ModelReply
from mimi_v2.simulator import FaultStore, OfflineModel


def dataset(n=1):
    return {"version": 1, "name": "synthetic batch tests", "cases": [
        {"id": f"case-{i}", "turns": [{"user": "first"}, {"user": "second"}]} for i in range(n)]}


class FakeEngine:
    def __init__(self, calls, engine):
        self.calls, self.name = calls, engine
        self.store = FaultStore()
        self.model = SimpleNamespace()
        self._recovery = {}

    async def respond(self, user_id, text):
        self.calls.append((self.name, text))
        state = await self.store.load(user_id)
        messages = state.setdefault("messages", [])
        messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": "reply"}])
        await self.store.save(user_id, state)
        return {"response_text": "reply", "messages": messages, "diagnostics": []}


def test_atomic_completed_turns_resume_without_paid_replay(tmp_path):
    calls = []
    factory = lambda cfg, case, journal, circuit: FakeEngine(calls, cfg["engine"])
    original = asyncio.run(run_batch(dataset(), tmp_path, engine_factory=factory))
    assert original["processed_turns"] == 4 and len(calls) == 4
    resumed = asyncio.run(run_batch(dataset(), tmp_path, resume=True, engine_factory=factory))
    assert resumed["state"] == "complete" and len(calls) == 4
    # Simulate a process interruption between already-completed turns.
    path = tmp_path / "cases" / "case-0-v2.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint.update(phase="ready", next_turn=1, turns=checkpoint["turns"][:1])
    checkpoint["state"]["session"]["messages"] = checkpoint["state"]["session"]["messages"][:2]
    atomic_json(path, checkpoint)
    asyncio.run(run_batch(dataset(), tmp_path, resume=True, engine_factory=factory))
    assert len(calls) == 5 and calls[-1] == ("v2", "second")
    final = json.loads(path.read_text(encoding="utf-8"))
    assert len(final["state"]["session"]["messages"]) == 4
    altered = dataset()
    altered["cases"][0]["turns"][0]["user"] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        asyncio.run(run_batch(altered, tmp_path, resume=True, engine_factory=factory))


def test_inflight_turn_never_automatically_replayed(tmp_path):
    calls = []
    factory = lambda cfg, case, journal, circuit: FakeEngine(calls, cfg["engine"])
    asyncio.run(run_batch(dataset(), tmp_path, engine_factory=factory))
    path = tmp_path / "cases" / "case-0-v2.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint.update(phase="inflight", next_turn=0, turns=[])
    checkpoint["state"]["session"] = {}
    atomic_json(path, checkpoint)
    result = asyncio.run(run_batch(dataset(), tmp_path, resume=True, engine_factory=factory))
    assert len(calls) == 5 and calls[-1] == ("v2", "second")
    row = next(r for r in result["results"] if r["configuration"]["engine"] == "v2")
    assert row["turns"][0]["error"] == "unknown_inflight_no_automatic_replay"
    assert [m["content"] for m in row["state"]["session"]["messages"]].count("first") == 1


class FakeProvider(OfflineModel):
    def __init__(self, failures=()):
        super().__init__()
        self.config = ModelConfig()
        self.failures = list(failures)
        self.calls = 0
        self._usage_observer = None

    async def complete(self, messages, purpose="reply"):
        self.calls += 1
        if self.failures:
            raise ModelError(self.failures.pop(0))
        if self._usage_observer:
            self._usage_observer({"usage": {"input_tokens": 10, "output_tokens": 5}, "model": "gpt-5-mini", "status": "completed", "config": self.config.public_config()})
        return await super().complete(messages, purpose)


def test_retry_budget_and_durable_unknown_attempts(tmp_path):
    async def scenario():
        journal, circuit = Journal(tmp_path), Circuit()
        provider = FakeProvider(["rate_limit", "timeout"])
        delays = []
        async def sleep(delay):
            delays.append(delay)
        model = DurableModel(provider, journal, circuit, {"engine": "v2", "case_id": "one"}, sleeper=sleep)
        await model.complete([{"role": "user", "content": "first"}], "reply")
        assert provider.calls == 3 and delays == [1, 2]
        assert len(journal.events) == 3
        assert journal.cost()["unknown_usage_calls"] == 2
        assert len({e["attempt_id"] for e in journal.events.values()}) == 3
        reread = Journal(tmp_path)
        assert len(reread.events) == 3
        assert len(reread.ledger.path.read_text().splitlines()) == 3
        no_retry = FakeProvider(["invalid_output"])
        model = DurableModel(no_retry, journal, circuit, {"engine": "v2", "case_id": "two"}, sleeper=sleep)
        with pytest.raises(ModelError, match="invalid_output"):
            await model.complete([{"role": "user", "content": "first"}], "reply")
        assert no_retry.calls == 1
    asyncio.run(scenario())


def test_quota_bypasses_core_fallback_and_stops_new_requests(tmp_path):
    providers = []
    quota = True
    def factory(cfg, case, journal, circuit):
        provider = FakeProvider(["insufficient_quota"] if quota and not providers else [])
        providers.append(provider)
        model = DurableModel(provider, journal, circuit, {"engine": cfg["engine"], "case_id": case["id"]})
        return CounselingEngine(model, FaultStore())
    result = asyncio.run(run_batch(dataset(2), tmp_path, engine_factory=factory, concurrency=1))
    assert result["state"] == "needs_input" and result["pause_reason"] == "insufficient_quota"
    assert sum(p.calls for p in providers) == 1
    assert result["processed_turns"] == 1 and result["summary"]["unprocessed"] == 7
    assert all(r["phase"] == "waiting_quota" for r in result["results"])
    quota = False
    result = asyncio.run(run_batch(dataset(2), tmp_path, resume=True, engine_factory=factory, concurrency=1))
    assert result["state"] == "complete" and result["processed_turns"] == 8
    failed = next(r for r in result["results"] if r["case_id"] == "case-0" and r["configuration"]["engine"] == "baseline")
    assert failed["turns"][0]["status"] == "error"
    assert result["cost_summary"]["unknown_usage_calls"] == 1


def test_setup_faults_and_recovery_are_checkpointed(tmp_path):
    cases = dataset()
    cases["cases"][0]["setup"] = {"handoff_status": "pending", "messages": []}
    cases["cases"][0]["turns"] = [{"user": "accepted", "action": "acknowledge"},
                                    {"user": "resumed", "action": "resume", "fault": "model"},
                                    {"user": "last", "action": "reset"}]
    def factory(cfg, case, journal, circuit):
        return CounselingEngine(OfflineModel(), FaultStore())
    result = asyncio.run(run_batch(cases, tmp_path, engine_factory=factory))
    for row in result["results"]:
        assert row["turns"][0]["result"]["handoff_status"] == "accepted"
        assert row["turns"][1]["status"] == "error"
        assert len(row["state"]["session"]["messages"]) == 2
    assert result["summary"]["error"] == 2


def test_live_process_lock_verified_and_released(tmp_path):
    assert process_token(os.getpid()) is not None
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="live"):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):
        pass
    assert json.loads((tmp_path / "run.owner.json").read_text())["active"] is False


def test_atomic_attempt_recovery_from_partial_ledger(tmp_path):
    journal = Journal(tmp_path)
    journal.save({"attempt_id": "known", "engine": "v2", "case_id": "one", "turn": 0,
                  "phase": "response_observed", "provider_response": True,
                  "usage": {"input_tokens": 10, "output_tokens": 5}})
    with journal.ledger.path.open("a") as handle:
        handle.write('{"broken":')
    recovered = Journal(tmp_path)
    assert recovered.events["known"]["uncertain"] is True
    assert recovered.cost()["unknown_usage_calls"] == 0
    lines = recovered.ledger.path.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["attempt_id"] == "known"


def test_ordinary_factory_failure_is_durable_and_other_cases_continue(tmp_path):
    calls = []
    def bad_factory(cfg, case, journal, circuit):
        if cfg["engine"] == "v2" and case["id"] == "case-0":
            raise ValueError("synthetic factory error")
        return FakeEngine(calls, cfg["engine"])
    job = asyncio.run(run_batch(dataset(2), tmp_path, engine_factory=bad_factory))
    assert job["state"] == "incomplete" and job["summary"]["unprocessed"] == 2
    assert len(calls) == 6
    assert (tmp_path / "execution_errors" / "case-0-v2.json").exists()
    fixed_factory = lambda cfg, case, journal, circuit: FakeEngine(calls, cfg["engine"])
    job = asyncio.run(run_batch(dataset(2), tmp_path, resume=True, engine_factory=fixed_factory))
    assert job["state"] == "complete" and len(calls) == 8


def test_failed_save_recovery_hydrates_after_resume(tmp_path):
    class BrokenSave(FaultStore):
        async def save(self, user_id, state):
            raise ConnectionError("synthetic save failure")
    def factory(cfg, case, journal, circuit):
        return CounselingEngine(OfflineModel(), BrokenSave())
    asyncio.run(run_batch(dataset(), tmp_path, engine_factory=factory))
    path = tmp_path / "cases" / "case-0-v2.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    assert len(row["state"]["recovery"]["messages"]) == 4
    row.update(phase="ready", next_turn=1, turns=row["turns"][:1])
    row["state"]["recovery"]["messages"] = row["state"]["recovery"]["messages"][:2]
    atomic_json(path, row)
    job = asyncio.run(run_batch(dataset(), tmp_path, resume=True, engine_factory=factory))
    resumed = next(r for r in job["results"] if r["configuration"]["engine"] == "v2")
    assert len(resumed["state"]["recovery"]["messages"]) == 4
    assert resumed["turns"][1]["status"] == "error"


def test_concurrent_quota_does_not_consume_untouched_case_turns(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        providers = []
        class ConcurrentProvider(FakeProvider):
            def __init__(self, case_id):
                super().__init__()
                self.case_id = case_id
            async def complete(self, messages, purpose="reply"):
                if self.case_id == "case-0":
                    await gate.wait()
                    self.calls += 1
                    raise ModelError("insufficient_quota")
                if self.case_id == "case-1" and self.calls == 0:
                    gate.set()
                    await asyncio.sleep(0)
                return await super().complete(messages, purpose)
        def factory(cfg, case, journal, circuit):
            provider = ConcurrentProvider(case["id"])
            providers.append(provider)
            model = DurableModel(provider, journal, circuit, {"engine": cfg["engine"], "case_id": case["id"]})
            return CounselingEngine(model, FaultStore())
        result = await run_batch(dataset(4), tmp_path, concurrency=2, engine_factory=factory)
        assert result["state"] == "needs_input"
        assert sum(p.calls for p in providers) == 2
        assert result["processed_turns"] == 2
        untouched = [r for r in result["results"] if r["case_id"] in {"case-2", "case-3"}]
        assert all(r["next_turn"] == 0 and r["turns"] == [] and r["phase"] == "waiting_quota" for r in untouched)
    asyncio.run(scenario())
