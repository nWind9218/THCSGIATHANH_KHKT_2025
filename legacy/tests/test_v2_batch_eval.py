import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from legacy.eval_tools.batch_eval import Circuit, DurableModel, Journal, JobPause, RunLock, atomic_json, process_token, run_batch
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
        # Safety and assessment are dispatched concurrently, so each of the 2 turns made 2 provider attempts.
        assert sum(p.calls for p in providers) == 4
        assert result["processed_turns"] == 2
        untouched = [r for r in result["results"] if r["case_id"] in {"case-2", "case-3"}]
        assert all(r["next_turn"] == 0 and r["turns"] == [] and r["phase"] == "waiting_quota" for r in untouched)
    asyncio.run(scenario())


def test_total_budget_three_with_four_concurrent_cases_retries_and_resume(tmp_path):
    providers = []
    async def no_backoff(_delay):
        pass
    def factory(cfg, case, journal, circuit):
        provider = FakeProvider(["rate_limit", "network_error"] if not providers else [])
        providers.append(provider)
        model = DurableModel(provider, journal, circuit, {"engine": cfg["engine"], "case_id": case["id"]}, sleeper=no_backoff)
        return CounselingEngine(model, FaultStore())
    job = asyncio.run(run_batch(dataset(4), tmp_path, engine_factory=factory, concurrency=4, request_cap=3))
    assert sum(p.calls for p in providers) == 3
    assert job["pause_reason"] == "request_budget"
    assert all(r["phase"] == "waiting_budget" for r in job["results"])
    assert job["request_budget"]["reserved_attempts"] == 3
    assert job["request_budget"]["remaining_attempts"] == 0
    assert job["summary"].get("pass", 0) == 0
    assert job["processed_turns"] == 1  # The partial model chain is honest error.
    ledger = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("*.json")]
    ordered = sorted(ledger, key=lambda e: e["request_sequence"])
    assert [e["request_sequence"] for e in ordered] == [1, 2, 3]
    assert [e["retry"] for e in ordered] == [0, 1, 2]
    assert [e.get("error") for e in ordered[:2]] == ["rate_limit", "network_error"]
    timestamps = [datetime.fromisoformat(e["request_started_at_utc"]) for e in ordered]
    assert timestamps == sorted(timestamps)
    assert all(datetime.fromisoformat(e["request_reserved_at_utc"]) <= datetime.fromisoformat(e["request_started_at_utc"]) for e in ordered)
    for cap in (3, 2, 0):
        resumed = asyncio.run(run_batch(dataset(4), tmp_path, resume=True, engine_factory=factory, concurrency=4, request_cap=cap))
        assert sum(p.calls for p in providers) == 3
        assert resumed["request_budget"]["reserved_attempts"] == 3
        assert resumed["request_budget"]["remaining_attempts"] == 0
    extended = asyncio.run(run_batch(dataset(4), tmp_path, resume=True, engine_factory=factory, concurrency=4, request_cap=4))
    assert sum(p.calls for p in providers) == 4
    assert extended["request_budget"]["reserved_attempts"] == 4
    assert Journal(tmp_path, request_cap=4).sequence == 4


def test_live_cap_required_before_credentials_and_old_identity_not_loosened(tmp_path):
    result = subprocess.run([sys.executable, "-m", "legacy.eval_tools.batch_eval", "--live", "--resume"], capture_output=True, text=True)
    assert result.returncode == 2 and "--request-cap" in result.stderr
    with pytest.raises(ValueError, match="explicit total request cap"):
        asyncio.run(run_batch(dataset(), tmp_path / "missing-cap", key="fake-do-not-use"))
    calls = []
    factory = lambda cfg, case, journal, circuit: FakeEngine(calls, cfg["engine"])
    asyncio.run(run_batch(dataset(), tmp_path / "old-job", engine_factory=factory))
    path = tmp_path / "old-job" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["source_hashes"]["batch_eval.py"] = "old-frozen-source"
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="Frozen dataset/config/source mismatch"):
        asyncio.run(run_batch(dataset(), path.parent, resume=True, engine_factory=factory, request_cap=3))
    assert len(calls) == 4


def test_historical_order_is_never_fabricated(tmp_path):
    journal = Journal(tmp_path)
    journal.save({"attempt_id": "old-unordered", "engine": "v2", "case_id": "one", "turn": 0,
                  "phase": "complete", "provider_response": False, "usage": {}, "error": "timeout"})
    recovered = Journal(tmp_path, request_cap=1)
    assert "request_sequence" not in recovered.events["old-unordered"]
    assert "request_started_at_utc" not in recovered.events["old-unordered"]
    assert recovered.budget()["historical_order_unknown_attempts"] == 1
    circuit = Circuit()
    with pytest.raises(JobPause):
        recovered.reserve({"attempt_id": "blocked"}, circuit)
    assert len(recovered.events) == 1 and circuit.reason == "request_budget"


def test_budget_reservation_with_genuinely_overlapping_provider_attempts(tmp_path):
    async def scenario():
        tracker = {"calls": 0, "inflight": 0, "maximum_overlap": 0, "case_ids": []}
        class YieldingProvider(FakeProvider):
            def __init__(self, case_id):
                super().__init__()
                self.case_id = case_id
            async def complete(self, messages, purpose="reply"):
                tracker["calls"] += 1
                tracker["case_ids"].append(self.case_id)
                tracker["inflight"] += 1
                tracker["maximum_overlap"] = max(tracker["maximum_overlap"], tracker["inflight"])
                try:
                    # A real scheduling point: another case can reserve/start
                    # while this HTTP-like attempt has not returned yet.
                    await asyncio.sleep(0)
                    return await super().complete(messages, purpose)
                finally:
                    tracker["inflight"] -= 1
        def factory(cfg, case, journal, circuit):
            provider = YieldingProvider(case["id"])
            model = DurableModel(provider, journal, circuit, {"engine": cfg["engine"], "case_id": case["id"]})
            return CounselingEngine(model, FaultStore())
        job = await run_batch(dataset(4), tmp_path, engine_factory=factory, concurrency=4, request_cap=3)
        assert tracker["calls"] == 3
        assert tracker["maximum_overlap"] >= 2
        # One turn's safety+assessment pair are concurrent attempts, so 3 reserved attempts span at least 2 cases.
        assert len(set(tracker["case_ids"])) >= 2
        assert "case-3" not in tracker["case_ids"]
        assert job["pause_reason"] == "request_budget"
        events = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("*.json")]
        assert sorted(e["request_sequence"] for e in events) == [1, 2, 3]
        untouched = [r for r in job["results"] if r["case_id"] == "case-3"]
        assert all(r["phase"] == "waiting_budget" and r["turns"] == [] for r in untouched)
        resumed = await run_batch(dataset(4), tmp_path, resume=True, engine_factory=factory, concurrency=4, request_cap=3)
        assert tracker["calls"] == 3
        assert resumed["request_budget"]["reserved_attempts"] == 3
        assert resumed["request_budget"]["remaining_attempts"] == 0
    asyncio.run(scenario())
