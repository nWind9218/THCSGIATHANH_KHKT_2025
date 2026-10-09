"""Durable full-dataset evaluation. Checkpoints are local, synthetic and resumable."""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import time
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from mimi_v2.datasets import digest, load_dataset, validate_dataset
from mimi_v2.evaluation import check_turn, execute_local_turn, prepare_local_case, source_provenance, review_template
from mimi_v2.pilot import MODEL_SETTINGS, UsageLedger


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_token(pid: int) -> str | None:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            code = ctypes.c_uint32()
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            times = [ctypes.c_uint64() for _ in range(4)]
            kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                raise RuntimeError("Cannot verify process creation time")
            return str(times[0].value)
        finally:
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        raise RuntimeError("Cannot safely verify lock owner") from None
    proc = Path(f"/proc/{pid}/stat")
    if proc.exists():
        return proc.read_text().rsplit(")", 1)[1].split()[19]
    # Portable fallback conservatively treats a live PID as owned.
    return "live"


class RunLock:
    def __init__(self, output: Path):
        self.path = output / "run.lock"
        self.owner_path = output / "run.owner.json"
        self.owner = {"pid": os.getpid(), "token": process_token(os.getpid()), "active": True}
        self.handle = None

    def __enter__(self):
        # OS-held byte/advisory lock closes the stale-owner race that exclusive
        # file creation alone cannot prevent. Crashes release it automatically.
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b" ")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError("This batch already has a verified live process lock") from None
        try:
            if self.owner_path.exists():
                old = json.loads(self.owner_path.read_text(encoding="utf-8"))
                token = process_token(int(old["pid"]))
                if old.get("active") and token is not None and token == old.get("token"):
                    raise RuntimeError("Previous batch owner PID is still live; refusing ambiguous resume")
            atomic_json(self.owner_path, self.owner)
        except BaseException:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *args):
        try:
            atomic_json(self.owner_path, {**self.owner, "active": False})
        finally:
            if self.handle:
                self.handle.close()
                self.handle = None


class JobPause(BaseException):
    """Bypass workflow fallback handlers: unavailable quota is not a safe answer."""


class Circuit:
    def __init__(self):
        self.reason = None


class Journal:
    def __init__(self, output: Path):
        self.directory = output / "attempts"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ledger = UsageLedger(output / "usage.jsonl")
        self.seen = set()
        self.events = {}
        self.failed = False
        for path in sorted(self.directory.glob("*.json")):
            event = json.loads(path.read_text(encoding="utf-8"))
            if event.get("phase") != "complete":
                event.update(phase="complete", error="unknown_inflight", uncertain=True)
                atomic_json(path, event)
            self.events[event["attempt_id"]] = event
        # Reconstruct once from authoritative atomic files, including recovery
        # from a truncated JSONL tail without appending onto malformed bytes.
        temporary = self.ledger.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for event in self.events.values():
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.ledger.path)
        self.seen = set(self.events)

    def save(self, event):
        self.events[event["attempt_id"]] = deepcopy(event)
        try:
            atomic_json(self.directory / (event["attempt_id"] + ".json"), event)
        except OSError:
            self.failed = True
            raise

    def finalize(self, event):
        self.save(event)
        if event["attempt_id"] not in self.seen:
            self.ledger.append(event)
            if self.ledger.write_failed:
                self.failed = True
                raise OSError("Cannot durably persist usage ledger")
            # UsageLedger writes synchronously; strengthen durability before
            # checkpointing a completed turn.
            with self.ledger.path.open("a", encoding="utf-8") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            self.seen.add(event["attempt_id"])

    def cost(self):
        self.ledger.events = list(self.events.values())
        return self.ledger.summary()


class DurableModel:
    def __init__(self, provider, journal: Journal, circuit: Circuit, labels: dict, sleeper=asyncio.sleep):
        self.provider, self.journal, self.circuit, self.labels = provider, journal, circuit, labels
        self.config, self.sleeper = provider.config, sleeper
        self.current = None
        self.turn = 0
        provider._usage_observer = self.observe

    def observe(self, event):
        self.current.update(provider_response=True, usage=event["usage"], model=event["model"],
                            provider_status=event["status"], phase="response_observed")
        self.journal.save(self.current)

    async def complete(self, messages, purpose="reply"):
        for retry in range(3):
            if self.circuit.reason:
                raise JobPause(self.circuit.reason)
            event = {**self.labels, "turn": self.turn, "purpose": purpose, "attempt_id": uuid.uuid4().hex,
                     "retry": retry, "phase": "inflight", "provider_response": False, "usage": {},
                     "model": self.config.model, "config": self.config.public_config()}
            self.current = event
            self.journal.save(event)
            error = None
            try:
                reply = await self.provider.complete(messages, purpose=purpose)
            except Exception as exc:
                error = exc
                event["error"] = getattr(exc, "code", type(exc).__name__)
            finally:
                event["phase"] = "complete"
                try:
                    self.journal.finalize(event)
                except OSError:
                    self.circuit.reason = "checkpoint_unavailable"
                    raise
                self.current = None
            if self.journal.failed:
                self.circuit.reason = "checkpoint_unavailable"
                raise JobPause(self.circuit.reason)
            if error is None:
                return reply
            code = event["error"]
            if code in {"insufficient_quota", "authentication"}:
                self.circuit.reason = code
                raise JobPause(code)
            transient = code in {"rate_limit", "network_error", "timeout"} or (code == "provider_error" and getattr(error, "status", None) in {500, 502, 503, 504})
            if not transient or retry == 2:
                raise error
            await self.sleeper(2 ** retry)


def factory_live(config, case, journal, circuit, key):
    from mimi_v2.models import OpenAIModel, ModelConfig
    from mimi_v2.core import CounselingEngine
    from mimi_v2.baseline import BaselineEngine
    from mimi_v2.simulator import FaultStore
    model = DurableModel(OpenAIModel(ModelConfig(**MODEL_SETTINGS), api_key=key), journal, circuit,
                         {"engine": config["engine"], "case_id": case["id"]})
    return (BaselineEngine if config["engine"] == "baseline" else CounselingEngine)(model, FaultStore())


def checkpoint_state(engine, user_id):
    return {"session": deepcopy(engine.store.sessions.get(user_id, {})),
            "recovery": deepcopy(getattr(engine, "_recovery", {}).get(user_id)),
            "knowledge": deepcopy(engine.store.knowledge), "alerts": deepcopy(engine.store.alerts)}


def waiting_phase(reason):
    return {"insufficient_quota": "waiting_quota", "authentication": "waiting_credential"}.get(reason, "waiting_infrastructure")


def hydrate(engine, user_id, state):
    engine.store.sessions[user_id] = deepcopy(state["session"])
    engine.store.knowledge = deepcopy(state.get("knowledge", {}))
    engine.store.alerts = deepcopy(state.get("alerts", []))
    if state.get("recovery") is not None and hasattr(engine, "_recovery"):
        engine._recovery[user_id] = deepcopy(state["recovery"])


def unknown_turn(case, checkpoint, append_input=True):
    index = checkpoint["next_turn"]
    entry = {"index": index, "input": case["turns"][index]["user"], "status": "error",
             "error": "unknown_inflight_no_automatic_replay", "human_review": None}
    checkpoint["turns"].append(entry)
    # No invented assistant output. Retain the unacknowledged user input so the
    # remaining conversation can refer to it without repeating a paid request.
    state = checkpoint["state"]
    session = state.get("recovery") or state["session"]
    messages = session.setdefault("messages", [])
    if append_input:
        messages.append({"role": "user", "content": entry["input"]})
    session["messages"] = messages[-20:]
    checkpoint.update(next_turn=index + 1, phase="ready")


async def run_batch(dataset: dict, output: Path, *, key=None, resume=False, concurrency=8, engine_factory=None):
    validate_dataset(dataset)
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise ValueError("Concurrency must be 1..16")
    if engine_factory is None and (len(dataset["cases"]) != 1000 or not key):
        raise ValueError("Live batch requires authorized key and exactly 1000 cases")
    output.mkdir(parents=True, exist_ok=True)
    with RunLock(output):
        from mimi_v2.baseline import BaselineEngine
        configs = [{"engine": e, "live": True, **MODEL_SETTINGS} for e in ("baseline", "v2")]
        identity = {"dataset_hash": digest(dataset), "config_hash": digest(configs),
                    "source_hashes": source_provenance(), "legacy_sources": BaselineEngine.provenance()["source_hashes"]}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Existing job requires --resume; completed turns will not be replayed")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["identity"] != identity:
                raise ValueError("Frozen dataset/config/source mismatch; refusing paid resume")
        else:
            if resume:
                raise ValueError("No existing manifest to resume")
            manifest = {"job_id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                        "identity": identity, "case_count": len(dataset["cases"]), "configurations": configs}
            atomic_json(manifest_path, manifest)
            atomic_json(output / "dataset.json", dataset)
        journal, circuit = Journal(output), Circuit()
        semaphore = asyncio.Semaphore(concurrency)
        finished_pairs = 0

        def construct(config, case):
            return engine_factory(config, case, journal, circuit) if engine_factory else factory_live(config, case, journal, circuit, key)

        async def pair(case, config):
            nonlocal finished_pairs
            path = output / "cases" / f"{case['id']}-{config['engine']}.json"
            user_id = f"batch-{case['id']}-{config['engine']}"
            checkpoint = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if checkpoint and checkpoint["phase"] == "complete":
                finished_pairs += 1
                return checkpoint
            engine = construct(config, case)
            if checkpoint:
                if checkpoint["phase"] == "inflight":
                    unknown_turn(case, checkpoint)
                    atomic_json(path, checkpoint)
                hydrate(engine, user_id, checkpoint["state"])
            else:
                environment = await prepare_local_case(engine, user_id, case)
                checkpoint = {"case_id": case["id"], "category": case.get("category"), "configuration": config,
                              "config_hash": digest(config), "environment": environment, "next_turn": 0,
                              "phase": "ready", "turns": [], "state": checkpoint_state(engine, user_id)}
                atomic_json(path, checkpoint)
            for index in range(checkpoint["next_turn"], len(case["turns"])):
                if circuit.reason:
                    checkpoint.update(phase=waiting_phase(circuit.reason), pause_reason=circuit.reason)
                    atomic_json(path, checkpoint)
                    return checkpoint
                turn = case["turns"][index]
                if turn.get("action") == "reset":
                    engine = construct(config, case)
                if hasattr(engine.model, "turn"):
                    engine.model.turn = index
                checkpoint.update(phase="inflight", next_turn=index, state=checkpoint_state(engine, user_id))
                atomic_json(path, checkpoint)
                started = time.perf_counter()
                try:
                    result = await execute_local_turn(engine, user_id, turn)
                    checks = check_turn(result, turn.get("expect", {}))
                    diagnostics = result.get("diagnostics", [])
                    errors = result.get("error") or result.get("errors", []) or (diagnostics.get("errors", []) if isinstance(diagnostics, dict) else diagnostics)
                    status = "error" if errors else "fail" if any(not c["passed"] for c in checks) else "pending_review" if turn.get("review") or not checks else "pass"
                    entry = {"index": index, "input": turn["user"], "output": result.get("response_text", result.get("response", "")),
                             "result": result, "checks": checks, "status": status, "review_criteria": turn.get("review", []), "human_review": None}
                except JobPause:
                    # Only this case's attempts determine whether its turn was
                    # attempted; other concurrent cases must not consume it.
                    attempted = any(e["case_id"] == case["id"] and e["engine"] == config["engine"] and e["turn"] == index for e in journal.events.values())
                    if attempted:
                        actual = checkpoint_state(engine, user_id)
                        changed = actual != checkpoint["state"]
                        if changed:
                            # Legacy save_memory writes history before its final
                            # model call; preserve that real state if quota fails.
                            checkpoint["state"] = actual
                        unknown_turn(case, checkpoint, append_input=not changed)
                        checkpoint["turns"][-1]["error"] = "provider_pause_incomplete_turn: " + str(circuit.reason)
                    checkpoint.update(phase=waiting_phase(circuit.reason), pause_reason=circuit.reason)
                    atomic_json(path, checkpoint)
                    return checkpoint
                except Exception as exc:
                    entry = {"index": index, "input": turn["user"], "status": "error", "error": getattr(exc, "code", type(exc).__name__), "human_review": None}
                entry["latency_seconds"] = round(time.perf_counter() - started, 6)
                checkpoint["turns"].append(entry)
                checkpoint.update(next_turn=index + 1, phase="ready", state=checkpoint_state(engine, user_id))
                atomic_json(path, checkpoint)
                with (output / "turn_progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"case_id": case["id"], "engine": config["engine"], "turn": index, "status": entry["status"]}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            checkpoint["phase"] = "complete"
            atomic_json(path, checkpoint)
            finished_pairs += 1
            if finished_pairs % 10 == 0:
                print(f"Completed {finished_pairs}/{len(dataset['cases'])*2} case-configs", flush=True)
            return checkpoint

        async def case_job(case):
            async with semaphore:
                rows = []
                for config in configs:
                    try:
                        rows.append(await pair(case, config))
                    except Exception as exc:
                        # Preserve an existing inflight checkpoint on IO/factory
                        # failure. Other cases continue; never overwrite evidence.
                        failure = {"case_id": case["id"], "category": case.get("category"), "configuration": config,
                                     "config_hash": digest(config), "phase": "execution_error", "next_turn": 0,
                                     "turns": [], "execution_error": type(exc).__name__}
                        # Separate artifact preserves failures without replacing
                        # an inflight checkpoint needed for no-replay recovery.
                        try:
                            atomic_json(output / "execution_errors" / f"{case['id']}-{config['engine']}.json", failure)
                        except OSError:
                            circuit.reason = "checkpoint_unavailable"
                        rows.append(failure)
                return rows
        records = [row for rows in await asyncio.gather(*(case_job(c) for c in dataset["cases"])) for row in rows]
        expected = sum(len(c["turns"]) * 2 for c in dataset["cases"])
        counts = Counter(t["status"] for row in records for t in row["turns"])
        completed_turns = sum(counts.values())
        counts["unprocessed"] = expected - completed_turns
        job = {**manifest, "schema_version": 1, "dataset_name": dataset.get("name"), "dataset_hash": identity["dataset_hash"],
               "dataset_provenance": dataset.get("provenance", {}), "results": records, "summary": dict(counts),
               "expected_turns": expected, "processed_turns": completed_turns, "cost_summary": journal.cost(),
               "acceptance": "pending_human_review", "state": "needs_input" if circuit.reason else "incomplete" if counts["unprocessed"] else "complete",
               "pause_reason": circuit.reason}
        atomic_json(output / "report.json", job)
        atomic_json(output / "cost_summary.json", job["cost_summary"])
        atomic_json(output / "human_review.json", review_template(job))
        print(f"Batch state={job['state']}; turns saved={completed_turns}/{expected}; report={output / 'report.json'}", flush=True)
        return job


def main():
    parser = argparse.ArgumentParser(description="Durable 1000-case baseline/v2 job; no replay of completed turns")
    parser.add_argument("--dataset", default="eval_datasets/v2_acceptance_1000.json")
    parser.add_argument("--output", default="eval_results/full_1000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("Full paid batch requires explicit --live")
    from mimi_v2.simulator import load_local_model_environment
    key = load_local_model_environment().get("OPENAI_API_KEY")
    job = asyncio.run(run_batch(load_dataset(args.dataset), Path(args.output), key=key, resume=args.resume, concurrency=args.concurrency))
    if job["state"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
