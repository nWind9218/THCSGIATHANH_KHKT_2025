"""Bounded, explicitly authorized 20-case live comparison and usage ledger."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from mimi_v2.datasets import digest, load_dataset, validate_dataset
from mimi_v2.evaluation import run_job, source_provenance, summarize, write_report

VARIANTS = ["continuation", "correction", "listen_only", "multi_intent", "deep_reflection"]
MODEL_SETTINGS = {"api": "responses", "model": "gpt-5-mini", "reasoning_effort": "minimal",
                  "verbosity": "low", "max_output_tokens": 2048, "timeout": 60.0}
RATES = {"input_per_million_usd": "0.25", "cached_input_per_million_usd": "0.025", "output_per_million_usd": "2.00",
         "source": "https://developers.openai.com/api/docs/models/gpt-5-mini", "verified_date": "2026-10-09",
         "tier": "standard", "reasoning": "Reasoning tokens are already included in output_tokens; not charged twice."}


def choose_pilot(development: dict) -> dict:
    validate_dataset(development)
    categories = list(dict.fromkeys(case.get("category") for case in development["cases"]))
    if len(categories) != 20 or any(not c for c in categories):
        raise ValueError("Expected 20 categorized development groups")
    selected = []
    for index, category in enumerate(categories):
        variant = VARIANTS[index % 5]
        candidates = [c for c in development["cases"] if c.get("category") == category
                      and c.get("variant") == variant and c.get("split") == "development"
                      and not c.get("setup") and not c.get("expected_infrastructure_error")
                      and all(t.get("action", "send") == "send" and t.get("fault", "none") == "none" for t in c["turns"])]
        if not candidates:
            raise ValueError("No eligible pilot scenario: " + category)
        # Deterministic preference for richer contexts, then stable case id.
        # Holdout/regression are never considered.
        candidates.sort(key=lambda c: (-max(len(t["user"].split()) for t in c["turns"]), c["id"]))
        selected.append(deepcopy(candidates[0]))
    result = {"version": 1, "name": "Authorized live pilot: 20 categories, baseline and v2",
              "provenance": {"source_dataset_hash": digest(development), "source_provenance": development.get("provenance", {}),
                             "selected_ids": [c["id"] for c in selected], "selection": "one/category; category index mod5 variant; longest eligible turn then id",
                             "holdout_used": False, "fault_or_setup_cases_used": False, "synthetic": True}, "cases": selected}
    return validate_dataset(result)


class UsageLedger:
    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict] = []
        self.write_failed = False

    def append(self, event: dict):
        self.events.append(event)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            self.write_failed = True

    def summary(self) -> dict:
        engines = {}
        for name in ("baseline", "v2"):
            events = [e for e in self.events if e["engine"] == name]
            totals = {"api_calls_attempted": len(events), "provider_responses": 0, "unknown_usage_calls": 0,
                      "failed_or_unvalidated_calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                      "output_tokens": 0, "reasoning_tokens_included_in_output": 0}
            for event in events:
                if event.get("provider_response"):
                    totals["provider_responses"] += 1
                if event.get("error"):
                    totals["failed_or_unvalidated_calls"] += 1
                usage = event.get("usage", {})
                inp, out = usage.get("input_tokens"), usage.get("output_tokens")
                cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                if type(inp) is not int or type(out) is not int or type(cached) is not int or min(inp, out, cached) < 0 or cached > inp:
                    totals["unknown_usage_calls"] += 1
                    continue
                totals["input_tokens"] += inp
                totals["cached_input_tokens"] += cached
                totals["output_tokens"] += out
                reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                if type(reasoning) is int and reasoning >= 0:
                    totals["reasoning_tokens_included_in_output"] += reasoning
            cost = ((Decimal(totals["input_tokens"] - totals["cached_input_tokens"]) * Decimal("0.25")
                     + Decimal(totals["cached_input_tokens"]) * Decimal("0.025")
                     + Decimal(totals["output_tokens"]) * Decimal("2")) / Decimal(1_000_000))
            totals["observed_usage_estimate_usd"] = str(cost.quantize(Decimal("0.00000001")))
            engines[name] = totals
        return {"rates": RATES, "engines": engines,
                "observed_usage_estimate_usd": str(sum(Decimal(e["observed_usage_estimate_usd"]) for e in engines.values())),
                "unknown_usage_calls": sum(e["unknown_usage_calls"] for e in engines.values()),
                "ledger_write_failed": self.write_failed,
                "billing_limit": "Estimate from observed provider usage, not a confirmed invoice. Failed requests without usage may have unknown billing."}


class MeteredModel:
    def __init__(self, model, ledger: UsageLedger, engine: str, case_id: str):
        self.model, self.ledger, self.engine, self.case_id = model, ledger, engine, case_id
        self.config = model.config
        self._current = None
        model._usage_observer = self.observe

    def observe(self, event):
        if self._current is not None:
            self._current.update(provider_response=True, usage=event["usage"], model=event["model"],
                                 provider_status=event["status"], config=event["config"])

    async def complete(self, messages, purpose="reply"):
        event = {"engine": self.engine, "case_id": self.case_id, "purpose": purpose,
                 "attempt_id": uuid.uuid4().hex, "provider_response": False, "usage": {},
                 "model": self.config.model, "config": self.config.public_config()}
        self._current = event
        try:
            return await self.model.complete(messages, purpose=purpose)
        except Exception as exc:
            event["error"] = getattr(exc, "code", type(exc).__name__)
            raise
        finally:
            self.ledger.append(event)
            self._current = None


def live_engine_factory(config, case, ledger, key):
    from mimi_v2.models import ModelConfig, OpenAIModel
    from mimi_v2.core import CounselingEngine
    from mimi_v2.baseline import BaselineEngine
    from mimi_v2.store import InMemoryStore
    model = MeteredModel(OpenAIModel(ModelConfig(**MODEL_SETTINGS), api_key=key), ledger, config["engine"], case["id"])
    return (BaselineEngine if config["engine"] == "baseline" else CounselingEngine)(model, InMemoryStore())


async def run_pilot(pilot: dict, output: Path, *, key=None, engine_factory=None, concurrency=4) -> dict:
    validate_dataset(pilot)
    if len(pilot["cases"]) != 20 or len({c.get("category") for c in pilot["cases"]}) != 20:
        raise ValueError("Pilot is strictly limited to 20 cases, one per category")
    if any(c.get("split") != "development" or c.get("setup") or any(t.get("fault", "none") != "none" or t.get("action", "send") != "send" for t in c["turns"]) for c in pilot["cases"]):
        raise ValueError("Pilot must use development conversations without setup/actions/faults")
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("Concurrency must be 1..4")
    if engine_factory is None and not key:
        raise ValueError("Existing authorized key is required")
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "usage.jsonl"
    if ledger_path.exists() or (output / "started.json").exists():
        raise ValueError("Pilot already started here; no automatic paid restart")
    configs = [{"engine": name, "live": True, **MODEL_SETTINGS} for name in ("baseline", "v2")]
    ledger = UsageLedger(ledger_path)
    started = time.perf_counter()
    job = {"schema_version": 1, "job_id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
           "dataset_name": pilot["name"], "dataset_hash": digest(pilot), "dataset_provenance": pilot.get("provenance", {}),
           "configurations": configs, "source_hashes": source_provenance(), "results": [], "acceptance": "pending_human_review"}
    (output / "started.json").write_text(json.dumps({k: v for k, v in job.items() if k != "results"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "pilot_dataset.json").write_text(json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)
    progress = []

    async def case_job(case):
        async with semaphore:
            records = []
            for config in configs:
                factory = (lambda c, item: engine_factory(c, item, ledger)) if engine_factory else (lambda c, item: live_engine_factory(c, item, ledger, key))
                one = await run_job({"version": 1, "name": pilot["name"], "cases": [case]}, [config], factory)
                records.extend(one["results"])
                progress.append({"case_id": case["id"], "engine": config["engine"], "summary": one["summary"], "elapsed_seconds": round(time.perf_counter() - started, 2)})
                (output / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
                (output / f"{case['id']}-{config['engine']}.checkpoint.json").write_text(json.dumps(one, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Completed {len(progress)}/40 case-configs; elapsed {time.perf_counter()-started:.0f}s", flush=True)
            return records

    results = await asyncio.gather(*(case_job(case) for case in pilot["cases"]))
    job["results"] = [row for rows in results for row in rows]
    summarize(job)
    job["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    job["cost_summary"] = ledger.summary()
    report = write_report(job, output)
    (output / "cost_summary.json").write_text(json.dumps(job["cost_summary"], indent=2), encoding="utf-8")
    print("Report: " + str(report), flush=True)
    return job


def main():
    parser = argparse.ArgumentParser(description="Exactly 20 development cases × baseline/v2, no retries")
    parser.add_argument("--dataset", default="eval_datasets/v2_acceptance_development.json")
    parser.add_argument("--output", default="eval_results/live_pilot_20")
    parser.add_argument("--live", action="store_true", help="Execute the already-authorized bounded paid pilot")
    args = parser.parse_args()
    pilot = choose_pilot(load_dataset(args.dataset))
    if not args.live:
        print(json.dumps({"selected_ids": pilot["provenance"]["selected_ids"], "cases": 20, "configurations": 2, "settings": MODEL_SETTINGS}, indent=2))
        return
    from mimi_v2.simulator import load_local_model_environment
    env = load_local_model_environment()
    if not env.get("OPENAI_API_KEY"):
        parser.error("Existing OPENAI_API_KEY is required")
    asyncio.run(run_pilot(pilot, Path(args.output), key=env["OPENAI_API_KEY"]))


if __name__ == "__main__":
    main()
