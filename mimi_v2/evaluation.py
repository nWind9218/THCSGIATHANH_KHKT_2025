"""Isolated multi-turn job runner with explicit automated and human evidence."""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mimi_v2.datasets import digest, load_dataset, validate_dataset


async def prepare_local_case(engine, user_id: str, case: dict) -> dict:
    """Only synthetic in-memory environments may receive setup or fault actions."""
    from mimi_v2.store import InMemoryStore
    if not isinstance(getattr(engine, "store", None), InMemoryStore):
        if case.get("setup") or case.get("knowledge") or any(
            t.get("action", "send") != "send" or t.get("fault", "none") != "none" for t in case["turns"]
        ):
            raise ValueError("Controlled cases require isolated InMemoryStore")
        return {}
    if case.get("setup"):
        await engine.store.save(user_id, case["setup"])
    if case.get("knowledge"):
        engine.store.knowledge.update(case["knowledge"])
    return {"initial_session": await engine.store.load(user_id), "knowledge_fixture": case.get("knowledge", {})}


async def execute_local_turn(engine, user_id: str, turn: dict) -> dict:
    from mimi_v2.store import InMemoryStore
    action, fault = turn.get("action", "send"), turn.get("fault", "none")
    controlled = action != "send" or fault != "none"
    if not controlled:
        return await engine.respond(user_id, turn["user"])
    if controlled and not isinstance(engine.store, InMemoryStore):
        raise ValueError("Controlled turns require isolated InMemoryStore")
    if action not in {"send", "reset"}:
        await getattr(engine.store, action)(user_id)
    if hasattr(engine.store, "fault"):
        engine.store.fault = fault
    original = engine.model
    if fault == "model":
        class TimeoutModel:
            async def complete(self, *args, **kwargs):
                raise TimeoutError("synthetic model outage")
        engine.model = TimeoutModel()
    try:
        result = await engine.respond(user_id, turn["user"])
        result["environment_action"] = {"action": action, "fault": fault}
        if controlled and fault != "store":
            result["observed_session"] = await engine.store.load(user_id)
        return result
    finally:
        engine.model = original
        if hasattr(engine.store, "fault"):
            engine.store.fault = "none"


def check_turn(result: dict, expected: dict) -> list[dict]:
    text = result.get("response_text", result.get("response", "")) or ""
    assessment = result.get("assessment") or {}
    values = {
        "intent": assessment.get("intent", result.get("intent_category")),
        "role": assessment.get("user_role", assessment.get("role")),
        "support_mode": assessment.get("need", assessment.get("support_mode")),
        "handoff": result.get("handoff_status"),
    }
    checks = []
    for key, wanted in expected.items():
        if key == "contains":
            actual = [s for s in wanted if s.casefold() not in text.casefold()]
            ok = not actual
        elif key == "not_contains":
            actual = [s for s in wanted if s.casefold() in text.casefold()]
            ok = not actual
        elif key == "max_questions":
            actual = text.count("?")
            ok = actual <= wanted
        elif key == "max_chars":
            actual = len(text)
            ok = actual <= wanted
        else:
            actual = values.get(key)
            ok = actual == wanted
        checks.append({"criterion": key, "expected": wanted, "actual": actual, "passed": ok})
    return checks


def source_provenance() -> dict:
    root = Path(__file__).resolve().parent
    return {p.name: digest(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.py"))}


async def run_job(dataset: dict, configurations: list[dict], engine_factory) -> dict:
    """Factory receives config and case; every invocation MUST create fresh storage."""
    validate_dataset(dataset)
    if not configurations or len({digest(c) for c in configurations}) != len(configurations):
        raise ValueError("Provide nonempty, distinct configurations")
    job = {
        "schema_version": 1, "job_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset.get("name", "unnamed"), "dataset_hash": digest(dataset),
        "configurations": configurations, "source_hashes": source_provenance(),
        "python": platform.python_version(), "results": [],
        "acceptance": "pending_human_review",
        "dataset_provenance": dataset.get("provenance", {}),
    }
    for config in configurations:
        for case in dataset["cases"]:
            record = {"case_id": case["id"], "configuration": config, "config_hash": digest(config), "turns": []}
            try:
                engine = engine_factory(config, case)
                if hasattr(engine, "provenance"):
                    record["engine_provenance"] = engine.provenance()
                if hasattr(getattr(engine, "model", None), "config"):
                    record["resolved_model_config"] = engine.model.config.public_config()
                else:
                    record["resolved_model_config"] = {"model": "deterministic-fixture", "api": "offline"}
                user_id = f"eval-{job['job_id']}-{case['id']}"
                record["category"] = case.get("category")
                record["environment"] = await prepare_local_case(engine, user_id, case)
                for index, turn in enumerate(case["turns"]):
                    started = time.perf_counter()
                    try:
                        if turn.get("action") == "reset":
                            engine = engine_factory(config, case)
                            # New engine/store clears process recovery too. Do not re-seed setup.
                            await prepare_local_case(engine, user_id, {"turns": []})
                        result = await execute_local_turn(engine, user_id, turn)
                        checks = check_turn(result, turn.get("expect", {}))
                        diagnostics = result.get("diagnostics", [])
                        errors = result.get("errors", []) or (diagnostics.get("errors", []) if isinstance(diagnostics, dict) else diagnostics)
                        status = "error" if errors else "fail" if any(not c["passed"] for c in checks) else "pending_review" if turn.get("review") or not checks else "pass"
                        entry = {"index": index, "input": turn["user"], "output": result.get("response_text", result.get("response", "")),
                                 "result": result, "checks": checks, "status": status,
                                 "review_criteria": turn.get("review", []), "human_review": None}
                    except Exception as exc:
                        entry = {"index": index, "input": turn["user"], "status": "error", "error": type(exc).__name__, "human_review": None}
                    entry["latency_seconds"] = round(time.perf_counter() - started, 6)
                    record["turns"].append(entry)
            except Exception as exc:
                record["turns"] = [{"index": i, "input": t["user"], "status": "error", "error": type(exc).__name__, "human_review": None} for i, t in enumerate(case["turns"])]
            job["results"].append(record)
    return summarize(job)


def summarize(job: dict) -> dict:
    totals = {s: 0 for s in ("pass", "fail", "pending_review", "error", "skipped")}
    for record in job["results"]:
        for turn in record["turns"]:
            totals[turn["status"]] += 1
    job["summary"] = totals
    job["summary"]["total"] = sum(totals.values())
    return job


def review_template(job: dict) -> dict:
    return {"job_id": job["job_id"], "ratings": [
        {"case_id": row["case_id"], "config_hash": row["config_hash"], "turn": t["index"],
         "verdict": None, "reviewer": "", "comment": "", "criteria": t.get("review_criteria", [])}
        for row in job["results"] for t in row["turns"]]}


def apply_reviews(job: dict, reviews: dict) -> dict:
    if reviews.get("job_id") != job["job_id"]:
        raise ValueError("Review job id does not match")
    targets = {(r["case_id"], r["config_hash"], t["index"]): t for r in job["results"] for t in r["turns"]}
    seen = set()
    for rating in reviews.get("ratings", []):
        key = (rating["case_id"], rating["config_hash"], rating["turn"])
        if key not in targets or key in seen:
            raise ValueError("Unknown or duplicate review target")
        seen.add(key)
        if rating.get("verdict") is None:
            continue
        if rating["verdict"] not in {"pass", "fail"} or not str(rating.get("reviewer", "")).strip():
            raise ValueError("Review requires pass/fail and named reviewer")
        turn = targets[key]
        turn["human_review"] = rating
        # Human approval cannot erase runtime errors or failed automated checks.
        if turn["status"] not in {"error", "skipped", "fail"}:
            turn["status"] = rating["verdict"]
    return summarize(job)


def write_report(job: dict, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{job['job_id']}.json"
    target.write_text(json.dumps(job, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Conversation eval", "", f"Job: `{job['job_id']}`", f"Dataset SHA256: `{job['dataset_hash']}`", "", "Acceptance remains subject to the user dataset and review.", "", "| Case | Config | Turn | Status | Latency (s) |", "|---|---|---:|---|---:|"]
    for row in job["results"]:
        for turn in row["turns"]:
            lines.append(f"| {row['case_id']} | {row['config_hash'][:12]} | {turn['index']} | {turn['status']} | {turn.get('latency_seconds', '-')} |")
    target.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.with_name(target.stem + ".review.json").write_text(json.dumps(review_template(job), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description="Isolated multi-turn eval (offline by default)")
    parser.add_argument("dataset", nargs="?", default="eval_datasets/v2_smoke.json")
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--api", choices=["responses", "chat_completions"], default="responses")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--compare", action="store_true", help="Compare actual isolated legacy graph and v2")
    parser.add_argument("--configs", help="JSON file containing a list of engine/API/model configurations")
    parser.add_argument("--review-job")
    parser.add_argument("--ratings")
    args = parser.parse_args()
    if args.review_job:
        if not args.ratings:
            parser.error("--ratings required with --review-job")
        job = json.loads(Path(args.review_job).read_text(encoding="utf-8"))
        apply_reviews(job, json.loads(Path(args.ratings).read_text(encoding="utf-8")))
    else:
        from mimi_v2.simulator import make_engine
        configs = json.loads(Path(args.configs).read_text(encoding="utf-8")) if args.configs else [{"engine": "v2", "api": args.api, "model": args.model, "live": args.live}]
        if args.compare and not args.configs:
            configs.insert(0, {**configs[0], "engine": "baseline"})
        if any(c.get("live") for c in configs) and not args.live:
            parser.error("Live configurations require explicit --live")
        job = asyncio.run(run_job(load_dataset(args.dataset), configs, lambda c, case: make_engine(c)))
    print(write_report(job, args.output))


if __name__ == "__main__":
    main()
