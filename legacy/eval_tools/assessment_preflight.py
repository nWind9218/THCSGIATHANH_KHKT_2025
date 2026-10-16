"""Replay the six pilot assessment failures once, without production services."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .contracts import normalize_assessment, parse_decision
from .core import Assessment
from .models import ModelConfig, OpenAIModel
from .pilot import MeteredModel, MODEL_SETTINGS, UsageLedger
from .prompts import ASSESSMENT
from .simulator import load_local_model_environment


async def replay(source: Path, output: Path):
    report = json.loads(source.read_text(encoding="utf-8"))
    failures = []
    for row in report["results"]:
        if row["configuration"]["engine"] != "v2":
            continue
        previous = {}
        for turn in row["turns"]:
            if "assessment_unavailable_or_invalid" in turn.get("result", {}).get("diagnostics", []):
                messages = list(turn["result"]["messages"])
                if messages and messages[-1]["role"] == "assistant":
                    messages.pop()  # Exclude the fallback emitted after the failed decision.
                failures.append((row["case_id"], turn["index"], messages, dict(previous)))
            elif turn.get("result", {}).get("assessment"):
                previous = turn["result"]["assessment"]
    if len(failures) != 6:
        raise ValueError("Preflight expects the six observed pilot failures")
    if output.exists():
        raise ValueError("Preflight output already exists; no automatic paid replay")
    key = load_local_model_environment().get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Existing authorized API key required")
    output.mkdir(parents=True)
    ledger = UsageLedger(output / "usage.jsonl")
    results = []
    for case_id, index, messages, previous in failures:
        model = MeteredModel(OpenAIModel(ModelConfig(**MODEL_SETTINGS), api_key=key), ledger, "v2", case_id)
        context = {k: previous.get(k) for k in ("known", "topic", "core_issue", "need", "user_role")}
        prompt = ASSESSMENT + "\nĐã biết: " + json.dumps(context, ensure_ascii=False)
        try:
            reply = await model.complete([{"role": "system", "content": prompt}, *messages], purpose="assessment")
            normalized = Assessment.model_validate(normalize_assessment(parse_decision(reply.text)))
            entry = {"case_id": case_id, "turn": index, "contract_valid": True,
                     "need": normalized.need, "role": normalized.user_role,
                     "safety_hint": normalized.safety_hint, "known_is_dict": isinstance(normalized.known, dict)}
        except Exception as exc:
            entry = {"case_id": case_id, "turn": index, "contract_valid": False, "error_type": type(exc).__name__}
        results.append(entry)
        (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (output / "cost_summary.json").write_text(json.dumps(ledger.summary(), indent=2), encoding="utf-8")
        print(json.dumps(entry), flush=True)
    if not all(r["contract_valid"] for r in results):
        raise RuntimeError("Assessment preflight failed; do not launch the full job")
    print(json.dumps({"validated": len(results), "cost": ledger.summary()["observed_usage_estimate_usd"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="eval_results/live_pilot_20/6de33600246a4016b6e0d8278d72fdc5.json")
    parser.add_argument("--output", default="eval_results/assessment_preflight")
    args = parser.parse_args()
    asyncio.run(replay(Path(args.source), Path(args.output)))


if __name__ == "__main__":
    main()
