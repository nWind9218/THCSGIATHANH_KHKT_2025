"""Offline report of a stopped batch; never imports providers or loads credentials."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legacy.eval_tools.batch_eval import Journal, RunLock, atomic_json, unknown_turn
from legacy.eval_tools.pilot import UsageLedger
from legacy.eval_tools.evaluation import source_provenance


def summarize(output: Path, limit: int):
    if (output / "paused_by_user.json").exists():
        raise ValueError("Pause report already exists; preserving original stop and resume evidence")
    snapshot = output / "pause_snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if source_provenance() != manifest["identity"]["source_hashes"]:
        raise ValueError("Runtime source changed; checkpoint recovery must use frozen source")
    ordered = [json.loads(line) for line in (snapshot / "usage.original.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({e["attempt_id"] for e in ordered}) != len(ordered):
        raise ValueError("Duplicate ledger attempt IDs")
    selected = ordered[:limit]
    selected_ids = {e["attempt_id"] for e in selected}
    cases = {c["id"]: c for c in json.loads((output / "dataset.json").read_text(encoding="utf-8"))["cases"]}
    calls = defaultdict(list)
    raw_attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (snapshot / "attempts").glob("*.json")]
    for event in raw_attempts:
        calls[(event["case_id"], event["engine"], event["turn"])].append(event)
    bounded = []
    boundary = set()
    full_counts = Counter()
    resume_points = []
    # This recovers interrupted turns without any provider/model invocation.
    with RunLock(output):
        journal = Journal(output)
        for path in sorted((output / "cases").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            name = record["configuration"]["engine"]
            for turn in record["turns"]:
                full_counts[turn["status"]] += 1
                events = calls[(record["case_id"], name, turn["index"])]
                ids = {e["attempt_id"] for e in events}
                if ids & selected_ids:
                    if ids <= selected_ids:
                        bounded.append({"case_id": record["case_id"], "engine": name,
                                        "category": record["category"], "index": turn["index"],
                                        "status": turn["status"], "checks": turn.get("checks", []),
                                        "diagnostics": turn.get("result", {}).get("diagnostics", []),
                                        "error": turn.get("error"), "latency_seconds": turn.get("latency_seconds")})
                    else:
                        boundary.add((record["case_id"], name, turn["index"]))
            interrupted = record["phase"] == "inflight"
            if interrupted:
                # Original pre-turn checkpoint is retained in pause_snapshot.
                unknown_turn(cases[record["case_id"]], record)
                record["turns"][-1]["error"] = "user_stop_incomplete_turn_no_automatic_replay"
                atomic_json(path, record)
            resume_points.append({"case_id": record["case_id"], "engine": name,
                                  "next_turn": record["next_turn"], "phase": record["phase"],
                                  "interrupted_turn_preserved_as_error": interrupted})
        full_cost = journal.cost()
        pause = {"state": "paused_by_user", "job_id": manifest["job_id"],
                 "requested_request_limit": limit, "actual_attempts_at_stop": len(raw_attempts),
                 "completed_turn_statuses_before_recovery": dict(full_counts),
                 "resume_points": resume_points,
                 "resume_requires_explicit_user_authorization": True,
                 "resume_command": ".venv/Scripts/python.exe -u -m legacy.eval_tools.batch_eval --live --resume --output eval_results/full_1000 --concurrency 8",
                 "limitation": "Checkpoint is per turn, not per provider subcall. Interrupted turns are errors and are not automatically replayed."}
        atomic_json(output / "paused_by_user.json", pause)
        atomic_json(output / "cost_summary.paused.json", full_cost)
    ledger = UsageLedger(output / "unused")
    ledger.events = selected
    statuses = {name: dict(Counter(t["status"] for t in bounded if t["engine"] == name)) for name in ("baseline", "v2")}
    check_counts = Counter()
    for turn in bounded:
        for check in turn["checks"]:
            if not check["passed"]:
                check_counts[(turn["engine"], check["criterion"])] += 1
    report = {"job_id": manifest["job_id"], "request_window": len(selected),
              "window_order": "First finalized ledger records in preserved append order, NOT exact request-start order (no start timestamps recorded).",
              "request_ids_in_order": [e["attempt_id"] for e in selected],
              "cost_summary": ledger.summary(), "whole_turn_statuses": statuses,
              "categories": dict(Counter(t["category"] for t in bounded)),
              "boundary_partial_turns_excluded": [list(t) for t in sorted(boundary)],
              "failed_checks": [{"engine": k[0], "criterion": k[1], "count": v} for k, v in check_counts.items()],
              "turns": bounded, "acceptance": "pending_semantic_review",
              "limitations": ["Request count is not conversation/case count.",
                              "Only complete turns whose provider attempts all fall inside the request window are counted.",
                              "Zero-provider-call fault/control turns cannot be chronologically assigned to this window.",
                              "The prefix is not a balanced sample of all 20 categories.",
                              "Automatic checks do not prove empathy, intent fidelity, fact grounding or clinical safety."]}
    atomic_json(output / "first_3000_requests.report.json", report)
    lines = ["# Paused live evaluation — first 3,000 ledger records", "",
             f"Job: `{manifest['job_id']}`. State: paused by user; no automatic resume.", "",
             f"Requested limit: {limit}; actual attempt records before stop: {len(raw_attempts)}.",
             "The limit was requested after the job had already exceeded it; earlier billed requests cannot be undone.", "",
             report["window_order"], "",
             "| Engine | Requests | Known-usage estimated USD | Whole turns | Statuses |",
             "|---|---:|---:|---:|---|"]
    for name in ("baseline", "v2"):
        stats = report["cost_summary"]["engines"][name]
        lines.append(f"| {name} | {stats['api_calls_attempted']} | {stats['observed_usage_estimate_usd']} | {sum(statuses[name].values())} | {statuses[name]} |")
    lines += ["", f"Window known-usage estimate: **${report['cost_summary']['observed_usage_estimate_usd']}**; unknown usage requests: {report['cost_summary']['unknown_usage_calls']}.",
              f"Entire stopped job known-usage estimate: **${full_cost['observed_usage_estimate_usd']}**; unknown usage requests: {full_cost['unknown_usage_calls']}.",
              "These are token-usage estimates, not confirmed account balance deductions.", "",
              f"Categories (whole turns): {report['categories']}",
              f"Partial boundary turns excluded: {len(boundary)}.", "", "## Failed objective checks", ""]
    lines += [f"- {r['engine']}: {r['criterion']} = {r['count']}" for r in report["failed_checks"]]
    lines += ["", "## Interpretation", "", "Semantic acceptance remains pending. In-window audits found listen-only/closing confusion (0008 index 2), unsupported safety/relationship assertions (0051 index 1), duplicate known keys (0032 index 2 and 0057 index 4) and possible over-escalation. Other findings in AGENT_DESIGN_DOCS/MIMI_V2_IMPLEMENTATION.md may come from outside this request window; do not attribute those to this sample. The request prefix is not a paired/balanced comparison. No production rollout.", "", "## Resume", "", "Only after explicit user authorization, execute the command in paused_by_user.json. Completed turns are skipped; interrupted turns are preserved as incomplete errors with no paid replay. Resume continues from each persisted next_turn and restores the saved conversation state, but cannot resume inside a provider subcall.", ""]
    (output / "first_3000_requests.summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"actual_attempts": len(raw_attempts), "window_requests": len(selected),
                      "window_cost": report["cost_summary"], "full_cost": full_cost,
                      "turn_statuses": statuses, "categories": report["categories"],
                      "interrupted_turns": sum(p["interrupted_turn_preserved_as_error"] for p in resume_points)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval_results/full_1000"))
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()
    summarize(args.output, args.limit)
