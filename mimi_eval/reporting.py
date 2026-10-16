"""Report deterministic harness checks separately from semantic acceptance."""
from collections import Counter
import json
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    counts = Counter(r["result"]["verdict"] for r in rows)
    return {"cases": len(rows), "verdict_counts": {k: counts[k] for k in
                ("PASS", "FAIL", "NEEDS_REVIEW", "INVALID")},
            "technical_failures": sum(r["result"]["technical_failure"] for r in rows),
            "quality_failures": sum(any(f["code"] == "QUALITY_BELOW_THRESHOLD"
                                      for f in r["result"]["failure_flags"]) for r in rows),
            "need_compliance_failures": sum(any(f["code"] == "NEED_COMPLIANCE"
                                       for f in r["result"]["failure_flags"]) for r in rows),
            "invalid_observations": sum(r["result"]["verdict"] == "INVALID" and
                                       not r["result"]["technical_failure"] for r in rows),
            "scripted_expectation_mismatches": sum(not r["matches_scripted_expectation"] for r in rows),
            "human_gold_status": "UNREVIEWED", "semantic_agreement": "NOT_EVALUATED",
            "false_pass_rate": "NOT_EVALUATED", "safety_calibration": "NOT_EVALUATED",
            "medium_high_comparison": "NOT_EVALUATED", "provider_calls": 0,
            "production_readiness": "NOT_EVALUATED"}


def write_report(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier report or silently replace a reviewed artifact.
    if (directory / "offline-report.json").exists() or (directory / "offline-report.md").exists():
        raise FileExistsError("report_already_exists")
    (directory / "offline-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    lines = ["# Mimi v0.2 offline harness replay", "",
             "Scripted synthetic observations; all candidates UNREVIEWED. "
             "PASS here is a harness outcome, not human agreement or production safety.", "",
             f"Source manifest SHA256: `{report['manifest_sha256']}`", "",
             "| Verdict | Cases |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in summary["verdict_counts"].items()]
    lines += ["", f"Technical failures: {summary['technical_failures']}; "
              f"invalid observations: {summary['invalid_observations']}; "
              f"quality failures: {summary['quality_failures']}; "
              f"need-compliance failures: {summary['need_compliance_failures']}.", "",
              f"Scripted expectation mismatches: {summary['scripted_expectation_mismatches']}.", "",
              "Semantic agreement, false PASS/FAIL, safety calibration, medium/high comparison "
              "and production readiness: **NOT_EVALUATED**. Provider calls: **0**.", "",
              "| Case | Scope | Verdict | Failure flags | Review flags |", "|---|---|---|---|---|"]
    for row in report["rows"]:
        r = row["result"]
        lines.append(f"| {row['fixture_id']} | {r['evaluation_scope']} | {r['verdict']} | "
                     f"{', '.join(x['code'] for x in r['failure_flags'])} | "
                     f"{', '.join(x['code'] for x in r['review_flags'])} |")
    (directory / "offline-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
