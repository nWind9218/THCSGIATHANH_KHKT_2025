"""Frozen inputs + scripted observations only. There is deliberately no live switch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket

from .contracts import ROOT, fingerprint
from .reporting import summarize, write_report
from .validator import evaluate


@contextmanager
def network_denied():
    """Defense in depth; no provider, process launcher or DB client is implemented."""
    def deny(*args, **kwargs):
        raise RuntimeError("offline_network_denied")
    targets = [(socket, "create_connection"), (socket, "getaddrinfo"),
               (socket.socket, "connect"), (socket.socket, "connect_ex"),
               (socket.socket, "sendto"), (socket.socket, "sendmsg")]
    originals = [(obj, name, getattr(obj, name)) for obj, name in targets if hasattr(obj, name)]
    try:
        for obj, name, _ in originals:
            setattr(obj, name, deny)
        yield
    finally:
        for obj, name, value in originals:
            setattr(obj, name, value)


def verify_manifest(path: Path, root: Path) -> str:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("source_hash_mode") != "lf_utf8":
        raise ValueError("unsupported_manifest_hash_mode")
    for relative, expected in manifest["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("manifest_path_escape")
        if not target.is_file() or source_hash(target) != expected:
            raise ValueError(f"source_manifest_drift:{relative}")
    if not manifest["files"]:
        raise ValueError("empty_manifest")
    return hashlib.sha256(raw).hexdigest()


def source_hash(path: Path) -> str:
    """Canonicalize artifact file line endings, NEVER decoded transcript text.

    Git's Windows CRLF conversion must not invalidate an unchanged checkout.
    Message quote/source identities still hash exact unnormalized UTF-8 text.
    """
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def run_suite(suite: dict, manifest_path: Path | None = None) -> dict:
    repo = ROOT.parent
    manifest = manifest_path or ROOT / "specs" / "source-manifest-v0.2.json"
    manifest_hash = verify_manifest(manifest, repo)
    if (suite.get("suite_version") != "mimi-contrastive-v0.2"
            or suite.get("human_gold_status") != "UNREVIEWED"
            or suite.get("semantic_metrics") != "NOT_EVALUATED" or not suite.get("cases")):
        raise ValueError("invalid_scripted_suite")
    ids = [entry["fixture_id"] for entry in suite["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_fixture_id")
    for entry in suite["cases"]:
        if (entry["fixture_id"] != entry["input"]["case_id"]
                or entry["human_adjudication"]["status"] != "UNREVIEWED"):
            raise ValueError("fixture_identity_or_review_status")
    rows = []
    with network_denied():
        for entry in suite["cases"]:
            # No stateful engine, store or model is shared between cases.
            result = evaluate(entry["input"], entry["scripted_observation"])
            rows.append({"fixture_id": entry["fixture_id"], "nucleus_id": entry["nucleus_id"],
                         "human_adjudication": entry["human_adjudication"],
                         "input_sha256": fingerprint(entry["input"]),
                         "observation_sha256": fingerprint(entry["scripted_observation"]),
                         "result": result,
                         "matches_scripted_expectation": result["verdict"] == entry["expected_validator_verdict"]})
    # Catch source changes during the run; don't publish a mixed-baseline report.
    if verify_manifest(manifest, repo) != manifest_hash:
        raise ValueError("manifest_changed_during_replay")
    return {"report_version": "mimi-offline-replay-v0.2", "manifest_sha256": manifest_hash,
            "suite_sha256": fingerprint(suite), "observation_origin": "scripted_synthetic",
            "summary": summarize(rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--suite", type=Path, default=ROOT / "fixtures" / "mimi-contrastive-suite-v0.2.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and args.output is None:
        parser.error("--output required unless --validate-only")
    with network_denied():
        report = run_suite(json.loads(args.suite.read_text(encoding="utf-8")))
        if not args.validate_only:
            write_report(report, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(report["summary"]["scripted_expectation_mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
