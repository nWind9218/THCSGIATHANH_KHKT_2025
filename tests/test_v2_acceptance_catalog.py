from collections import Counter
from pathlib import Path

from eval_datasets.build_acceptance import build_dataset, coverage, VARIANTS
from mimi_v2.datasets import load_dataset, digest


def test_materialized_thousand_cases_are_reproducible_and_distinct():
    data = build_dataset()
    assert digest(load_dataset("eval_datasets/v2_acceptance_1000.json")) == digest(data)
    assert len(data["cases"]) == 1000
    assert len({digest(c["turns"]) for c in data["cases"]}) == 1000
    assert all(n == 50 for n in Counter(c["category"] for c in data["cases"]).values())
    assert len({c["category"] for c in data["cases"]}) == 20
    assert {c["variant"] for c in data["cases"]} == set(VARIANTS)
    assert all(c["synthetic"] and len(c["turns"]) >= 4 for c in data["cases"])
    assert all(t["review"] for c in data["cases"] for t in c["turns"])


def test_split_groups_never_leak_base_scenarios():
    data = build_dataset()
    groups = {}
    for c in data["cases"]:
        groups.setdefault(c["scenario_id"], set()).add(c["split"])
    assert len(groups) == 200
    assert all(len(splits) == 1 for splits in groups.values())
    assert coverage(data)["splits"] == {"development": 600, "regression": 200, "holdout": 200}
    assert all(n > 0 for n in coverage(data)["long_turns_by_split"].values())
    for split in ("development", "regression", "holdout"):
        subset = load_dataset(Path("eval_datasets") / f"v2_acceptance_{split}.json")
        assert all(c["split"] == split for c in subset["cases"])


def test_long_short_deep_safety_parent_fault_and_handoff_coverage():
    data = build_dataset()
    report = coverage(data)
    assert report["long_user_turns_150_words"] >= 200
    assert report["short_openings_12_words"] >= 200
    tags = report["tags"]
    for tag in ("parent", "urgent", "uncertain", "listen_only", "multi_intent", "deep_reflection", "model_outage",
                "storage_outage", "notification_outage", "accepted_silence", "accept_then_resume", "retrieval_injection"):
        assert tags[tag] > 0
    assert data["provenance"]["clinical_validation"] is False
    assert data["provenance"]["production_transcripts_used"] is False
