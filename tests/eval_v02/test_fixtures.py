from collections import Counter,defaultdict
import json

from mimi_eval.contracts import ROOT, fingerprint
from mimi_eval.validator import evaluate


def test_all_scripted_outcomes_and_human_status(suite):
    assert len(suite["cases"])==79
    ids=[r["fixture_id"] for r in suite["cases"]]
    assert len(ids)==len(set(ids))
    for row in suite["cases"]:
        assert row["human_adjudication"]["status"]=="UNREVIEWED"
        assert row["provenance"]=="synthetic"
        assert evaluate(row["input"],row["scripted_observation"])["verdict"]==row["expected_validator_verdict"],row["fixture_id"]


def test_nuclei_and_family_split_no_leakage(suite):
    counts=Counter(r["nucleus_id"] for r in suite["cases"] if r["nucleus_id"]!="CONTROL")
    assert counts=={f"T{i:02}":3 for i in range(1,26)}
    splits=defaultdict(set)
    for r in suite["cases"]:splits[r["family_id"]].add(r["split"])
    assert all(len(v)==1 for v in splits.values())
    assert {r["family_id"] for r in suite["cases"] if r["nucleus_id"] in ("T14","T20")}=={"unsafe-parent-source"}


def test_gold_is_empty_unreviewed_and_bound_to_inputs(suite):
    gold=json.loads((ROOT/"fixtures"/"human-gold-template.v0.2.json").read_text(encoding="utf-8"))
    assert gold["status"]=="UNREVIEWED"
    annotations={r["fixture_id"]:r for r in gold["annotations"]}
    for e in suite["cases"]:
        row=annotations[e["fixture_id"]]
        assert row["input_sha256"]==fingerprint(e["input"])
        assert row["actions"] is row["labels"] is row["dimensions"] is None
        assert row["reviewers"]==[]
