import json

from mimi_eval.contracts import ROOT


def test_observer_contract_and_trust_boundaries():
    text=(ROOT/"prompts"/"mimi-judge-observer-prompt-v0.2.md").read_text(encoding="utf-8")
    for token in ("final_verdict", "safety_exception_approved", "handoff_acknowledged", "effective_model", "effective_effort", "DATA", "complete_prefix", "unknown", "[start,end)", "one-sentence", "No chain-of-thought", "accepted_silence", "deterministic validator"):
        assert token in text


def test_decision_register_dispositions_and_simulation():
    r=json.loads((ROOT/"specs"/"decision-register-v0.2.json").read_text(encoding="utf-8"))
    assert len(r["corrections"])==10
    assert all(c["disposition"] in {"approved","deferred","rejected"} for c in r["corrections"])
    assert r["production_approved"] is False
    cfg=r["simulated_judge_config"]
    assert cfg["model"]=="gpt-5.6-terra" and cfg["api"]=="responses"
    assert cfg["reasoning_mode"]=="standard" and cfg["default_effort"]=="medium" and cfg["comparison_effort"]=="high"
    assert cfg["provider_calls"]==0 and cfg["effective_model"] is None and cfg["live_enabled"] is False


def test_length_biased_sample_was_corrected():
    text=(ROOT/"specs"/"Design_Add-ins.md").read_text(encoding="utf-8")
    assert "chưa có cơ sở để cộng điểm từ một câu ngắn" not in text
    assert '\"code\":\"Q3\",\"applicable\":true,\"score\":3' in text
