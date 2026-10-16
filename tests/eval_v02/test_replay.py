import builtins
import json
from pathlib import Path
import socket
import sys

import pytest

from mimi_eval.contracts import ROOT
from mimi_eval.replay import main, network_denied, run_suite, verify_manifest
from mimi_eval.reporting import write_report


def test_replay_deterministic_independent_and_four_states(suite):
    first=run_suite(suite)
    assert run_suite(suite)==first
    assert all(first["summary"]["verdict_counts"].values())
    assert first["summary"]["scripted_expectation_mismatches"]==0
    assert first["summary"]["semantic_agreement"]=="NOT_EVALUATED"
    assert first["summary"]["provider_calls"]==0
    assert first["summary"]["technical_failures"]==1
    assert first["summary"]["invalid_observations"]==2


def test_manifest_drift_stops_before_eval(tmp_path,monkeypatch,suite):
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"source_hash_mode":"lf_utf8","files":{"mimi_eval/__init__.py":"0"*64}}))
    monkeypatch.setattr("mimi_eval.replay.evaluate",lambda *_:pytest.fail("must not dispatch"))
    with pytest.raises(ValueError,match="source_manifest_drift"):run_suite(suite,manifest)


def test_manifest_escape_and_empty(tmp_path):
    m=tmp_path/"m.json"
    for files in ({"../escape":"0"*64},{}):
        m.write_text(json.dumps({"source_hash_mode":"lf_utf8","files":files}))
        with pytest.raises(ValueError):verify_manifest(m,ROOT.parent)


def test_report_preserves_verdicts_and_does_not_overwrite(tmp_path,suite):
    report=run_suite(suite);write_report(report,tmp_path)
    stored=json.loads((tmp_path/"offline-report.json").read_text(encoding="utf-8"))
    assert stored==report
    assert "NOT_EVALUATED" in (tmp_path/"offline-report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):write_report(report,tmp_path)


@pytest.mark.parametrize("operation", [lambda:socket.create_connection(("127.0.0.1",9)),lambda:socket.getaddrinfo("example.com",443),lambda:socket.socket().connect(("127.0.0.1",9))])
def test_network_denied_even_loopback(operation):
    with pytest.raises(RuntimeError,match="offline_network_denied"):operation()


def test_credential_reads_and_production_imports_are_blocked():
    with pytest.raises(AssertionError,match="credential_file_read_denied"):Path(".env.production").read_text()
    with pytest.raises(AssertionError,match="production_or_provider_import_denied"):builtins.__import__("mimi_v2.production")
    assert not any(name in sys.modules for name in ("openai","supabase","redis","dotenv","mimi_v2.production"))


def test_cli_validate_only(capsys):
    assert main(["--offline","--validate-only"])==0
    assert '"provider_calls": 0' in capsys.readouterr().out


def test_cli_rejects_live_switch():
    with pytest.raises(SystemExit):main(["--offline","--live"])


def test_manifest_drift_during_replay_stops_publication(suite,monkeypatch):
    hashes=iter(["before","after"])
    monkeypatch.setattr("mimi_eval.replay.verify_manifest",lambda *_:next(hashes))
    with pytest.raises(ValueError,match="manifest_changed_during_replay"):run_suite(suite)


@pytest.mark.parametrize("mutation", [
    lambda s:s.update(human_gold_status="APPROVED"),
    lambda s:s["cases"].append(s["cases"][0]),
    lambda s:s["cases"][0].update(fixture_id="wrong"),
    lambda s:s["cases"][0]["human_adjudication"].update(status="APPROVED"),
])
def test_scripted_suite_cannot_claim_adjudicated_gold(suite,mutation):
    mutation(suite)
    with pytest.raises(ValueError):run_suite(suite)


def test_git_crlf_artifact_conversion_preserves_manifest_identity(tmp_path):
    from mimi_eval.replay import source_hash
    source=tmp_path/"module.py"
    source.write_bytes(b"x = 1\n")
    m=tmp_path/"manifest.json"
    m.write_text(json.dumps({"source_hash_mode":"lf_utf8","files":{"module.py":source_hash(source)}}))
    source.write_bytes(b"x = 1\r\n")
    verify_manifest(m,tmp_path)
    source.write_bytes(b"x = 2\r\n")
    with pytest.raises(ValueError,match="source_manifest_drift"):verify_manifest(m,tmp_path)
