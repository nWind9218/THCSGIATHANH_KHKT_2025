import asyncio
import json
import sys
import subprocess

import pytest

from legacy.eval_tools.datasets import load_dataset, validate_dataset
from legacy.eval_tools.evaluation import apply_reviews, review_template, run_job, write_report
from mimi_v2.simulator import LocalSession, load_local_model_environment, make_engine


def run(awaitable):
    return asyncio.run(awaitable)


def dataset(expect=None, review=None):
    return {"version": 1, "cases": [{"id": "one", "turns": [{"user": "Mình muốn được lắng nghe", "expect": expect or {}, "review": review or []}]}]}


def test_isolated_offline_multiturn_and_no_production_imports():
    async def scenario():
        first, second = LocalSession(), LocalSession()
        await first.send("Mình buồn vì cãi nhau với bạn")
        history, result = await first.send("Vì chuyện đó mình không muốn đến lớp")
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
        assert await second.engine.store.load(second.user_id) == {}
        assert result["handoff_status"] == "none"
        await first.control("reset")
        assert await first.engine.store.load(first.user_id) == {}
    run(scenario())
    # Importing local path never imports connection modules.
    probe = subprocess.run([sys.executable, "-c", "import sys; from mimi_v2.simulator import make_engine; make_engine(); assert not any(n.startswith(('memory.', 'api.', 'db.')) for n in sys.modules)"], capture_output=True)
    assert probe.returncode == 0, probe.stderr.decode()


def test_faults_and_handoff_controls():
    async def scenario():
        session = LocalSession()
        _, result = await session.send("Mình muốn tự làm hại bản thân", "notify")
        assert result["handoff_status"] == "pending"
        assert "notification_unavailable" in result["diagnostics"]
        await session.control("acknowledge")
        _, result = await session.send("Có ai ở đây không?")
        assert result.get("response_text", result.get("response")) == ""
        await session.control("resume")
        _, result = await session.send("Xin chào", "store")
        assert "session_load_failed" in result["diagnostics"]
    run(scenario())


def test_dataset_validation():
    valid = dataset()
    valid["cases"].append(valid["cases"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset(valid)
    with pytest.raises(ValueError, match="Unknown expectation"):
        validate_dataset(dataset({"made_up": True}))
    assert load_dataset("legacy/eval_datasets/v2_smoke.json")["version"] == 1
    malformed = dataset()
    malformed["version"] = True
    with pytest.raises(ValueError):
        validate_dataset(malformed)
    malformed = dataset()
    malformed["cases"][0]["id"] = 123
    with pytest.raises(ValueError):
        validate_dataset(malformed)


def test_eval_error_and_human_review_accounting(tmp_path):
    job = run(run_job(dataset({"max_questions": 0}, ["Tone"]), [{"engine": "v2"}], lambda c, case: make_engine(c)))
    assert job["summary"]["pending_review"] == 1
    ratings = review_template(job)
    ratings["ratings"][0].update(verdict="pass", reviewer="Teacher", comment="Reviewed")
    apply_reviews(job, ratings)
    assert job["summary"]["pass"] == 1
    target = write_report(job, tmp_path)
    assert target.exists() and target.with_suffix(".md").exists()
    assert json.loads(target.read_text(encoding="utf-8"))["dataset_hash"]
    class Failing:
        async def respond(self, user_id, text):
            raise TimeoutError()
    failed = run(run_job(dataset(), [{"engine": "v2"}], lambda c, case: Failing()))
    rates = review_template(failed)
    rates["ratings"][0].update(verdict="pass", reviewer="Teacher")
    apply_reviews(failed, rates)
    assert failed["summary"]["error"] == 1 and failed["summary"]["pass"] == 0


def test_key_environment_whitelist(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=test-key\nOPENAI_CHAT_MODEL=test-model\nFACEBOOK_PAGE_ACCESS_TOKEN=do-not-load\nREDIS_URL=do-not-load\n", encoding="utf-8")
    for name in ("OPENAI_API_KEY", "MIMI_V2_BASE_URL", "MIMI_V2_API_KEY", "MIMI_V2_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env = load_local_model_environment(path)
    assert not {k for k in env if not k.startswith("MIMI_V2_")} - {"OPENAI_API_KEY", "OPENAI_CHAT_MODEL"}
    assert env["OPENAI_API_KEY"] == "test-key"
    with pytest.raises(ValueError):
        make_engine({"facebook_token": "anything"})


def test_factory_errors_count_all_turns():
    def fail(config, case):
        raise RuntimeError()
    job = run(run_job(dataset(), [{"engine": "v2"}], fail))
    assert job["summary"]["error"] == 1
    assert job["summary"]["pass"] == 0


def test_actual_baseline_graph_is_isolated_and_observably_legacy():
    async def scenario():
        baseline = make_engine({"engine": "baseline"})
        result = await baseline.respond("baseline-test", "Mình buồn vì cãi nhau với bạn")
        assert result["response_text"]
        assert len(result["model_calls"]) == 5
        assert result["model_calls"][0]["purpose"] == "legacy_safety"
        # Preserve legacy reducer defect for honest comparison, not a fake v2 baseline.
        user_messages = [m for m in result["messages"] if m["role"] == "user"]
        assert len(user_messages) > 1
        assert baseline.provenance()["engine"] == "legacy_sandbox"
    run(scenario())


def test_local_session_reset_and_concurrent_operations():
    async def scenario():
        session = LocalSession()
        await asyncio.gather(session.send("Mình muốn được lắng nghe"), session.send("Vì chuyện đó"))
        saved = await session.engine.store.load(session.user_id)
        assert len(saved["messages"]) == 4
        old_engine = session.engine
        old_engine._recovery[session.user_id] = {"messages": [{"role": "user", "content": "stale"}]}
        await session.control("reset")
        assert session.engine is not old_engine
        assert not session.engine._recovery
        history, _ = await session.send("Xin chào")
        assert len(history) == 2
        assert all(m["content"] != "stale" for m in history)
        with pytest.raises(ValueError):
            await session.control("load")
    run(scenario())


def test_live_model_fault_makes_zero_provider_calls():
    async def scenario():
        session = LocalSession()
        class ForbiddenModel:
            async def complete(self, *args, **kwargs):
                raise AssertionError("provider must not be called")
        model = ForbiddenModel()
        session.engine.model = model
        _, result = await session.send("Xin chào", "model")
        assert "safety_unavailable_or_invalid" in result["diagnostics"]
        assert session.engine.model is model
    run(scenario())


def test_optional_gradio_has_no_remote_fonts_or_analytics():
    pytest.importorskip("gradio")
    from mimi_v2.simulator import build_app
    app = build_app()
    assert app.analytics_enabled is False
    assert not app.theme.to_dict()["theme"]["_stylesheets"]


def test_diagnostics_failure_cannot_be_human_approved():
    class Engine:
        async def respond(self, user_id, text):
            return {"response_text": "fallback", "diagnostics": ["reply_unavailable_or_invalid"]}
    job = run(run_job(dataset({"contains": ["fallback"]}), [{"engine": "v2"}], lambda c, case: Engine()))
    assert job["summary"]["error"] == 1
    ratings = review_template(job)
    ratings["ratings"][0].update(verdict="pass", reviewer="Teacher")
    apply_reviews(job, ratings)
    assert job["summary"]["error"] == 1 and job["summary"]["pass"] == 0


def test_grading_ignores_first_turn_greeting():
    from legacy.eval_tools.evaluation import check_turn, reply_text
    from mimi_v2.prompts import GREETING_MESSAGE
    result = {"response_parts": [GREETING_MESSAGE, "Mình nghe bạn nè."]}
    result["response_text"] = "\n\n".join(result["response_parts"])
    assert reply_text(result) == "Mình nghe bạn nè."
    checks = check_turn(result, {"max_chars": 40, "not_contains": [GREETING_MESSAGE[:10]], "contains": ["nghe"]})
    assert checks and all(c["passed"] for c in checks), checks
