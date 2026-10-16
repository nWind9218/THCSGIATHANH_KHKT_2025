"""Per-request timing trace; offline, no provider or storage."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from mimi_v2.core import CounselingEngine
from mimi_v2.store import InMemoryStore
from mimi_v2.tracing import Trace, model_attrs, summarize, waterfall


def assessment(**changes):
    result = dict(known={}, missing_fields=[], next_question_focus="", need="comfort", intent="support",
                  topic="school", core_issue="school pressure", safety_hint="safe", user_role="unknown")
    result.update(changes)
    return result


class SlowModel:
    DELAYS = {"safety": 0.03, "assessment": 0.02, "reply": 0.05}

    async def complete(self, messages, purpose):
        await asyncio.sleep(self.DELAYS[purpose])
        text = {"safety": json.dumps({"risk": "safe", "reason": "x"}),
                "assessment": json.dumps(assessment(need="listen")),
                "reply": "Mimi đang lắng nghe."}[purpose]
        usage = {"input_tokens": 100, "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 25}}
        return SimpleNamespace(text=text, usage=usage, metadata={})


class SlowStore(InMemoryStore):
    async def load(self, user_id):
        await asyncio.sleep(0.01)
        return await super().load(user_id)

    async def save(self, user_id, session):
        await asyncio.sleep(0.01)
        await super().save(user_id, session)


def test_trace_records_offsets_and_attrs_without_none_values():
    trace = Trace()
    with trace.span("a", n=1, skip=None) as extra:
        extra["ok"] = True
    with pytest.raises(ValueError):
        with trace.span("b"):
            raise ValueError("boom")
    done = trace.finish()
    assert [s["name"] for s in done["spans"]] == ["a", "b"]
    assert done["spans"][0]["attrs"] == {"n": 1, "ok": True}
    assert done["spans"][1]["attrs"] == {"error": "ValueError"}
    assert done["total"] >= sum(s["duration"] for s in done["spans"])


def test_model_attrs_and_summary_and_waterfall():
    assert model_attrs({"input_tokens": 5, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 3}}) == \
        {"input_tokens": 5, "output_tokens": 7, "reasoning_tokens": 3}
    assert model_attrs({"prompt_tokens": 2, "completion_tokens": 4}) == {"input_tokens": 2, "output_tokens": 4, "reasoning_tokens": None}
    trace = {"total": 10.0, "unaccounted": 0.5, "spans": [
        {"name": "model.safety", "start": 0.0, "duration": 6.0, "attrs": {"output_tokens": 70}},
        {"name": "model.safety", "start": 6.0, "duration": 1.0, "attrs": {}},
        {"name": "store.save", "start": 7.0, "duration": 2.5, "attrs": {}}]}
    assert summarize(trace) == {"model.safety": 7.0, "store.save": 2.5}
    table = waterfall(trace)
    assert "**model.safety** ⚠" in table and "60%" in table and "output_tokens=70" in table and "10.00s" in table
    assert "chưa có dữ liệu" in waterfall({}).lower()


@pytest.mark.asyncio
async def test_engine_turn_has_waterfall_with_every_step_and_token_detail():
    engine = CounselingEngine(SlowModel(), SlowStore())
    result = await engine.respond("u", "Hôm nay mình buồn")
    trace = result["trace"]
    names = [s["name"] for s in trace["spans"]]
    assert names[:2] == ["lock_wait", "store.load"] and names[-2:] == ["model.reply", "store.save"]
    assert sorted(names[2:4]) == ["model.assessment", "model.safety"]  # concurrent, so start order is arbitrary
    by = {s["name"]: s for s in trace["spans"]}
    assert by["model.reply"]["duration"] >= 0.045 and by["model.safety"]["duration"] >= 0.025
    assert by["model.safety"]["attrs"] == {"input_tokens": 100, "output_tokens": 40, "reasoning_tokens": 25}
    starts = [s["start"] for s in trace["spans"]]
    assert starts == sorted(starts)
    assert trace["total"] >= 0.1 and trace["unaccounted"] >= 0
    assert "Tổng" in waterfall(trace)


@pytest.mark.asyncio
async def test_trace_marks_handoff_notify_and_never_contains_message_text():
    class Notifying(SlowStore):
        async def notify(self, user_id, text):
            await asyncio.sleep(0.01)
            return True

    class Urgent(SlowModel):
        async def complete(self, messages, purpose):
            reply = await super().complete(messages, purpose)
            if purpose == "safety":
                reply.text = json.dumps({"risk": "urgent", "reason": "x"})
            return reply

    secret = "Bí mật riêng tư"
    result = await CounselingEngine(Urgent(), Notifying()).respond("u", secret)
    names = [s["name"] for s in result["trace"]["spans"]]
    assert "store.notify" in names and "model.reply" not in names
    assert next(s for s in result["trace"]["spans"] if s["name"] == "store.notify")["attrs"] == {"sent": True}
    assert secret not in json.dumps(result["trace"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_search_internal_timings_are_traced_and_kept_out_of_the_prompt():
    seen = {}

    class Searching(SlowStore):
        async def search(self, user_id, query):
            return {"psychology": "tài liệu", "errors": [], "_timings": {"embed_s": 0.2, "sql_s": 0.1}}

    class Capturing(SlowModel):
        async def complete(self, messages, purpose):
            if purpose == "reply":
                seen["prompt"] = messages[0]["content"]
            return await super().complete(messages, purpose)

    class Advice(Capturing):
        async def complete(self, messages, purpose):
            reply = await super().complete(messages, purpose)
            if purpose == "assessment":
                reply.text = json.dumps(assessment(need="advice"))
            return reply

    result = await CounselingEngine(Advice(), Searching()).respond("u", "Em nên học thế nào?")
    search = next(s for s in result["trace"]["spans"] if s["name"] == "store.search")
    assert search["attrs"] == {"embed_s": 0.2, "sql_s": 0.1}
    assert "_timings" not in seen["prompt"] and "embed_s" not in seen["prompt"]


@pytest.mark.asyncio
async def test_failed_model_call_still_records_span_with_error_code():
    from mimi_v2.models import ModelError

    class Broken(SlowModel):
        async def complete(self, messages, purpose):
            if purpose == "safety":
                raise ModelError("timeout")
            return await super().complete(messages, purpose)

    result = await CounselingEngine(Broken(), SlowStore()).respond("u", "xin chào")
    span = next(s for s in result["trace"]["spans"] if s["name"] == "model.safety")
    assert span["attrs"]["error"] == "timeout"
