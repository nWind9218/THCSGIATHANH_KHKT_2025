import asyncio
import json
from types import SimpleNamespace

import pytest

from mimi_v2.core import CounselingEngine
from mimi_v2.store import InMemoryStore


def assessment(**changes):
    result = dict(known={}, missing_fields=[], next_question_focus="", need="comfort",
                  intent="support", topic="school", core_issue="school pressure",
                  safety_hint="safe", user_role="unknown")
    result.update(changes)
    return result


class FakeModel:
    def __init__(self, assessments=None, risk="safe", fail=None):
        self.assessments = list(assessments or [])
        self.risk, self.fail, self.calls = risk, fail, []

    async def complete(self, messages, purpose):
        self.calls.append((purpose, messages))
        await asyncio.sleep(0)
        if self.fail == purpose:
            raise RuntimeError("fixture failure")
        if purpose == "safety":
            text = json.dumps({"risk": self.risk, "reason": "fixture"})
        elif purpose == "assessment":
            text = json.dumps(self.assessments.pop(0) if self.assessments else assessment())
        else:
            text = "Mimi hiểu điều bạn vừa chia sẻ."
        return SimpleNamespace(text=text, usage={"input_tokens": 2, "output_tokens": 1}, metadata={"model": "fixture"})


@pytest.mark.asyncio
async def test_history_is_canonical_and_reaches_reply_with_corrections():
    store = InMemoryStore()
    await store.save("a", [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}])
    model = FakeModel([assessment(known={"grade": "8"}), assessment(known={"grade": "9"}, missing_fields=["grade"], next_question_focus="grade")])
    engine = CounselingEngine(model, store)
    await engine.respond("a", "Em học lớp 8")
    result = await engine.respond("a", "Em sửa lại: lớp 9")
    assert [m["content"] for m in result["messages"]] == ["old", "reply", "Em học lớp 8", "Mimi hiểu điều bạn vừa chia sẻ.", "Em sửa lại: lớp 9", "Mimi hiểu điều bạn vừa chia sẻ."]
    assert result["assessment"]["known"] == {"grade": "9"}
    assert result["assessment"]["missing_fields"] == []
    assert result["assessment"]["next_question_focus"] == ""
    reply_input = [m for p, m in model.calls if p == "reply"][-1]
    assert reply_input[1]["content"] == "old"
    assert '"grade": "9"' in reply_input[0]["content"]


@pytest.mark.asyncio
async def test_parent_listening_and_closing_use_contract():
    store = InMemoryStore()
    model = FakeModel([assessment(need="listen", user_role="parent"), assessment(intent="closing")])
    engine = CounselingEngine(model, store)
    first = await engine.respond("parent", "Tôi là mẹ của bé, tôi chỉ muốn được lắng nghe")
    assert first["assessment"]["user_role"] == "parent"
    assert first["sources"] == {}
    prompt = [m for p, m in model.calls if p == "reply"][-1][0]["content"]
    assert '"need": "listen"' in prompt
    assert "không cấp quyền" in prompt
    assert "CHẾ ĐỘ ĐÃ CHỌN: listen" in prompt
    assert "câu hỏi tiếp nối" in prompt
    assert "Không cần lấp đầy missing_fields" in prompt
    second = await engine.respond("parent", "Cảm ơn, vậy đủ rồi")
    assert second["assessment"]["user_role"] == "parent"
    assert "?" not in second["response"]
    assert len([p for p, _ in model.calls if p == "reply"]) == 1


@pytest.mark.asyncio
async def test_listen_instruction_overrides_remaining_missing_info_without_stripping_output():
    model = FakeModel([assessment(need="listen", missing_fields=["time"], next_question_focus="time")])
    result = await CounselingEngine(model, InMemoryStore()).respond("a", "Chỉ muốn bạn nghe thôi")
    prompt = next(messages[0]["content"] for purpose, messages in model.calls if purpose == "reply")
    assert "quy tắc listen ưu tiên" in prompt
    assert result["assessment"]["missing_fields"] == ["time"]
    assert result["response"] == "Mimi hiểu điều bạn vừa chia sẻ."


@pytest.mark.asyncio
@pytest.mark.parametrize("risk, hint", [("urgent", "safe"), ("safe", "urgent"), ("bogus", "safe")])
async def test_all_risk_routes_pending_and_accepted_silence(risk, hint):
    store = InMemoryStore()
    model = FakeModel([assessment(safety_hint=hint)], risk=risk)
    engine = CounselingEngine(model, store)
    result = await engine.respond("a", "nguy cơ")
    assert result["handoff_status"] == "pending"
    assert "chưa có xác nhận" in result["response"]
    assert result["sources"] == {}
    assert len(store.alerts) == 1
    await engine.respond("a", "tiếp nối")
    assert len(store.alerts) == 1
    await store.acknowledge("a")
    prior_calls = len(model.calls)
    accepted = await engine.respond("a", "thầy cô đang nghe")
    assert accepted["response"] == ""
    assert accepted["messages"][-1]["content"] == "thầy cô đang nghe"
    assert len(model.calls) == prior_calls
    await store.resume("a")
    assert (await store.load("a"))["handoff_status"] == "none"


@pytest.mark.asyncio
async def test_safety_before_retrieval_and_notification_failure():
    class BrokenStore(InMemoryStore):
        async def search(self, *args):
            raise AssertionError("must not retrieve before safety")
        async def notify(self, *args):
            raise RuntimeError("notify down")
    engine = CounselingEngine(FakeModel(fail="safety"), BrokenStore())
    result = await engine.respond("a", "message")
    assert "notification_unavailable" in result["diagnostics"]
    assert "chưa gửi được" in result["response"]
    assert result["handoff_status"] == "pending"


@pytest.mark.asyncio
async def test_load_failure_never_overwrites_unknown_takeover():
    class BrokenStore(InMemoryStore):
        async def load(self, *args):
            raise RuntimeError("down")
        async def save(self, *args):
            raise AssertionError("unsafe overwrite")
    model = FakeModel()
    result = await CounselingEngine(model, BrokenStore()).respond("a", "hello")
    assert result["handoff_status"] == "unknown"
    assert model.calls == []


@pytest.mark.asyncio
async def test_save_failure_recovery_retains_turn_and_reports_limit():
    class BrokenStore(InMemoryStore):
        async def save(self, *args):
            raise RuntimeError("down")
    engine = CounselingEngine(FakeModel(), BrokenStore())
    first = await engine.respond("a", "first")
    assert "session_save_failed_local_recovery_only" in first["diagnostics"]
    second = await engine.respond("a", "second")
    assert [m["content"] for m in second["messages"] if m["role"] == "user"] == ["first", "second"]


@pytest.mark.asyncio
async def test_concurrency_serializes_one_user_and_isolates_another():
    engine = CounselingEngine(FakeModel(), InMemoryStore())
    await asyncio.gather(engine.respond("a", "a1"), engine.respond("a", "a2"), engine.respond("b", "b1"))
    a, b = await engine.store.load("a"), await engine.store.load("b")
    assert [m["content"] for m in a["messages"] if m["role"] == "user"] == ["a1", "a2"]
    assert [m["content"] for m in b["messages"] if m["role"] == "user"] == ["b1"]


@pytest.mark.asyncio
async def test_retrieval_and_reply_failures_are_visible_and_keep_history():
    class BrokenStore(InMemoryStore):
        async def search(self, *args):
            raise RuntimeError("down")
    result = await CounselingEngine(FakeModel(fail="reply"), BrokenStore()).respond("a", "hello")
    assert set(result["diagnostics"]) == {"retrieval_unavailable", "reply_unavailable_or_invalid"}
    assert len(result["messages"]) == 2
    assert result["usage"]["input_tokens"] == 4


@pytest.mark.asyncio
async def test_invalid_assessment_is_review_not_an_assumed_safe_reply():
    model = FakeModel([{"verdict": "sufficient", "scratchpad": []}])
    result = await CounselingEngine(model, InMemoryStore()).respond("a", "hello")
    assert result["handoff_status"] == "pending"
    assert "assessment_unavailable_or_invalid" in result["diagnostics"]
    assert [p for p, _ in model.calls] == ["safety", "assessment"]


@pytest.mark.asyncio
async def test_admin_acknowledgement_cannot_be_overwritten_by_inflight_turn():
    store = InMemoryStore()
    engine = CounselingEngine(FakeModel(risk="urgent"), store)
    await engine.respond("a", "danger")
    await asyncio.gather(engine.respond("a", "follow up"), store.acknowledge("a"))
    assert (await store.load("a"))["handoff_status"] == "accepted"
    assert (await engine.respond("a", "next"))["response_text"] == ""


@pytest.mark.asyncio
async def test_recovery_does_not_hide_external_acceptance():
    class FailOnce(InMemoryStore):
        failed = False
        async def save(self, user_id, session):
            if not self.failed:
                self.failed = True
                raise RuntimeError("down")
            await super().save(user_id, session)
    store = FailOnce()
    engine = CounselingEngine(FakeModel(risk="urgent"), store)
    await engine.respond("a", "danger")
    store.sessions["a"] = {"handoff_status": "accepted", "messages": []}
    result = await engine.respond("a", "still here")
    assert result["handoff_status"] == "accepted"
    assert result["response_text"] == ""


@pytest.mark.asyncio
async def test_history_window_and_store_deep_copy():
    store = InMemoryStore()
    old = [{"role": "user", "content": str(i)} for i in range(25)]
    await store.save("a", {"messages": old})
    old.clear()
    result = await CounselingEngine(FakeModel(), store).respond("a", "latest")
    assert len(result["messages"]) == 20
    assert result["messages"][-2]["content"] == "latest"
    result["messages"].clear()
    assert len((await store.load("a"))["messages"]) == 20
