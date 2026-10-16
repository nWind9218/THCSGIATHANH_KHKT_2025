"""Offline behavioral regressions; no provider credentials or production dependencies."""
import json
from types import SimpleNamespace

import pytest

from mimi_v2.contracts import ContractError, normalize_assessment
from mimi_v2.core import CounselingEngine
from mimi_v2.store import InMemoryStore


class Model:
    def __init__(self, *, intent="closing", need="listen", risk="safe", hint="safe", reply="Mimi nghe bạn đang mệt vì mạng yếu, vẫn muốn được lắng nghe.", role="student"):
        self.intent, self.need, self.risk, self.hint = intent, need, risk, hint
        self.reply, self.role, self.calls = reply, role, []

    async def complete(self, messages, purpose):
        self.calls.append((purpose, messages))
        text = json.dumps({"risk": self.risk, "reason": "synthetic fixture"}) if purpose == "safety" else json.dumps({
            "known": [], "missing_fields": [], "next_question_focus": "", "need": self.need,
            "intent": self.intent, "topic": "network", "core_issue": "limited network",
            "safety_hint": self.hint, "user_role": self.role,
        }) if purpose == "assessment" else self.reply
        return SimpleNamespace(text=text, usage={}, metadata={})


@pytest.mark.asyncio
async def test_confirmed_listen_closing_conflict_does_not_preempt_actual_reply():
    model = Model()
    result = await CounselingEngine(model, InMemoryStore()).respond(
        "synthetic", "Mình chỉ muốn được lắng nghe lúc này, chưa muốn lời khuyên hay câu hỏi tiếp.")
    assert result["response_text"] == model.reply
    assert "?" not in result["response_text"]
    assert [purpose for purpose, _ in model.calls] == ["safety", "assessment", "reply"]
    prompt = model.calls[-1][1][0]["content"]
    assert "từ chối lời khuyên" in prompt
    assert "không đồng nghĩa kết thúc" in prompt
    assert "CHẾ ĐỘ ĐÃ CHỌN: listen" in prompt


@pytest.mark.asyncio
async def test_genuine_closing_listen_is_given_stop_rule_and_complete_user_context():
    model = Model(reply="Cảm ơn bạn. Bạn nghỉ ngơi nhé.")
    user = "Cảm ơn, mình muốn dừng ở đây và nghỉ một chút."
    result = await CounselingEngine(model, InMemoryStore()).respond("synthetic", user)
    assert result["response_text"] == model.reply
    assert "?" not in result["response_text"]
    assert result["assessment"]["intent"] == "closing"
    assert model.calls[-1][0] == "reply"
    assert model.calls[-1][1][-1]["content"] == user
    assert "dừng cuộc trò chuyện" in model.calls[-1][1][0]["content"]


@pytest.mark.asyncio
async def test_unambiguous_nonlisten_closing_keeps_bounded_goodbye():
    model = Model(need="comfort")
    result = await CounselingEngine(model, InMemoryStore()).respond("synthetic", "Mình dừng ở đây, tạm biệt.")
    assert "Cảm ơn bạn" in result["response_text"]
    assert "?" not in result["response_text"]
    assert [purpose for purpose, _ in model.calls] == ["safety", "assessment"]


def test_identical_fact_duplicates_normalize_without_information_loss():
    source = {"known": [{"key": "grade", "value": "8"}, {"key": "grade", "value": "8"}]}
    assert normalize_assessment(source)["known"] == {"grade": "8"}
    assert len(source["known"]) == 2


def test_conflicting_fact_duplicates_never_pick_arbitrary_winner():
    with pytest.raises(ContractError, match="duplicate_fact_key"):
        normalize_assessment({"known": [{"key": "grade", "value": "8"}, {"key": "grade", "value": "9"}]})


@pytest.mark.asyncio
async def test_actual_assessment_and_reply_payloads_ground_user_facts_and_corrected_aliases():
    model = Model(intent="support")
    await CounselingEngine(model, InMemoryStore()).respond("synthetic", "Bạn mới đề nghị ba câu, mình chưa đồng ý; em họ, không phải anh họ.")
    assessment_prompt = next(messages[0]["content"] for purpose, messages in model.calls if purpose == "assessment")
    reply_prompt = next(messages[0]["content"] for purpose, messages in model.calls if purpose == "reply")
    assert "không phải dữ kiện người dùng" in assessment_prompt
    assert "khóa cũ" in assessment_prompt
    assert "không chứng minh" in reply_prompt
    assert "bạn thân" in reply_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("risk,hint", [("urgent", "safe"), ("uncertain", "safe"), ("safe", "urgent")])
async def test_listen_closing_conflict_never_bypasses_safety(risk, hint):
    store = InMemoryStore()
    model = Model(risk=risk, hint=hint)
    engine = CounselingEngine(model, store)
    result = await engine.respond("synthetic", "Mình chỉ muốn được lắng nghe")
    assert result["handoff_status"] == "pending"
    assert "Mình đang nhờ thầy cô" in result["response_text"]
    assert all(purpose != "reply" for purpose, _ in model.calls)
    await store.acknowledge("synthetic")
    before = len(model.calls)
    assert (await engine.respond("synthetic", "Mình vẫn ở đây"))["response_text"] == ""
    assert len(model.calls) == before


@pytest.mark.asyncio
async def test_parent_listen_conflict_does_not_grant_other_session_access():
    store = InMemoryStore()
    await store.save("child", {"messages": [{"role": "user", "content": "synthetic-private"}]})
    model = Model(role="parent")
    result = await CounselingEngine(model, store).respond("parent", "Tôi là phụ huynh, chỉ muốn được lắng nghe.")
    assert result["assessment"]["user_role"] == "parent"
    assert "synthetic-private" not in str(model.calls)
    assert (await store.load("child"))["messages"][0]["content"] == "synthetic-private"
