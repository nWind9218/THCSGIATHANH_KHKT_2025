import pytest
import asyncio
from graph.nodes import deep_reasoning_node
from graph.state import CounselingState

@pytest.mark.asyncio
async def test_deep_reasoning_node_escalation():
    # Simulate a crisis user message
    state = CounselingState(
        user_id="test_user",
        messages=[{"role": "user", "content": "Mình muốn biến mất khỏi thế giới này vì quá mệt mỏi"}],
    )
    result = await deep_reasoning_node(state)
    # Should trigger escalation
    assert result.get("human_takeover") is True
    assert "escalation_reason" in result
    assert "biến mất" in result["escalation_reason"] or "crisis" in result["escalation_reason"].lower()

@pytest.mark.asyncio
async def test_deep_reasoning_node_no_escalation():
    # Simulate a normal user message
    state = CounselingState(
        user_id="test_user",
        messages=[{"role": "user", "content": "Mimi ơi, hôm nay mình đi học vui lắm!"}],
    )
    result = await deep_reasoning_node(state)
    # Should NOT trigger escalation
    assert result.get("human_takeover") is not True
    assert "escalation_reason" not in result or not result["escalation_reason"]
