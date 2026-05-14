from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import CounselingState
from graph.tools import get_llm, latest_user_message
from memory import (
    load_history,
    load_takeover_flag,
    load_topic,
    notify_human_admin,
    save_history,
    save_topic,
    search_psychology_kb,
    search_student_knowledge_kb,
    search_user_memory_kb,
    set_takeover_flag,
    update_user_longterm_style,
    upsert_user_memory_chunk,
)
from prompts import (
    get_safety_check_prompt,
    get_intent_routing_prompt,
    get_simple_response_system_prompt,
    get_simple_response_user_prompt,
    get_out_of_scope_response,
    get_clarification_prompt,
    get_deep_reasoning_system_prompt,
    get_deep_reasoning_user_prompt,
    get_info_gap_assessment_prompt,
    get_memory_gate_prompt,
)

logger = logging.getLogger(__name__)


async def load_memory_node(state: CounselingState) -> dict:
    user_id = state.get("user_id") or state.get("conversation", {}).get("user_id")
    incoming_messages = state.get("messages", [])

    if not user_id:
        return {"error": "Missing user_id", "human_takeover": False}

    is_takeover = await load_takeover_flag(user_id)
    if is_takeover:
        return {"human_takeover": True}

    history = await load_history(user_id)
    merged = (history + incoming_messages)[-20:]
    current_topic = await load_topic(user_id)

    query = latest_user_message(merged)
    user_memory = await search_user_memory_kb(user_id=user_id, query=query, top_k=3)

    return {
        "user_id": user_id,
        "messages": merged,
        "human_takeover": False,
        "current_topic": current_topic,
        "user_memory": user_memory,
    }


async def safety_gateway_node(state: CounselingState) -> dict:
    if state.get("human_takeover"):
        return {}

    user_text = latest_user_message(state.get("messages", []))
    if not user_text:
        return {"is_emergency": False}

    llm = get_llm()
    prompt = get_safety_check_prompt(user_text)

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    verdict = (response.content or "").strip().upper()
    return {"is_emergency": verdict.startswith("YES")}


async def handoff_human_node(state: CounselingState) -> dict:
    user_id = state.get("user_id", "")
    user_text = latest_user_message(state.get("messages", []))

    comfort_text = (
        "Mình rất lo cho bạn khi nghe điều này. Bạn không đơn độc đâu nhé. "
        "Nếu bạn cần được giúp đỡ ngay, hãy gọi cho các cô để được hỗ trợ trực tiếp:\n"
        "📞 Cô Thúy: 0962186108\n"
        "📞 Cô Trâm: 0915266338\n"
        "Hoặc gọi Tổng đài Quốc gia 111 (miễn phí, 24/7). "
        "Mình đang nhờ thầy cô hỗ trợ bạn ngay bây giờ."
    )

    summary = f"Emergency signal from {user_id}"
    await notify_human_admin(user_id=user_id, summary=summary, raw_message=user_text)
    await set_takeover_flag(user_id, True)

    return {"response_text": comfort_text, "human_takeover": True}


async def intent_routing_node(state: CounselingState) -> dict:
    user_text = latest_user_message(state.get("messages", []))
    history = state.get("messages", [])[-5:]
    current_topic = state.get("current_topic", "")

    llm = get_llm()
    prompt = get_intent_routing_prompt(user_text, history, current_topic)

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    parsed = json.loads((response.content or "{}").strip("` \n"))

    return {
        "intent_category": parsed.get("intent", "simple"),
        "current_topic": parsed.get("topic", current_topic),
    }


async def out_of_scope_node(state: CounselingState) -> dict:
    return {"response_text": get_out_of_scope_response()}


async def simple_response_node(state: CounselingState) -> dict:
    user_text = latest_user_message(state.get("messages", []))
    memory = state.get("user_memory", "")

    llm = get_llm()
    messages = [
        SystemMessage(content=get_simple_response_system_prompt()),
        HumanMessage(content=get_simple_response_user_prompt(user_text, memory)),
    ]
    response = await llm.ainvoke(messages)
    return {"response_text": response.content}


async def info_gap_assessor_node(state: CounselingState) -> dict:
    history = state.get("messages", [])
    user_text = latest_user_message(history)
    llm = get_llm()

    prompt = get_info_gap_assessment_prompt(user_text, history)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    parsed = json.loads((response.content or "{}").strip("` \n"))

    return {
        "reasoning_scratchpad": parsed.get("scratchpad", []),
        "info_gap_status": parsed.get("verdict", "missing"),
    }


async def clarification_node(state: CounselingState) -> dict:
    scratchpad = state.get("reasoning_scratchpad", [])
    user_text = latest_user_message(state.get("messages", []))

    llm = get_llm()
    prompt = get_clarification_prompt(scratchpad, user_text)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"response_text": response.content}


async def deep_reasoning_node(state: CounselingState) -> dict:
    user_id = state.get("user_id", "")
    user_text = latest_user_message(state.get("messages", []))

    kb_guidelines = await search_psychology_kb(user_text, top_k=3)
    student_knowledge = await search_student_knowledge_kb(user_text, top_k=3)
    user_memory = await search_user_memory_kb(user_id=user_id, query=user_text, top_k=3)

    # Escalation pattern detection
    escalation_patterns = [
        "escalate to emergency services",
        "escalate to human intervention",
        "escalate case",
        "escalate to authorities",
        "escalate to trusted adult",
        "escalate to emergency help",
        "escalate to emergency",
        "escalate to human",
        "escalate",
    ]
    escalation_reason = None
    human_takeover = False
    # Parse student_knowledge for should_do field(s)
    import re
    for chunk in (student_knowledge or "").split("---"):
        m = re.search(r"should_do:\s*(.*)", chunk, re.IGNORECASE)
        if m:
            should_do = m.group(1).lower()
            for pat in escalation_patterns:
                if pat in should_do:
                    human_takeover = True
                    # escalation_reason: include should_do + user_intent if possible
                    intent_match = re.search(r"intent:\s*(.*)", chunk, re.IGNORECASE)
                    escalation_reason = f"should_do: {should_do}"
                    if intent_match:
                        escalation_reason += f" | intent: {intent_match.group(1)}"
                    break
        if human_takeover:
            break

    llm = get_llm()
    messages = [
        SystemMessage(content=get_deep_reasoning_system_prompt()),
        HumanMessage(
            content=get_deep_reasoning_user_prompt(
                user_text,
                kb_guidelines,
                user_memory,
                student_knowledge,
            )
        ),
    ]
    response = await llm.ainvoke(messages)

    result = {
        "kb_guidelines": kb_guidelines,
        "student_knowledge": student_knowledge,
        "user_memory": user_memory,
        "response_text": response.content,
    }
    if human_takeover:
        result["human_takeover"] = True
        result["escalation_reason"] = escalation_reason or "escalation detected from should_do"
    return result


async def save_memory_node(state: CounselingState) -> dict:
    user_id = state.get("user_id", "")
    messages = state.get("messages", [])
    response_text = state.get("response_text", "")
    current_topic = state.get("current_topic", "")

    if response_text:
        messages = (messages + [{"role": "assistant", "content": response_text}])[-20:]

    await save_history(user_id, messages)
    await save_topic(user_id, current_topic)

    user_text = latest_user_message(messages)

    llm = get_llm()
    memory_gate_prompt = get_memory_gate_prompt(user_text)
    gate_response = await llm.ainvoke([HumanMessage(content=memory_gate_prompt)])

    parsed = {}
    try:
        parsed = json.loads((gate_response.content or "{}").strip("` \n"))
    except json.JSONDecodeError:
        parsed = {"should_store": False}

    if parsed.get("should_store") and parsed.get("memory"):
        await upsert_user_memory_chunk(user_id, parsed["memory"])

    await update_user_longterm_style(ip_or_user_id=user_id, latest_message=user_text)

    return {"messages": messages}
