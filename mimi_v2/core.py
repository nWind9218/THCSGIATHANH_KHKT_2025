"""Opt-in counseling workflow; dependencies supplied by transport or local runner."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from contextlib import AsyncExitStack
from time import perf_counter
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from .prompts import ASSESSMENT, LISTEN_MODE, PERSONA, SAFETY
from .store import SessionStore


class Safety(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    risk: Literal["safe", "urgent", "uncertain"]
    reason: str


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    known: dict[str, str]
    missing_fields: list[str]
    next_question_focus: str
    need: Literal["listen", "comfort", "clarify", "advice", "explain"]
    intent: Literal["simple", "support", "closing", "out_of_scope"]
    topic: str
    core_issue: str
    safety_hint: Literal["safe", "urgent", "uncertain"]
    user_role: Literal["unknown", "student", "parent"]


class Turn(TypedDict, total=False):
    user_id: str
    session: dict
    response: str
    assessment: dict
    safety: str
    sources: dict
    diagnostics: list[str]
    timings: dict
    usage: dict
    model_calls: list[dict]


def canonical_session(raw: dict | list) -> dict:
    """Read old role/content lists without deleting storage or doubling history."""
    session = deepcopy(raw if isinstance(raw, dict) else {"messages": raw})
    session["messages"] = [
        {"role": m["role"], "content": m["content"]}
        for m in session.get("messages", session.get("history", []))
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ][-20:]
    session.setdefault("known", {})
    session.setdefault("user_role", "unknown")
    session.setdefault("handoff_status", "none")
    session.setdefault("notification_sent", False)
    return session


class CounselingEngine:
    def __init__(self, model, store: SessionStore):
        self.model, self.store = model, store
        self._locks: dict[str, asyncio.Lock] = {}
        self._recovery: dict[str, dict] = {}
        graph = StateGraph(Turn)
        for name in ("safety", "assessment", "retrieve", "reply", "handoff", "persist"):
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "safety")
        graph.add_conditional_edges("safety", lambda s: "assessment" if s["safety"] == "safe" else "handoff")
        graph.add_conditional_edges("assessment", lambda s: "retrieve" if s["safety"] == "safe" else "handoff")
        graph.add_edge("retrieve", "reply")
        graph.add_edge("reply", "persist")
        graph.add_edge("handoff", "persist")
        graph.add_edge("persist", END)
        # SessionStore is the sole persistence owner; no competing history reducer.
        self.graph = graph.compile()

    async def _complete(self, state: Turn, purpose: str, prompt: str):
        start = perf_counter()
        messages = [{"role": "system", "content": prompt}, *state["session"]["messages"]]
        try:
            answer = await self.model.complete(messages, purpose=purpose)
            state["model_calls"].append({"purpose": purpose, **getattr(answer, "metadata", {})})
            for key, value in getattr(answer, "usage", {}).items():
                if isinstance(value, (int, float)):
                    state["usage"][key] = state["usage"].get(key, 0) + value
            return answer.text
        finally:
            state["timings"][purpose] = perf_counter() - start

    async def _safety(self, state: Turn) -> dict:
        try:
            parsed = Safety.model_validate_json(await self._complete(state, "safety", SAFETY))
            risk = parsed.risk
        except Exception:
            state["diagnostics"].append("safety_unavailable_or_invalid")
            risk = "uncertain"
        return {"safety": risk, "diagnostics": state["diagnostics"],
                "timings": state["timings"], "usage": state["usage"], "model_calls": state["model_calls"]}

    async def _assessment(self, state: Turn) -> dict:
        previous = state["session"]
        context = json.dumps({k: previous.get(k) for k in ("known", "topic", "core_issue", "need", "user_role")}, ensure_ascii=False)
        try:
            item = Assessment.model_validate_json(await self._complete(state, "assessment", ASSESSMENT + "\nĐã biết: " + context)).model_dump()
        except Exception:
            state["diagnostics"].append("assessment_unavailable_or_invalid")
            return {"safety": "uncertain", "diagnostics": state["diagnostics"]}
        session = deepcopy(previous)
        session["known"].update(item["known"])
        item["known"] = deepcopy(session["known"])
        item["missing_fields"] = [key for key in item["missing_fields"] if not item["known"].get(key)]
        if item["next_question_focus"] not in item["missing_fields"]:
            item["next_question_focus"] = ""
        if item["user_role"] == "unknown":
            item["user_role"] = session["user_role"]
        for key in ("topic", "core_issue", "need", "user_role"):
            session[key] = item[key]
        session["current_topic"] = item["topic"]
        return {"session": session, "assessment": item, "safety": item["safety_hint"],
                "timings": state["timings"], "usage": state["usage"], "model_calls": state["model_calls"]}

    async def _retrieve(self, state: Turn) -> dict:
        item = state["assessment"]
        if item["intent"] != "support" or item["need"] == "listen":
            return {"sources": {}}
        start = perf_counter()
        try:
            sources = await self.store.search(state["user_id"], item["core_issue"] or state["session"]["messages"][-1]["content"])
            if isinstance(sources, dict):
                state["diagnostics"].extend(str(error) for error in sources.get("errors", []))
        except Exception:
            sources = {}
            state["diagnostics"].append("retrieval_unavailable")
        state["timings"]["retrieval"] = perf_counter() - start
        return {"sources": sources, "diagnostics": state["diagnostics"], "timings": state["timings"]}

    async def _reply(self, state: Turn) -> dict:
        if state["assessment"]["intent"] == "closing":
            return {"response": "Cảm ơn bạn đã chia sẻ. Khi cần, bạn có thể quay lại trò chuyện cùng Mimi."}
        context = json.dumps({"assessment": state["assessment"], "reference_material": state.get("sources", {}),
                              "handoff_status": state["session"]["handoff_status"]}, ensure_ascii=False)
        mode = LISTEN_MODE if state["assessment"]["need"] == "listen" else ""
        try:
            response = (await self._complete(state, "reply", PERSONA + "\nNgữ cảnh hỗ trợ: " + context + "\n" + mode)).strip()
            if not response:
                raise ValueError("Empty reply")
        except Exception:
            state["diagnostics"].append("reply_unavailable_or_invalid")
            response = "Mimi đang gặp trục trặc khi phản hồi. Bạn không cần kể lại những điều vừa chia sẻ; hãy thử gửi tiếp sau một chút."
        return {"response": response, "diagnostics": state["diagnostics"],
                "timings": state["timings"], "usage": state["usage"], "model_calls": state["model_calls"]}

    async def _handoff(self, state: Turn) -> dict:
        session = deepcopy(state["session"])
        session["handoff_status"] = "pending"
        if not session["notification_sent"]:
            try:
                session["notification_sent"] = bool(await self.store.notify(state["user_id"], state["session"]["messages"][-1]["content"]))
            except Exception:
                session["notification_sent"] = False
            if not session["notification_sent"]:
                state["diagnostics"].append("notification_unavailable")
        response = ("Mimi muốn bảo đảm bạn đang an toàn. Nếu có nguy hiểm ngay lúc này, hãy đến chỗ có người đáng tin cậy "
                    "và nhờ họ ở cùng, hỗ trợ trực tiếp. ")
        response += ("Mimi đã gửi yêu cầu hỗ trợ nhưng chưa có xác nhận người tiếp nhận. " if session["notification_sent"] else
                     "Mimi chưa gửi được yêu cầu hỗ trợ; bạn hãy chủ động liên hệ người lớn đáng tin cậy ở gần. ")
        response += "Bạn có đang ở nơi an toàn không?"
        return {"session": session, "response": response, "diagnostics": state["diagnostics"]}

    async def _persist(self, state: Turn) -> dict:
        session = deepcopy(state["session"])
        if state.get("response"):
            session["messages"].append({"role": "assistant", "content": state["response"]})
        session["messages"] = session["messages"][-20:]
        try:
            await self.store.save(state["user_id"], session)
            self._recovery.pop(state["user_id"], None)
        except Exception:
            self._recovery[state["user_id"]] = deepcopy(session)
            state["diagnostics"].append("session_save_failed_local_recovery_only")
        return {"session": session, "diagnostics": state["diagnostics"]}

    async def respond(self, user_id: str, text: str) -> dict:
        if not isinstance(user_id, str) or not user_id.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("user_id and text must be nonempty strings")
        lock = self.store.session_lock(user_id) if hasattr(self.store, "session_lock") else self._locks.setdefault(user_id, asyncio.Lock())
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(lock)
            if hasattr(self.store, "conversation_lock"):
                await stack.enter_async_context(self.store.conversation_lock(user_id))
            started = perf_counter()
            diagnostics = []
            try:
                raw = await self.store.load(user_id)
                if user_id in self._recovery:
                    recovered = deepcopy(self._recovery[user_id])
                    # Always re-read takeover so process-local failure recovery cannot
                    # overwrite a subsequent human acceptance from another worker.
                    if isinstance(raw, dict) and raw.get("handoff_status") == "accepted":
                        recovered["handoff_status"] = "accepted"
                    raw = recovered
            except Exception:
                # A failed read may hide a human takeover. Do not overwrite unknown state.
                fallback = "Mimi đang gặp trục trặc khi đọc cuộc trò chuyện. Bạn hãy thử lại sau một chút; nếu đang không an toàn, hãy nhờ người đáng tin cậy ở gần hỗ trợ ngay."
                return {"response": fallback, "response_text": fallback,
                        "messages": [], "assessment": {}, "handoff_status": "unknown", "diagnostics": ["session_load_failed"],
                        "timings": {"total": perf_counter() - started}, "usage": {}, "model_calls": []}
            session = canonical_session(raw)
            session["messages"].append({"role": "user", "content": text.strip()})
            session["messages"] = session["messages"][-20:]
            state = Turn(user_id=user_id, session=session, diagnostics=diagnostics, timings={}, usage={}, model_calls=[], assessment={}, sources={})
            if session["handoff_status"] == "accepted":
                state["response"] = ""
                state.update(await self._persist(state))
                result = state
            else:
                result = await self.graph.ainvoke(state)
            result["timings"]["total"] = perf_counter() - started
            return {"response": result.get("response", ""), "response_text": result.get("response", ""), "messages": result["session"]["messages"],
                    "assessment": result.get("assessment", {}), "handoff_status": result["session"]["handoff_status"],
                    "sources": result.get("sources", {}), "diagnostics": result["diagnostics"],
                    "timings": result["timings"], "usage": result["usage"], "model_calls": result["model_calls"]}
