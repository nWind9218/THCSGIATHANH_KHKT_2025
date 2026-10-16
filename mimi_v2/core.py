"""Opt-in counseling workflow; dependencies supplied by transport or local runner."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from copy import deepcopy
from contextlib import AsyncExitStack
from time import perf_counter
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from .contracts import ContractError, normalize_assessment, parse_decision
from .models import ModelError
from .tracing import Trace, model_attrs, summarize

from .prompts import (ASSESSMENT, HANDOFF_MESSAGE, HANDOFF_REQUEST_NOT_SENT, HANDOFF_REQUEST_SENT,
                      LISTEN_CLOSING_CONFLICT, LISTEN_MODE, PERSONA, SAFETY)
from .store import SessionStore


logger = logging.getLogger("mimi_v2.turn")

# A failed attempt slower than this is not retried: with the production lock lease
# (300s) two retried stages plus the reply must still fit (2 x (10 + 60) + 60 < 300).
RETRY_MAX_ATTEMPT_SECONDS = 10.0
RETRY_BACKOFF_SECONDS = 0.5


def is_transient(exc: Exception) -> bool:
    """Provider/transport failures worth one retry. Never timeouts, auth, quota or bad content."""
    if not isinstance(exc, ModelError):
        return False
    if exc.code in {"network_error", "rate_limit"}:
        return True
    return exc.code == "provider_error" and exc.status in {408, 409, 500, 502, 503, 504}


def model_error_label(stage: str, exc: Exception) -> list[str]:
    if isinstance(exc, ModelError):
        return [f"{stage}_model_error:{exc.code}:{exc.status or '-'}"]
    return []


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


class StoredIdentity(BaseModel):
    model_config = ConfigDict(strict=True)
    known: dict[str, str]
    user_role: Literal["unknown", "student", "parent"]
    handoff_status: Literal["none", "pending", "accepted"]
    notification_sent: bool


def decision_error(stage: str, exc: Exception) -> list[str]:
    """Safe field/type diagnostics; no model payload, validation inputs or fact keys."""
    if isinstance(exc, ValidationError):
        fields = set(Assessment.model_fields) | set(Safety.model_fields) | set(StoredIdentity.model_fields)
        return [f"{stage}_validation:{error['loc'][0] if error['loc'] and error['loc'][0] in fields else 'payload'}:{error['type']}"
                for error in exc.errors(include_input=False, include_context=False)][:8]
    if isinstance(exc, ContractError):
        return [f"{stage}_validation:{exc.field}:{exc.code}"]
    return [f"{stage}_error_type:{type(exc).__name__}"]


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
    trace: object


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
    if not isinstance(raw, (dict, list)):
        raise ContractError("invalid_session", "payload")
    normalized = normalize_assessment({"known": session["known"]})
    StoredIdentity.model_validate({**session, "known": normalized["known"]})
    # Stored facts are always a dict; wire lists belong only to model transport.
    if not isinstance(session["known"], dict):
        raise ContractError("stored_facts_must_be_dict")
    return session


class CounselingEngine:
    def __init__(self, model, store: SessionStore):
        self.model, self.store = model, store
        self._locks: dict[str, asyncio.Lock] = {}
        self._recovery: dict[str, dict] = {}
        graph = StateGraph(Turn)
        for name in ("screen", "retrieve", "reply", "handoff", "persist"):
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "screen")
        graph.add_conditional_edges("screen", lambda s: "retrieve" if s["safety"] == "safe" else "handoff")
        graph.add_edge("retrieve", "reply")
        graph.add_edge("reply", "persist")
        graph.add_edge("handoff", "persist")
        graph.add_edge("persist", END)
        # SessionStore is the sole persistence owner; no competing history reducer.
        self.graph = graph.compile()

    async def _complete(self, state: Turn, purpose: str, prompt: str, retry: bool = False, effort: str | None = None):
        start = perf_counter()
        messages = [{"role": "system", "content": prompt}, *state["session"]["messages"]]
        answer, retried = None, False
        # Only an adaptive router understands these; every other model keeps the plain call.
        routed = getattr(self.model, "routable", False)
        extra = {"effort": effort} if routed and effort else {}
        try:
            attempt = perf_counter()
            try:
                answer = await self.model.complete(messages, purpose=purpose, **extra)
            except Exception as exc:
                # One retry for a fast transient failure, before any handoff decision.
                if not (retry and is_transient(exc) and perf_counter() - attempt <= RETRY_MAX_ATTEMPT_SECONDS):
                    raise
                state["diagnostics"].append(f"{purpose}_retried:{exc.code}:{exc.status or '-'}")
                retried = True
                await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                answer = await self.model.complete(messages, purpose=purpose, **extra, **({"fallback": True} if routed else {}))
            state["model_calls"].append({"purpose": purpose, **getattr(answer, "metadata", {})})
            for key, value in getattr(answer, "usage", {}).items():
                if isinstance(value, (int, float)):
                    state["usage"][key] = state["usage"].get(key, 0) + value
            return answer.text
        finally:
            end = perf_counter()
            state["timings"][purpose] = state["timings"].get(purpose, 0) + (end - start)
            trace = state.get("trace")
            if trace is not None:
                error = sys.exc_info()[1]
                trace.add(f"model.{purpose}", start, end, **model_attrs(getattr(answer, "usage", None)),
                          retried=True if retried else None,
                          error=(getattr(error, "code", type(error).__name__) if error else None))

    async def _screen(self, state: Turn) -> dict:
        """Run the safety check and the assessment concurrently (they only read the same messages).

        Safety stays the gate: unless it says "safe" the assessment is discarded, exactly as if it
        had never run, and the turn goes to handoff. An assessment failure still fails closed.
        """
        safety, assessment = await asyncio.gather(self._safety(state), self._assessment(state))
        if safety["safety"] != "safe":
            state["diagnostics"].append("assessment_discarded_not_safe")
            return {"safety": safety["safety"], "diagnostics": state["diagnostics"],
                    "timings": state["timings"], "usage": state["usage"], "model_calls": state["model_calls"]}
        return assessment

    async def _safety(self, state: Turn) -> dict:
        try:
            parsed = Safety.model_validate(parse_decision(await self._complete(state, "safety", SAFETY, retry=True)))
            risk = parsed.risk
            if risk == "uncertain" and getattr(self.model, "routable", False):
                # Cheap first pass flagged it: take one stronger second opinion before a handoff.
                # A failed or invalid recheck keeps "uncertain", so this can only reduce false alarms.
                try:
                    risk = Safety.model_validate(parse_decision(await self._complete(state, "safety", SAFETY, retry=True, effort="medium"))).risk
                    state["diagnostics"].append(f"safety_rechecked:{risk}")
                except Exception:
                    state["diagnostics"].append("safety_recheck_failed")
        except Exception as exc:
            state["diagnostics"].append("safety_unavailable_or_invalid")
            state["diagnostics"].extend(decision_error("safety", exc))
            state["diagnostics"].extend(model_error_label("safety", exc))
            risk = "uncertain"
        return {"safety": risk, "diagnostics": state["diagnostics"],
                "timings": state["timings"], "usage": state["usage"], "model_calls": state["model_calls"]}

    async def _assessment(self, state: Turn) -> dict:
        previous = state["session"]
        context = json.dumps({k: previous.get(k) for k in ("known", "topic", "core_issue", "need", "user_role")}, ensure_ascii=False)
        try:
            payload = parse_decision(await self._complete(state, "assessment", ASSESSMENT + "\nĐã biết: " + context, retry=True))
            item = Assessment.model_validate(normalize_assessment(payload)).model_dump()
        except Exception as exc:
            state["diagnostics"].append("assessment_unavailable_or_invalid")
            state["diagnostics"].extend(decision_error("assessment", exc))
            state["diagnostics"].extend(model_error_label("assessment", exc))
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
            inner = sources.pop("_timings", None) if isinstance(sources, dict) else None
            if isinstance(sources, dict):
                state["diagnostics"].extend(str(error) for error in sources.get("errors", []))
        except Exception:
            sources, inner = {}, None
            state["diagnostics"].append("retrieval_unavailable")
        state["timings"]["retrieval"] = perf_counter() - start
        if state.get("trace") is not None:
            state["trace"].add("store.search", start, perf_counter(), **(inner if isinstance(inner, dict) else {}))
        return {"sources": sources, "diagnostics": state["diagnostics"], "timings": state["timings"]}

    async def _reply(self, state: Turn) -> dict:
        closing = state["assessment"]["intent"] == "closing"
        listening = state["assessment"]["need"] == "listen"
        if closing and not listening:
            return {"response": "Cảm ơn bạn đã chia sẻ. Khi cần, bạn có thể quay lại trò chuyện cùng Mimi."}
        context = json.dumps({"assessment": state["assessment"], "reference_material": state.get("sources", {}),
                              "handoff_status": state["session"]["handoff_status"]}, ensure_ascii=False)
        mode = LISTEN_MODE if listening else ""
        if closing and listening:
            # Neither label alone proves the user's intention. Let the reply
            # consume the actual context rather than forcing a canned goodbye.
            # This is routing/payload hardening, not a semantic quality guarantee.
            mode += "\n" + LISTEN_CLOSING_CONFLICT
        try:
            # Listening/closing replies are short reflections; the rest keep the router's default.
            effort = "none" if listening or closing else ("medium" if state.get("sources") and state["assessment"]["need"] == "explain" else None)
            response = (await self._complete(state, "reply", PERSONA + "\nNgữ cảnh hỗ trợ: " + context + "\n" + mode,
                                                 retry=getattr(self.model, "routable", False), effort=effort)).strip()
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
            notify_start = perf_counter()
            try:
                session["notification_sent"] = bool(await self.store.notify(state["user_id"], state["session"]["messages"][-1]["content"]))
            except Exception:
                session["notification_sent"] = False
            if state.get("trace") is not None:
                state["trace"].add("store.notify", notify_start, perf_counter(), sent=session["notification_sent"])
            if not session["notification_sent"]:
                state["diagnostics"].append("notification_unavailable")
        response = HANDOFF_MESSAGE + (HANDOFF_REQUEST_SENT if session["notification_sent"] else HANDOFF_REQUEST_NOT_SENT)
        return {"session": session, "response": response, "diagnostics": state["diagnostics"]}

    async def _persist(self, state: Turn) -> dict:
        session = deepcopy(state["session"])
        if state.get("response"):
            session["messages"].append({"role": "assistant", "content": state["response"]})
        session["messages"] = session["messages"][-20:]
        save_start, save_ok = perf_counter(), True
        try:
            await self.store.save(state["user_id"], session)
            self._recovery.pop(state["user_id"], None)
        except Exception:
            save_ok = False
            self._recovery[state["user_id"]] = deepcopy(session)
            state["diagnostics"].append("session_save_failed_local_recovery_only")
        if state.get("trace") is not None:
            state["trace"].add("store.save", save_start, perf_counter(), ok=save_ok)
        return {"session": session, "diagnostics": state["diagnostics"]}

    @staticmethod
    def _log_turn(user_id: str, result: dict) -> None:
        """One line per turn: decisions, error codes and timings only; never message text."""
        assessment = result.get("assessment") or {}
        info = {
            "uid": hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:10],
            "handoff": result.get("handoff_status"),
            "safety_hint": assessment.get("safety_hint"),
            "need": assessment.get("need"), "intent": assessment.get("intent"),
            "diagnostics": result.get("diagnostics", []),
            "calls": [c.get("purpose") for c in result.get("model_calls", [])],
            "timings": {k: round(v, 2) for k, v in result.get("timings", {}).items()},
            "tokens": result.get("usage", {}).get("total_tokens"),
            "spans": summarize(result.get("trace", {})),
        }
        flagged = bool(info["diagnostics"]) or info["handoff"] in {"pending", "unknown"}
        logger.log(logging.WARNING if flagged else logging.INFO, "turn %s", json.dumps(info, ensure_ascii=False))

    async def respond(self, user_id: str, text: str) -> dict:
        if not isinstance(user_id, str) or not user_id.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("user_id and text must be nonempty strings")
        trace = Trace()
        lock = self.store.session_lock(user_id) if hasattr(self.store, "session_lock") else self._locks.setdefault(user_id, asyncio.Lock())
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(lock)
            if hasattr(self.store, "conversation_lock"):
                await stack.enter_async_context(self.store.conversation_lock(user_id))
            started = perf_counter()
            trace.add("lock_wait", trace.t0, started)
            diagnostics = []
            try:
                load_start = perf_counter()
                raw = await self.store.load(user_id)
                if user_id in self._recovery:
                    recovered = deepcopy(self._recovery[user_id])
                    # Always re-read takeover so process-local failure recovery cannot
                    # overwrite a subsequent human acceptance from another worker.
                    if isinstance(raw, dict) and raw.get("handoff_status") == "accepted":
                        recovered["handoff_status"] = "accepted"
                    raw = recovered
                session = canonical_session(raw)
                trace.add("store.load", load_start, perf_counter())
            except Exception as exc:
                # A failed read may hide a human takeover. Do not overwrite unknown state.
                fallback = "Mimi đang gặp trục trặc khi đọc cuộc trò chuyện. Bạn hãy thử lại sau một chút; nếu đang không an toàn, hãy nhờ người đáng tin cậy ở gần hỗ trợ ngay."
                failed = {"response": fallback, "response_text": fallback,
                          "messages": [], "assessment": {}, "handoff_status": "unknown", "diagnostics": ["session_load_failed", *decision_error("session", exc)],
                          "timings": {"total": perf_counter() - started}, "usage": {}, "model_calls": []}
                failed["trace"] = trace.finish()
                self._log_turn(user_id, failed)
                return failed
            session["messages"].append({"role": "user", "content": text.strip()})
            session["messages"] = session["messages"][-20:]
            state = Turn(user_id=user_id, session=session, diagnostics=diagnostics, timings={}, usage={}, model_calls=[], assessment={}, sources={}, trace=trace)
            if session["handoff_status"] == "accepted":
                state["response"] = ""
                state.update(await self._persist(state))
                result = state
            else:
                result = await self.graph.ainvoke(state)
            result["timings"]["total"] = perf_counter() - started
            output = {"response": result.get("response", ""), "response_text": result.get("response", ""), "messages": result["session"]["messages"],
                      "assessment": result.get("assessment", {}), "handoff_status": result["session"]["handoff_status"],
                      "sources": result.get("sources", {}), "diagnostics": result["diagnostics"],
                      "timings": result["timings"], "usage": result["usage"], "model_calls": result["model_calls"]}
            output["trace"] = trace.finish()
            self._log_turn(user_id, output)
            return output
