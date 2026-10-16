"""Run the real legacy node functions with isolated dependencies for comparison.

Only the repository's trusted function definitions are compiled. Module imports
from legacy nodes are not executed: no dotenv, Redis, Postgres or delivery client
can be initialized. This does not patch globals in the deployed application.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


ROOT = Path(__file__).resolve().parents[2]


def _definitions(path: str, namespace: dict):
    source = (ROOT / path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if path == "legacy/graph/tools.py":
        body = [node for node in body if node.name == "latest_user_message"]
    # Do not inherit this module's postponed annotations: this isolated namespace
    # is deliberately not a registered module for typing.get_type_hints to find.
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec", dont_inherit=True), namespace)


class BaselineEngine:
    def __init__(self, model, store):
        self.model, self.store = model, store
        self._lock = asyncio.Lock()
        self._active_id = ""
        self._calls = []
        namespace = {"json": json, "logging": logging, "logger": logging.getLogger(__name__),
                     "Annotated": Annotated, "Literal": Literal, "TypedDict": TypedDict,
                     "HumanMessage": HumanMessage, "SystemMessage": SystemMessage}
        _definitions("legacy/graph/state.py", namespace)
        _definitions("legacy/graph/edges.py", namespace)
        _definitions("legacy/graph/tools.py", namespace)
        # get_llm is explicitly overwritten below; no model initializer from tools executes.
        for path in ("safety", "intent", "simple_response", "info_gap", "clarification", "deep_reasoning", "save_memory"):
            _definitions(f"legacy/prompts/{path}.py", namespace)
        _definitions("legacy/graph/nodes.py", namespace)
        namespace.update({
            "get_llm": lambda: self,
            "load_takeover_flag": self._takeover,
            "load_history": self._history,
            "load_topic": self._topic,
            "save_history": self._save_history,
            "save_topic": self._save_topic,
            "search_user_memory_kb": self._user_memory,
            "search_psychology_kb": self._psychology,
            "search_student_knowledge_kb": self._student,
            "notify_human_admin": self._notify,
            "set_takeover_flag": self._set_takeover,
            "update_user_longterm_style": self._noop,
            "upsert_user_memory_chunk": self._noop,
        })
        builder = StateGraph(namespace["CounselingState"])
        for name in ("load_memory", "safety_gateway", "handoff_human", "intent_routing",
                     "out_of_scope", "simple_response", "info_gap_assessor", "clarification",
                     "deep_reasoning", "save_memory"):
            builder.add_node(name, namespace[name + "_node"])
        builder.add_edge(START, "load_memory")
        builder.add_conditional_edges("load_memory", namespace["route_after_load_memory"], {"safety_gateway": "safety_gateway", "end": END})
        builder.add_conditional_edges("safety_gateway", namespace["route_after_safety"], {"handoff_human": "handoff_human", "intent_routing": "intent_routing"})
        builder.add_conditional_edges("intent_routing", namespace["route_after_intent"], {"out_of_scope": "out_of_scope", "simple_response": "simple_response", "info_gap_assessor": "info_gap_assessor"})
        builder.add_conditional_edges("info_gap_assessor", namespace["route_after_info_gap"], {"clarification": "clarification", "deep_reasoning": "deep_reasoning"})
        for name in ("out_of_scope", "simple_response", "clarification", "deep_reasoning"):
            builder.add_edge(name, "save_memory")
        builder.add_edge("handoff_human", END)
        builder.add_edge("save_memory", END)
        self.graph = builder.compile()

    @staticmethod
    def provenance() -> dict:
        paths = ["legacy/graph/nodes.py", "legacy/graph/state.py", "legacy/graph/edges.py", "legacy/graph/tools.py"] + [f"legacy/prompts/{name}.py" for name in
            ("safety", "intent", "simple_response", "info_gap", "clarification", "deep_reasoning", "save_memory")]
        return {"engine": "legacy_sandbox", "source_hashes": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
                "limitations": ["delivery/storage/notifications replaced by local dependencies", "legacy OCEAN and durable memory writes disabled"]}

    async def ainvoke(self, messages):
        normalized = [{"role": "system" if isinstance(message, SystemMessage) else "user", "content": message.content} for message in messages]
        text = "\n".join(message["content"] for message in normalized)
        if "phát hiện nguy cơ" in text:
            purpose = "legacy_safety"
        elif "bộ phân loại ý định" in text:
            purpose = "legacy_intent"
        elif "đánh giá thiếu hụt" in text:
            purpose = "legacy_assessment"
        elif "bộ lọc trí nhớ" in text:
            purpose = "legacy_memory"
        else:
            purpose = "legacy_reply"
        reply = await self.model.complete(normalized, purpose=purpose)
        self._calls.append({"purpose": purpose, "usage": reply.usage, "metadata": reply.metadata})
        return SimpleNamespace(content=reply.text)

    async def respond(self, user_id: str, text: str) -> dict:
        # The bound adapter uses an active context; serialize this sandbox instance.
        async with self._lock:
            self._active_id, self._calls = user_id, []
            result = await self.graph.ainvoke({"user_id": user_id, "messages": [{"role": "user", "content": text}]})
            result["response"] = result.get("response_text", "")
            result["model_calls"] = self._calls
            result["assessment"] = {"intent": result.get("intent_category"), "verdict": result.get("info_gap_status")}
            return result

    async def _history(self, user_id):
        return (await self.store.load(user_id)).get("messages", [])

    async def _topic(self, user_id):
        return (await self.store.load(user_id)).get("current_topic", "")

    async def _takeover(self, user_id):
        return (await self.store.load(user_id)).get("handoff_status") == "accepted"

    async def _save_history(self, user_id, messages):
        session = await self.store.load(user_id)
        session["messages"] = messages
        await self.store.save(user_id, session)

    async def _save_topic(self, user_id, topic):
        session = await self.store.load(user_id)
        session["current_topic"] = topic
        await self.store.save(user_id, session)

    async def _user_memory(self, user_id, query, **kwargs):
        return (await self.store.search(user_id, query)).get("user_memory", "")

    async def _psychology(self, query, **kwargs):
        return (await self.store.search(self._active_id, query)).get("psychology", "")

    async def _student(self, query, **kwargs):
        return (await self.store.search(self._active_id, query)).get("student_knowledge", "")

    async def _notify(self, user_id, summary, raw_message):
        return await self.store.notify(user_id, raw_message)

    async def _set_takeover(self, user_id, value):
        session = await self.store.load(user_id)
        session["handoff_status"] = "accepted" if value else "none"
        await self.store.save(user_id, session)

    async def _noop(self, *args, **kwargs):
        return None
