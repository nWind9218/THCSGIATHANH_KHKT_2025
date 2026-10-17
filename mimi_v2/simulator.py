"""Local-only Facebook conversation simulator. No production connector imports."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from mimi_v2.core import CounselingEngine
from mimi_v2.store import InMemoryStore
from mimi_v2.teencode import scan


class OfflineModel:
    """Deterministic harness fixture, NOT an intelligence/quality benchmark."""
    def __init__(self):
        self.fault = "none"

    async def complete(self, messages, purpose="reply"):
        if self.fault == "model":
            raise TimeoutError("simulated model timeout")
        text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if purpose.startswith("legacy_"):
            matches = re.findall(r'Tin nhắn(?: mới nhất(?: của người dùng)?| của người dùng)?: "(.*?)"\n', text, re.DOTALL)
            if matches:
                text = matches[-1]
        lower = text.casefold()
        all_text = " ".join(m["content"] for m in messages if m["role"] == "user").casefold()
        found = scan(text)
        # Same deterministic detector as production; "tự làm hại" stays as a fixture-only phrase.
        urgent = bool(found.tier1) or "want_die" in found.tier2 or "tự làm hại" in lower
        if purpose == "legacy_safety":
            output = "YES" if urgent else "NO"
        elif purpose == "legacy_intent":
            output = json.dumps({"intent": "complex", "topic": "offline fixture", "core_issue": text})
        elif purpose == "legacy_assessment":
            output = json.dumps({"verdict": "sufficient", "known": {}, "missing_fields": [], "next_question_focus": "", "safety_hint": False})
        elif purpose == "legacy_memory":
            output = json.dumps({"should_store": False})
        elif purpose == "safety":
            output = json.dumps({"risk": "urgent" if urgent else "safe", "confidence": "high", "reason": "offline fixture"})
        elif purpose == "deliberate":
            output = json.dumps({"cues": [], "about_self": True, "negated_or_denied": False, "joking_or_hyperbole": False,
                                 "quoted_or_hypothetical": False, "intent": "active" if urgent else "none", "plan_or_means": False,
                                 "abuse_or_danger_from_others": False, "context_trend": "",
                                 "risk": "urgent" if urgent else "safe", "reason": "offline fixture"})
        elif purpose == "assessment":
            parent = "phụ huynh" in all_text or "tôi là mẹ" in all_text or "tôi là bố" in all_text
            listen = "lắng nghe" in all_text or "không cần lời khuyên" in all_text
            closing = any(s in lower for s in ("cảm ơn", "nghỉ một chút", "tạm biệt"))
            output = json.dumps({"known": {}, "missing_fields": [], "next_question_focus": "",
                                 "need": "listen" if listen else "advice", "intent": "closing" if closing else "support",
                                 "topic": "offline fixture", "core_issue": text, "safety_hint": "safe",
                                 "user_role": "parent" if parent else "unknown"}, ensure_ascii=False)
        elif purpose in {"reply", "legacy_reply"}:
            output = ("Mimi hiểu anh/chị muốn gần gũi với con hơn. Anh/chị có thể bắt đầu bằng việc lắng nghe và tôn trọng nhịp chia sẻ của con." if "phụ huynh" in all_text else
                      "Mimi đang lắng nghe. Những điều bạn vừa chia sẻ được giữ trong cuộc trò chuyện này; bạn có thể nói tiếp khi thấy thoải mái.")
        else:
            raise ValueError("Unsupported offline model purpose")
        return SimpleNamespace(text=output, usage={}, metadata={"model": "deterministic-fixture", "api": "offline", "purpose": purpose})


class FaultStore(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.fault = "none"

    async def load(self, user_id):
        if self.fault == "store":
            raise ConnectionError("simulated storage outage")
        return await super().load(user_id)

    async def notify(self, user_id, text):
        if self.fault == "notify":
            raise ConnectionError("simulated notification outage")
        return await super().notify(user_id, text)


def load_local_model_environment(path: str | Path = ".env") -> dict:
    """Copy only authorized model variables, never Facebook/DB/Redis credentials."""
    allowed = {"OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "MIMI_V2_BASE_URL", "MIMI_V2_API_KEY", "MIMI_V2_MODEL", "MIMI_V2_API",
               "MIMI_V2_REASONING_EFFORT", "MIMI_V2_VERBOSITY", "MIMI_V2_MAX_OUTPUT_TOKENS", "MIMI_V2_TIMEOUT",
               "MIMI_V2_ROUTING", "MIMI_V2_FALLBACK_MODEL", "MIMI_V2_FALLBACK_EFFORT"}
    prefixes = ("MIMI_V2_MODEL_", "MIMI_V2_EFFORT_")  # per-flow routing overrides read by RoutedModel
    ok = lambda key: key in allowed or key.startswith(prefixes)
    result = {key: value for key, value in os.environ.items() if ok(key) and value}
    if Path(path).is_file():
        from dotenv import dotenv_values
        values = dotenv_values(path, interpolate=False)
        for key, value in values.items():
            if ok(key) and key not in result and value:
                result[key] = value
    return result


def make_engine(config: dict | None = None):
    config = config or {}
    if set(config) - {"live", "api", "model", "engine", "reasoning_effort", "verbosity", "max_output_tokens", "timeout", "routing"}:
        raise ValueError("Unknown engine configuration")
    if config.get("engine", "v2") not in {"v2", "baseline"}:
        raise ValueError("Unknown engine")
    if type(config.get("live", False)) is not bool:
        raise ValueError("live must be boolean")
    if config.get("live"):
        from mimi_v2.models import ModelConfig, OpenAIModel, resolve_endpoint
        env = load_local_model_environment()
        base_url, key = resolve_endpoint(env)
        if not key:
            raise ValueError("API key required for explicit live mode")
        settings = ModelConfig.from_env(env).public_config()
        settings.update({k: v for k, v in config.items() if k not in {"live", "engine", "routing"}})
        model_config = ModelConfig(**settings)
        if config.get("routing") == "adaptive":
            from mimi_v2.models import RoutedModel
            model = RoutedModel.from_env(env, api_key=key, base_url=base_url)
        else:
            model = OpenAIModel(model_config, api_key=key, base_url=base_url)
    else:
        model = OfflineModel()
    store = FaultStore()
    if config.get("engine") == "baseline":
        from legacy.eval_tools.baseline import BaselineEngine
        return BaselineEngine(model, store)
    return CounselingEngine(model, store)


class LocalSession:
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.engine = make_engine(config)
        self.user_id = "local-" + uuid4().hex
        self._operation_lock = asyncio.Lock()

    async def send(self, text, fault="none"):
        if fault not in {"none", "model", "store", "notify"}:
            raise ValueError("Unknown fault")
        async with self._operation_lock:
            self.engine.store.fault = fault
            original = self.engine.model
            if isinstance(original, OfflineModel):
                original.fault = fault
            elif fault == "model":
                # Exercise the real core fallback even in live mode, with zero
                # provider requests for this injected failure.
                class TimeoutModel:
                    async def complete(self, *args, **kwargs):
                        raise TimeoutError("simulated model timeout")
                self.engine.model = TimeoutModel()
            try:
                result = await self.engine.respond(self.user_id, text)
                return result.get("messages", []), result
            except Exception as exc:
                return [], {"error": type(exc).__name__, "detail": "Local test operation failed"}
            finally:
                self.engine.model = original

    async def control(self, action):
        if action not in {"reset", "acknowledge", "resume"}:
            raise ValueError("Unknown local control")
        async with self._operation_lock:
            self.engine.store.fault = "none"
            await getattr(self.engine.store, action)(self.user_id)
            if action == "reset":
                # Reset process-local recovery buffers, model faults and alerts
                # as well as stored history; no stale data can reappear.
                self.engine = make_engine(self.config)
            return await self.engine.store.load(self.user_id)


def build_app(config=None):
    import gradio as gr
    # Default Gradio theme loads Google Fonts even for an offline simulator.
    theme = gr.themes.Default(font=["system-ui", "Arial", "sans-serif"],
                              font_mono=["Consolas", "monospace"])
    with gr.Blocks(title="Mimi — mô phỏng Messenger local", analytics_enabled=False, theme=theme) as app:
        routing_note = " Routing: **adaptive**." if (config or {}).get("routing") == "adaptive" else " Routing: **off**."
        gr.Markdown("# Mimi — kiểm thử Messenger local\nDữ liệu chỉ nằm trong RAM. Không kết nối Facebook, database, email hoặc alert production. "
                    + ("**Live model đang bật: request OpenAI có thể phát sinh phí.**" if (config or {}).get("live") else "Model giả lập offline: kiểm tra luồng, không đánh giá chất lượng trí tuệ.")
                    + (routing_note if (config or {}).get("live") else ""))
        state = gr.State(None)
        identity = gr.Textbox(label="PSID giả lập của phiên", interactive=False)
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Cuộc trò chuyện", type="messages", height=420)
                text = gr.Textbox(label="Tin nhắn", placeholder="Nhập tin nhắn để gửi vào lõi Mimi v2")
                send = gr.Button("Gửi tin nhắn", variant="primary")
            with gr.Column():
                fault = gr.Dropdown(["none", "model", "store", "notify"], value="none", label="Mô phỏng lỗi")
                gr.Markdown("`model`: timeout · `store`: không đọc được memory · `notify`: không gửi được cảnh báo local")
                accept = gr.Button("Người hỗ trợ tiếp nhận")
                resume = gr.Button("Cho chatbot tiếp tục")
                reset = gr.Button("Xóa phiên test")
                diagnostics = gr.JSON(label="Assessment, nguồn, handoff, thời gian và lỗi")

        def init():
            session = LocalSession(config)
            return session, session.user_id

        async def submit(value, selected, session):
            if session is None:
                session = LocalSession(config)
            if not value or not value.strip():
                return gr.skip(), {"error": "Hãy nhập tin nhắn."}, value, session, session.user_id
            history, result = await session.send(value, selected)
            return history, result, "", session, session.user_id

        async def command(action, session):
            if session is None:
                session = LocalSession(config)
            try:
                result = await session.control(action)
                return result.get("messages", []), result, session
            except Exception as exc:
                return gr.skip(), {"error": type(exc).__name__, "detail": "Cần trạng thái pending để tiếp nhận."}, session

        app.load(init, outputs=[state, identity])
        send.click(submit, [text, fault, state], [chat, diagnostics, text, state, identity], concurrency_limit=1)
        text.submit(submit, [text, fault, state], [chat, diagnostics, text, state, identity], concurrency_limit=1)
        for button, action in ((accept, "acknowledge"), (resume, "resume"), (reset, "reset")):
            async def handle(session, action=action):
                return await command(action, session)
            button.click(handle, [state], [chat, diagnostics, state], concurrency_limit=1)
    return app


def resolve_routing(flag: str | None = None) -> str:
    """CLI flag wins; otherwise follow MIMI_V2_ROUTING like production."""
    if flag:
        return flag
    value = load_local_model_environment().get("MIMI_V2_ROUTING", "").strip().lower()
    return "adaptive" if value == "adaptive" else "off"


def main():
    parser = argparse.ArgumentParser(description="Local-only Gradio Messenger simulator")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--api", choices=["responses", "chat_completions"], default="responses")
    parser.add_argument("--model", default=None, help="defaults to MIMI_V2_MODEL / OPENAI_CHAT_MODEL")
    parser.add_argument("--routing", choices=["adaptive", "off"], default=None,
                        help="defaults to MIMI_V2_ROUTING from the local model environment (as production)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    config = {"live": args.live, "api": args.api, **({"model": args.model} if args.model else {})}
    if args.live and resolve_routing(args.routing) == "adaptive":
        config["routing"] = "adaptive"
    build_app(config).queue().launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_error=False)


if __name__ == "__main__":
    main()
