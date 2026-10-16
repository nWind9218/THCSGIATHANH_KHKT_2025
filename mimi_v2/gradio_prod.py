"""Gradio chat wired to the same production engine and storage as the Facebook webhook.

Unlike mimi_v2.simulator this module DOES connect to Upstash Redis, Supabase Postgres,
the model provider and the emergency email/log path. Safety boundary: every session id
is forced to start with TEST_PREFIX, so real PSIDs can never be read, written or deleted.
"""
from __future__ import annotations

import argparse
import re

from mimi_v2.tracing import waterfall
from uuid import uuid4

TEST_PREFIX = "gradio-test-"
_SAFE = re.compile(r"[^a-z0-9-]")


def make_user_id(label: str | None = None) -> str:
    """Return a session id that always carries TEST_PREFIX."""
    suffix = _SAFE.sub("", (label or "").strip().lower())[:40] or uuid4().hex[:12]
    return TEST_PREFIX + suffix


def _require_test_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.startswith(TEST_PREFIX) or len(user_id) == len(TEST_PREFIX):
        raise ValueError("refusing to touch a non-test session")
    return user_id


async def ensure_pools() -> None:
    """Create the shared Postgres/Redis pools on the running event loop (idempotent)."""
    from utils.database import start_pooling
    await start_pooling()


async def delete_session(store, user_id: str) -> None:
    """Delete only this test session's keys (never a real PSID)."""
    _require_test_id(user_id)
    redis = await store._redis()
    await redis.delete(f"chat:{user_id}:history", f"chat:{user_id}:topic", f"chat:{user_id}:takeover",
                       f"mimi:v2:session:{user_id}")


async def send(engine, user_id: str, text: str) -> dict:
    _require_test_id(user_id)
    return await engine.respond(user_id, text)


def display_messages(result: dict, previous: list | None = None) -> list[dict]:
    """Chat history plus this turn's extra parts (the one-time greeting is delivered but kept out of history).

    `previous` is the chat as currently shown; a greeting shown earlier is put back at its old position.
    """
    from mimi_v2.prompts import GREETING_MESSAGE
    shown = [{"role": m["role"], "content": m["content"]} for m in result.get("messages", [])]
    parts = result.get("response_parts") or []
    if shown and shown[-1]["role"] == "assistant" and shown[-1]["content"] in parts:
        at = parts.index(shown[-1]["content"])
        shown[-1:] = [{"role": "assistant", "content": part} for part in parts[:at]] + shown[-1:] +                      [{"role": "assistant", "content": part} for part in parts[at + 1:]]
    greeting = {"role": "assistant", "content": GREETING_MESSAGE}
    if greeting not in shown:
        old = [{"role": m.get("role"), "content": m.get("content")} for m in previous or []]
        if greeting in old:
            shown.insert(min(old.index(greeting), len(shown)), greeting)
    return shown


async def clear_handoff(engine, user_id: str) -> dict:
    """Same effect as the admin #resume command: handoff removed, bot answers again."""
    _require_test_id(user_id)
    await engine.store.resume(user_id)
    return await engine.store.load(user_id)


async def accept_handoff(engine, user_id: str) -> dict:
    """Same effect as the admin #accept command."""
    _require_test_id(user_id)
    await engine.store.acknowledge(user_id)
    return await engine.store.load(user_id)


def build_app():
    import gradio as gr
    from dotenv import load_dotenv
    load_dotenv()
    theme = gr.themes.Default(font=["system-ui", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"])
    with gr.Blocks(title="Mimi — chat thử với hệ thống thật", analytics_enabled=False, theme=theme) as app:
        gr.Markdown(
            "# Mimi — chat thử với hệ thống thật\n"
            "**ĐANG KẾT NỐI PRODUCTION:** Upstash Redis, Supabase, model Shineshop/OpenAI và email khẩn. "
            f"Mọi phiên dùng ID bắt đầu bằng `{TEST_PREFIX}`. Khi Mimi chuyển người hỗ trợ, "
            "email và log khẩn thật sẽ được gửi.")
        state = gr.State(None)  # session id
        identity = gr.Textbox(label="ID phiên (tiền tố cố định)", interactive=False)
        label = gr.Textbox(label="Tên phiên (tùy chọn, để quay lại phiên cũ)", placeholder="vd: toan-lop8")
        with gr.Row():
            with gr.Column(scale=2):
                # Messenger shows plain text, so preview plain text: no Markdown, no LaTeX rendering.
                chat = gr.Chatbot(label="Cuộc trò chuyện", type="messages", height=420,
                                  render_markdown=False, latex_delimiters=[], sanitize_html=True)
                text = gr.Textbox(label="Tin nhắn", placeholder="Nhập tin nhắn, giống như nhắn trên Messenger")
                send_btn = gr.Button("Gửi tin nhắn", variant="primary")
            with gr.Column():
                status = gr.Markdown("Trạng thái handoff: —")
                clear = gr.Button("Gỡ handoff, tiếp tục chat", variant="stop")
                accept = gr.Button("Người hỗ trợ tiếp nhận (#accept)")
                new_session = gr.Button("Phiên mới")
                delete = gr.Button("Xóa dữ liệu phiên này")
                diagnostics = gr.JSON(label="Assessment, nguồn, handoff, thời gian và lỗi")
        gr.Markdown("### Phân tích thời gian của lượt chat gần nhất")
        trace_md = gr.Markdown("_Gửi một tin nhắn để xem từng bước tốn bao lâu._")

        def _status(session):
            handoff = (session or {}).get("handoff_status", "none")
            return {"none": "Trạng thái handoff: **không** (Mimi đang trả lời)",
                    "pending": "Trạng thái handoff: **đang chờ người hỗ trợ**. Bấm *Gỡ handoff* để tiếp tục chat.",
                    "accepted": "Trạng thái handoff: **đã tiếp nhận** (Mimi im lặng). Bấm *Gỡ handoff* để tiếp tục chat."
                    }.get(handoff, f"Trạng thái handoff: {handoff}")

        async def _engine():
            await ensure_pools()
            from mimi_v2.production import get_production_engine
            return get_production_engine()

        def _id(user_id, name=None):
            return user_id or make_user_id(name)

        async def submit(value, user_id, name, shown):
            user_id = _id(user_id, name)
            if not value or not value.strip():
                return gr.skip(), {"error": "Hãy nhập tin nhắn."}, value, user_id, user_id, gr.skip(), gr.skip()
            try:
                result = await send(await _engine(), user_id, value)
            except Exception as exc:
                return gr.skip(), {"error": type(exc).__name__, "detail": "Không xử lý được tin nhắn."}, value, user_id, user_id, gr.skip(), gr.skip()
            return (display_messages(result, shown), result, "", user_id, user_id,
                    _status({"handoff_status": result.get("handoff_status", "none")}), waterfall(result.get("trace", {})))

        async def control(action, user_id):
            if not user_id:
                return gr.skip(), {"error": "Chưa có phiên."}, user_id, gr.skip()
            try:
                engine = await _engine()
                session = await (clear_handoff if action == "clear" else accept_handoff)(engine, user_id)
            except Exception as exc:
                return gr.skip(), {"error": type(exc).__name__, "detail": "Cần trạng thái pending để tiếp nhận."}, user_id, gr.skip()
            return session.get("messages", []), session, user_id, _status(session)

        async def clear_click(user_id):
            return await control("clear", user_id)

        async def accept_click(user_id):
            return await control("accept", user_id)

        async def delete_click(user_id):
            if user_id:
                await delete_session((await _engine()).store, user_id)
            return [], {"deleted": user_id}, None, "", _status(None)

        def new_click():
            return [], {}, None, "", _status(None)

        send_btn.click(submit, [text, state, label, chat], [chat, diagnostics, text, state, identity, status, trace_md], concurrency_limit=1)
        text.submit(submit, [text, state, label, chat], [chat, diagnostics, text, state, identity, status, trace_md], concurrency_limit=1)
        clear.click(clear_click, [state], [chat, diagnostics, state, status], concurrency_limit=1)
        accept.click(accept_click, [state], [chat, diagnostics, state, status], concurrency_limit=1)
        delete.click(delete_click, [state], [chat, diagnostics, state, identity, status], concurrency_limit=1)
        new_session.click(new_click, None, [chat, diagnostics, state, identity, status])
    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio chat connected to the production engine and storage")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    build_app().queue().launch(server_name="127.0.0.1", server_port=args.port, share=False, show_error=False)


if __name__ == "__main__":
    main()
