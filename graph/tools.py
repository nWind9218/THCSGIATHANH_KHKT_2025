from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from utils.database import get_redis_client
from utils.embeddings import embed_text, vector_literal
from memory import (
    load_history,
    save_history,
    load_topic,
    save_topic,
    load_takeover_flag,
    set_takeover_flag,
    publish_admin_alert,
    search_psychology_kb,
    search_user_memory_kb,
    upsert_user_memory_chunk,
    log_emergency,
    send_emergency_email,
    notify_human_admin,
    update_user_longterm_style,
    infer_ocean_increment,
)

load_dotenv()
logger = logging.getLogger(__name__)


_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


def latest_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""
