"""Memory management layer - separates concerns for Redis and Supabase."""

from .redis_client import (
    load_history,
    save_history,
    load_topic,
    save_topic,
    load_takeover_flag,
    set_takeover_flag,
    publish_admin_alert,
)

from .supabase_client import (
    search_psychology_kb,
    search_student_knowledge_kb,
    search_user_memory_kb,
    upsert_user_memory_chunk,
    log_emergency,
    send_emergency_email,
    notify_human_admin,
    update_user_longterm_style,
    infer_ocean_increment,
)

__all__ = [
    # Redis operations
    "load_history",
    "save_history",
    "load_topic",
    "save_topic",
    "load_takeover_flag",
    "set_takeover_flag",
    "publish_admin_alert",
    # Supabase operations
    "search_psychology_kb",
    "search_student_knowledge_kb",
    "search_user_memory_kb",
    "upsert_user_memory_chunk",
    "log_emergency",
    "send_emergency_email",
    "notify_human_admin",
    "update_user_longterm_style",
    "infer_ocean_increment",
]
