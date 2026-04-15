"""Redis client abstraction for managing short-term memory (chat history, topics, takeover flags)."""

import asyncio
import json
import logging
from typing import Optional

from utils.database import get_redis_client

logger = logging.getLogger(__name__)

# Redis key patterns
KEY_HISTORY = "chat:{user_id}:history"
KEY_TOPIC = "chat:{user_id}:topic"
KEY_TAKEOVER = "chat:{user_id}:takeover"
ADMIN_ALERTS = "admin:alerts"

# TTL settings (in seconds)
TTL_HISTORY = 7 * 24 * 60 * 60  # 7 days
TTL_TOPIC = 7 * 24 * 60 * 60  # 7 days
TTL_TAKEOVER = 30 * 24 * 60 * 60  # 30 days


async def load_history(user_id: str) -> list[dict]:
    """Load last 20 chat messages from Redis."""
    redis = await get_redis_client()
    raw = await redis.get(KEY_HISTORY.format(user_id=user_id))
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            return payload[-20:]
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse history for {user_id}")
        return []
    return []


async def save_history(user_id: str, messages: list[dict]) -> None:
    """Save last 20 chat messages to Redis."""
    redis = await get_redis_client()
    await redis.set(
        KEY_HISTORY.format(user_id=user_id),
        json.dumps(messages[-20:], ensure_ascii=False),
        ex=TTL_HISTORY,
    )


async def load_topic(user_id: str) -> str:
    """Load current topic from Redis."""
    redis = await get_redis_client()
    return await redis.get(KEY_TOPIC.format(user_id=user_id)) or ""


async def save_topic(user_id: str, topic: str) -> None:
    """Save current topic to Redis."""
    redis = await get_redis_client()
    await redis.set(
        KEY_TOPIC.format(user_id=user_id),
        topic or "",
        ex=TTL_TOPIC,
    )


async def load_takeover_flag(user_id: str) -> bool:
    """Check if human has taken over this user's conversation."""
    redis = await get_redis_client()
    value = await redis.get(KEY_TAKEOVER.format(user_id=user_id))
    return str(value).lower() == "true"


async def set_takeover_flag(user_id: str, value: bool) -> None:
    """Set human takeover flag for this user."""
    redis = await get_redis_client()
    await redis.set(
        KEY_TAKEOVER.format(user_id=user_id),
        "true" if value else "false",
        ex=TTL_TAKEOVER,
    )


async def publish_admin_alert(user_id: str, alert_data: dict) -> None:
    """Publish emergency alert to admin channel via Redis Pub/Sub."""
    redis = await get_redis_client()
    alert_data["user_id"] = user_id
    try:
        await redis.publish(ADMIN_ALERTS, json.dumps(alert_data, ensure_ascii=False))
        logger.info(f"Published admin alert for {user_id}")
    except Exception as e:
        logger.error(f"Failed to publish admin alert: {e}")
