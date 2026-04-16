"""WebSocket endpoint for teacher/admin supervision and takeover."""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis

from utils.database import get_redis_client
from api.cors_config import get_cors_settings, is_origin_allowed
from memory import set_takeover_flag, publish_admin_alert

logger = logging.getLogger(__name__)
router = APIRouter()

# Keep track of active teacher connections
active_teachers: dict[str, WebSocket] = {}  # teacher_id -> WebSocket
teacher_subscriptions: dict[str, set[str]] = {}  # teacher_id -> set of subscribed user_ids
cors_settings = get_cors_settings()


@router.websocket("/admin/{teacher_id}")
async def websocket_admin_endpoint(websocket: WebSocket, teacher_id: str):
    """
    WebSocket endpoint for teacher supervision and emergency response.
    
    Workflow:
    1. Teacher connects via /ws/admin/{teacher_id}
    2. Accept connection and subscribe to admin alerts
    3. Send incoming alerts to teacher
    4. Process teacher commands:
       - takeover:{user_id} - take over a student's conversation
       - unsubscribe:{user_id} - stop observing a student
       - query:{user_id} - get student info
    """
    try:
        origin = websocket.headers.get("origin")
        if not is_origin_allowed(origin, cors_settings):
            logger.warning(f"❌ Blocked WS origin for admin endpoint: {origin}")
            await websocket.close(code=1008, reason="Origin not allowed")
            return

        await websocket.accept()
        active_teachers[teacher_id] = websocket
        teacher_subscriptions[teacher_id] = set()
        logger.info(f"✅ Teacher connected: {teacher_id}")

        # Send connection acknowledgment
        await websocket.send_json({
            "type": "connection_acknowledged",
            "teacher_id": teacher_id,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        })

        # Subscribe to admin:alerts via Redis Pub/Sub
        redis = await get_redis_client()
        pubsub = redis.pubsub()
        await pubsub.subscribe("admin:alerts")
        logger.debug(f"Teacher {teacher_id} subscribed to admin:alerts")

        # Create tasks for handling Redis messages and WebSocket messages
        async def handle_redis_messages():
            """Listen for messages from Redis Pub/Sub."""
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            alert_data = json.loads(message["data"])
                            # Filter alerts based on teacher's subscriptions
                            user_id = alert_data.get("user_id")
                            if not teacher_subscriptions[teacher_id] or user_id in teacher_subscriptions[teacher_id]:
                                await websocket.send_json({
                                    "type": "alert",
                                    "alert_type": alert_data.get("type"),
                                    **alert_data,
                                })
                                logger.debug(f"Sent alert to {teacher_id}: {alert_data.get('type')}")
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in Redis alert")
            except asyncio.CancelledError:
                await pubsub.unsubscribe("admin:alerts")
                await pubsub.close()
                raise

        async def handle_websocket_messages():
            """Listen for commands from teacher."""
            try:
                while True:
                    data = await websocket.receive_text()
                    command_data = json.loads(data)
                    command = command_data.get("command", "").strip()
                    user_id = command_data.get("user_id", "").strip()

                    if command == "subscribe":
                        # Start observing a student
                        teacher_subscriptions[teacher_id].add(user_id)
                        await websocket.send_json({
                            "type": "command_result",
                            "command": "subscribe",
                            "status": "success",
                            "user_id": user_id,
                            "message": f"Now observing {user_id}",
                        })
                        logger.info(f"Teacher {teacher_id} subscribed to student {user_id}")

                    elif command == "unsubscribe":
                        # Stop observing a student
                        teacher_subscriptions[teacher_id].discard(user_id)
                        await websocket.send_json({
                            "type": "command_result",
                            "command": "unsubscribe",
                            "status": "success",
                            "user_id": user_id,
                            "message": f"Stopped observing {user_id}",
                        })
                        logger.info(f"Teacher {teacher_id} unsubscribed from student {user_id}")

                    elif command == "takeover":
                        # Take over a student's conversation
                        await set_takeover_flag(user_id, True)
                        await publish_admin_alert(user_id, {
                            "type": "takeover_initiated",
                            "user_id": user_id,
                            "teacher_id": teacher_id,
                            "message": "Your school counselor is now taking over the conversation.",
                        })
                        await websocket.send_json({
                            "type": "command_result",
                            "command": "takeover",
                            "status": "success",
                            "user_id": user_id,
                            "message": f"You have taken over conversation with {user_id}",
                        })
                        logger.warning(f"Teacher {teacher_id} initiated takeover for {user_id}")

                    elif command == "release":
                        # Release a student's conversation
                        await set_takeover_flag(user_id, False)
                        await websocket.send_json({
                            "type": "command_result",
                            "command": "release",
                            "status": "success",
                            "user_id": user_id,
                            "message": f"Released control of {user_id}, bot will resume",
                        })
                        logger.info(f"Teacher {teacher_id} released control of {user_id}")

                    elif command == "send_message":
                        # Send a message to student (when takeover is active)
                        message_text = command_data.get("message", "").strip()
                        if message_text:
                            await publish_admin_alert(user_id, {
                                "type": "teacher_message",
                                "user_id": user_id,
                                "teacher_id": teacher_id,
                                "message": message_text,
                            })
                            logger.info(f"Teacher {teacher_id} sent message to {user_id}")

                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Unknown command: {command}",
                        })

            except asyncio.CancelledError:
                raise

        # Run both tasks concurrently
        redis_task = asyncio.create_task(handle_redis_messages())
        websocket_task = asyncio.create_task(handle_websocket_messages())

        try:
            await asyncio.gather(redis_task, websocket_task)
        except asyncio.CancelledError:
            redis_task.cancel()
            websocket_task.cancel()
            raise

    except WebSocketDisconnect:
        logger.info(f"🔴 Teacher disconnected: {teacher_id}")
        if teacher_id in active_teachers:
            del active_teachers[teacher_id]
        if teacher_id in teacher_subscriptions:
            del teacher_subscriptions[teacher_id]

    except Exception as e:
        logger.error(f"Admin WebSocket error for {teacher_id}: {e}")
        if teacher_id in active_teachers:
            del active_teachers[teacher_id]
        if teacher_id in teacher_subscriptions:
            del teacher_subscriptions[teacher_id]
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass


@router.get("/active-teachers", tags=["Monitoring"])
async def get_active_teachers():
    """Get list of currently active teacher connections."""
    return {
        "active_teachers": list(active_teachers.keys()),
        "subscriptions": {
            teacher_id: list(subs) for teacher_id, subs in teacher_subscriptions.items()
        },
        "count": len(active_teachers),
    }
