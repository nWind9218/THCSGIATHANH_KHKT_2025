"""WebSocket endpoint for student chat connections."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from langchain_core.messages import HumanMessage

from graph.workflow import build_counseling_graph
from graph.state import CounselingState
from api.cors_config import get_cors_settings, is_origin_allowed
from memory import load_history, load_topic, publish_admin_alert

logger = logging.getLogger(__name__)
router = APIRouter()

# Keep track of active student connections
active_connections: dict[str, WebSocket] = {}
cors_settings = get_cors_settings()


@router.websocket("/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for student chat.
    
    Workflow:
    1. Student connects via /ws/chat/{user_id}
    2. Load chat history from Redis
    3. Accept connection and send initial context
    4. Process incoming messages through LangGraph
    5. Send bot response back
    6. Publish to admin observers
    """
    try:
        origin = websocket.headers.get("origin")
        if not is_origin_allowed(origin, cors_settings):
            logger.warning(f"❌ Blocked WS origin for student endpoint: {origin}")
            await websocket.close(code=1008, reason="Origin not allowed")
            return

        await websocket.accept()
        active_connections[user_id] = websocket
        logger.info(f"✅ Student connected: {user_id}")

        # Load initial context
        history = await load_history(user_id)
        current_topic = await load_topic(user_id)

        # Send connection acknowledgment with context
        await websocket.send_json({
            "type": "connection_acknowledged",
            "user_id": user_id,
            "history": history[-5:],  # Last 5 messages for context
            "current_topic": current_topic,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        })
        logger.debug(f"Sent initial context to {user_id}")

        # Message processing loop
        while True:
            # Receive student message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_input = message_data.get("message", "").strip()

            if not user_input:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty message not allowed",
                })
                continue

            # Add user message to history
            messages = history + [{"role": "user", "content": user_input}]

            # Build and run counseling graph
            try:
                graph = build_counseling_graph()
                state: CounselingState = {
                    "messages": messages,
                    "user_id": user_id,
                    "current_topic": current_topic,
                    "intent_category": "",
                    "info_gap_status": "",
                    "kb_guidelines": "",
                    "user_memory": "",
                    "reasoning_scratchpad": [],
                    "is_emergency": False,
                    "human_takeover": False,
                    "response_text": "",
                    "error": "",
                }

                result = await graph.ainvoke(state)
                bot_response = result.get("response_text", "")
                is_emergency = result.get("is_emergency", False)
                human_takeover = result.get("human_takeover", False)

                # Update history
                history = result.get("messages", messages)
                current_topic = result.get("current_topic", current_topic)

                # Send bot response
                await websocket.send_json({
                    "type": "bot_response",
                    "user_id": user_id,
                    "response": bot_response,
                    "intent": result.get("intent_category", ""),
                    "is_emergency": is_emergency,
                    "human_takeover": human_takeover,
                    "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                })
                logger.debug(f"Sent bot response to {user_id}: intent={result.get('intent_category')}")

                # Publish to admin observers for real-time monitoring
                await publish_admin_alert(user_id, {
                    "type": "chat_message",
                    "user_id": user_id,
                    "user_input": user_input,
                    "bot_response": bot_response,
                    "intent": result.get("intent_category", ""),
                    "is_emergency": is_emergency,
                    "human_takeover": human_takeover,
                })

                # If emergency detected, notify all watching teachers
                if is_emergency or human_takeover:
                    await publish_admin_alert(user_id, {
                        "type": "emergency",
                        "user_id": user_id,
                        "requires_takeover": True,
                    })

            except Exception as e:
                logger.error(f"Graph execution error for {user_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Internal processing error. Please try again.",
                    "error_code": "GRAPH_ERROR",
                })

    except WebSocketDisconnect:
        logger.info(f"🔴 Student disconnected: {user_id}")
        if user_id in active_connections:
            del active_connections[user_id]

        # Notify teachers that this student is offline
        await publish_admin_alert(user_id, {
            "type": "student_disconnected",
            "user_id": user_id,
        })

    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {e}")
        if user_id in active_connections:
            del active_connections[user_id]
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass


@router.get("/active-students", tags=["Monitoring"])
async def get_active_students():
    """Get list of currently active student connections."""
    return {
        "active_students": list(active_connections.keys()),
        "count": len(active_connections),
    }
