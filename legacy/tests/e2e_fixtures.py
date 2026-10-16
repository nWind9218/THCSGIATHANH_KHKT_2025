"""
Test fixtures and configuration for E2E integration tests
"""

import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI
from httpx import AsyncClient
import websockets
import json


# Test configuration constants
class TestConfig:
    """Configuration for E2E tests"""
    
    # Server configuration
    HOST = "127.0.0.1"
    PORT = 8000
    BASE_URL = f"http://{HOST}:{PORT}"
    WS_URL = f"ws://{HOST}:{PORT}"
    
    # Test timeouts
    HTTP_TIMEOUT = 5
    WS_TIMEOUT = 10
    MESSAGE_TIMEOUT = 5
    
    # Test data
    TEST_USER_IDS = [
        "test-user-001",
        "test-user-002", 
        "test-user-003",
    ]
    
    TEST_MESSAGES = [
        "I need help with my math homework",
        "Can you explain photosynthesis?",
        "What are the main themes in this book?",
        "How do I solve quadratic equations?",
    ]
    
    # Expected response patterns
    EXPECTED_CONNECTION_ACK_TYPE = "connection_acknowledged"
    EXPECTED_RESPONSE_TYPE = "response"
    EXPECTED_ERROR_TYPE = "error"


class E2ETestFixtures:
    """Reusable test fixtures"""
    
    @staticmethod
    async def create_http_client() -> AsyncClient:
        """Create HTTP client for testing"""
        return AsyncClient(
            base_url=TestConfig.BASE_URL,
            timeout=TestConfig.HTTP_TIMEOUT,
        )
    
    @staticmethod
    async def connect_websocket(user_id: str):
        """Connect to WebSocket endpoint"""
        url = f"{TestConfig.WS_URL}/ws/chat/{user_id}"
        return await websockets.connect(url, ping_interval=None)
    
    @staticmethod
    async def get_welcome_message(websocket):
        """Receive and parse welcome message"""
        raw = await asyncio.wait_for(
            websocket.recv(),
            timeout=TestConfig.MESSAGE_TIMEOUT
        )
        return json.loads(raw)
    
    @staticmethod
    async def send_message(websocket, message: str) -> dict:
        """Send message and get response"""
        await websocket.send(json.dumps({"message": message}))
        raw = await asyncio.wait_for(
            websocket.recv(),
            timeout=TestConfig.MESSAGE_TIMEOUT
        )
        return json.loads(raw)
    
    @staticmethod
    async def get_history(websocket) -> list:
        """Receive chat history"""
        raw = await asyncio.wait_for(
            websocket.recv(),
            timeout=TestConfig.MESSAGE_TIMEOUT
        )
        data = json.loads(raw)
        if data.get("type") == "history":
            return data.get("data", [])
        return []
    
    @staticmethod
    async def get_topic(websocket) -> str:
        """Receive current topic"""
        raw = await asyncio.wait_for(
            websocket.recv(),
            timeout=TestConfig.MESSAGE_TIMEOUT
        )
        data = json.loads(raw)
        if data.get("type") == "topic":
            return data.get("data", {}).get("topic", "")
        return ""


class MockResponse:
    """Mock response for testing"""
    
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
    
    def json(self) -> dict:
        return self._data


# Test assertions
def assert_valid_connection_message(message: dict) -> None:
    """Assert message is valid connection acknowledgment"""
    assert isinstance(message, dict), "Message must be dict"
    assert "type" in message, "Message must have 'type'"
    assert message["type"] == TestConfig.EXPECTED_CONNECTION_ACK_TYPE, \
        f"Expected connection_acknowledged, got {message['type']}"
    assert "data" in message or "message" in message, "Message must have data"


def assert_valid_response_message(message: dict) -> None:
    """Assert message is valid response"""
    assert isinstance(message, dict), "Message must be dict"
    assert "type" in message, "Message must have 'type'"
    assert message["type"] == TestConfig.EXPECTED_RESPONSE_TYPE, \
        f"Expected response, got {message['type']}"
    assert "data" in message or "message" in message, "Response must have data"


def assert_valid_error_message(message: dict) -> None:
    """Assert message is valid error"""
    assert isinstance(message, dict), "Message must be dict"
    assert "type" in message, "Message must have 'type'"
    assert message["type"] == TestConfig.EXPECTED_ERROR_TYPE, \
        f"Expected error, got {message['type']}"
    assert "data" in message or "message" in message, "Error must have data"


def assert_http_success(response) -> None:
    """Assert HTTP response is successful"""
    assert response.status_code in [200, 201, 202], \
        f"Expected 2xx status, got {response.status_code}"


def assert_http_error(response, status_code: int) -> None:
    """Assert HTTP response has specific error status"""
    assert response.status_code == status_code, \
        f"Expected {status_code}, got {response.status_code}"
