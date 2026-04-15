"""
End-to-End Integration Tests for Frontend-Backend Communication
Tests WebSocket and HTTP endpoints with realistic frontend scenarios
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import AsyncGenerator
import websockets
from httpx import AsyncClient

# Test configuration
TEST_HOST = "127.0.0.1"
TEST_PORT = 8000
TEST_BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"
TEST_WS_URL = f"ws://{TEST_HOST}:{TEST_PORT}"
TEST_TIMEOUT = 10
TEST_USER_ID = f"test-user-{int(datetime.now().timestamp() * 1000)}"


class E2ETestResults:
    """Container for E2E test results"""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors: list[dict] = []
        self.timings: dict[str, float] = {}

    def record_pass(self, test_name: str, duration: float = 0):
        self.tests_run += 1
        self.tests_passed += 1
        if duration > 0:
            self.timings[test_name] = duration
        print(f"✓ {test_name}")

    def record_fail(self, test_name: str, error: str, duration: float = 0):
        self.tests_run += 1
        self.tests_failed += 1
        self.errors.append({
            "test": test_name,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        if duration > 0:
            self.timings[test_name] = duration
        print(f"✗ {test_name}: {error}")

    def summary(self) -> dict:
        return {
            "total": self.tests_run,
            "passed": self.tests_passed,
            "failed": self.tests_failed,
            "success_rate": f"{(self.tests_passed / self.tests_run * 100):.1f}%" if self.tests_run > 0 else "N/A",
            "errors": self.errors,
            "timings": self.timings,
        }


# ==============================================================================
# HTTP Endpoint Tests
# ==============================================================================


async def test_http_health_check():
    """Test HTTP health check endpoint"""
    start = datetime.now()
    async with AsyncClient(base_url=TEST_BASE_URL, timeout=TEST_TIMEOUT) as client:
        try:
            response = await client.get("/health")
            duration = (datetime.now() - start).total_seconds()
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "status" in data
            return {"passed": True, "duration": duration, "response": data}
        except Exception as e:
            return {"passed": False, "error": str(e)}


async def test_http_metadata_endpoint():
    """Test HTTP metadata endpoint"""
    start = datetime.now()
    async with AsyncClient(base_url=TEST_BASE_URL, timeout=TEST_TIMEOUT) as client:
        try:
            response = await client.get("/")
            duration = (datetime.now() - start).total_seconds()
            assert response.status_code == 200
            data = response.json()
            return {"passed": True, "duration": duration, "response": data}
        except Exception as e:
            return {"passed": False, "error": str(e)}


# ==============================================================================
# WebSocket Tests
# ==============================================================================


async def test_websocket_connection(user_id: str = TEST_USER_ID):
    """Test WebSocket connection establishment"""
    start = datetime.now()
    try:
        url = f"{TEST_WS_URL}/ws/chat/{user_id}"
        async with websockets.connect(url, ping_interval=None) as websocket:
            duration = (datetime.now() - start).total_seconds()
            return {"passed": True, "duration": duration, "user_id": user_id}
    except Exception as e:
        return {"passed": False, "error": str(e)}


async def test_websocket_receive_welcome_message(user_id: str = TEST_USER_ID):
    """Test receiving welcome message after connection"""
    start = datetime.now()
    try:
        url = f"{TEST_WS_URL}/ws/chat/{user_id}"
        async with websockets.connect(url, ping_interval=None) as websocket:
            # Wait for welcome message
            message = await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
            data = json.loads(message)
            
            duration = (datetime.now() - start).total_seconds()
            
            # Verify message structure
            assert "type" in data, "Message missing 'type' field"
            assert data["type"] == "connection_acknowledged", f"Expected connection_acknowledged, got {data['type']}"
            
            return {"passed": True, "duration": duration, "message": data}
    except Exception as e:
        return {"passed": False, "error": str(e)}


async def test_websocket_send_and_receive(user_id: str = TEST_USER_ID):
    """Test sending a message and receiving response"""
    start = datetime.now()
    try:
        url = f"{TEST_WS_URL}/ws/chat/{user_id}"
        async with websockets.connect(url, ping_interval=None) as websocket:
            # Skip welcome message
            welcome = await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
            
            # Send test message
            test_message = {
                "message": "Hello, I need help with my homework"
            }
            await websocket.send(json.dumps(test_message))
            
            # Receive response
            response = await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
            data = json.loads(response)
            
            duration = (datetime.now() - start).total_seconds()
            
            assert "type" in data
            assert "data" in data or "message" in data
            
            return {"passed": True, "duration": duration, "response": data}
    except Exception as e:
        return {"passed": False, "error": str(e)}


async def test_websocket_multiple_messages(user_id: str = TEST_USER_ID):
    """Test sending multiple messages in sequence"""
    start = datetime.now()
    try:
        url = f"{TEST_WS_URL}/ws/chat/{user_id}"
        async with websockets.connect(url, ping_interval=None) as websocket:
            # Skip welcome
            await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
            
            messages_sent = []
            responses_received = []
            
            test_messages = [
                "First question about math",
                "Can you explain calculus?",
                "What about trigonometry?"
            ]
            
            for msg_text in test_messages:
                # Send message
                await websocket.send(json.dumps({"message": msg_text}))
                messages_sent.append(msg_text)
                
                # Receive response
                response = await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
                responses_received.append(json.loads(response))
                
                await asyncio.sleep(0.1)  # Small delay between messages
            
            duration = (datetime.now() - start).total_seconds()
            
            assert len(messages_sent) == len(test_messages)
            assert len(responses_received) == len(test_messages)
            
            return {
                "passed": True,
                "duration": duration,
                "messages_sent": len(messages_sent),
                "responses_received": len(responses_received),
            }
    except Exception as e:
        return {"passed": False, "error": str(e)}


async def test_websocket_connection_timeout():
    """Test WebSocket handling of inactive connections"""
    start = datetime.now()
    try:
        user_id = f"test-timeout-{int(datetime.now().timestamp() * 1000)}"
        url = f"{TEST_WS_URL}/ws/chat/{user_id}"
        async with websockets.connect(url, ping_interval=None) as websocket:
            # Receive welcome
            welcome = await asyncio.wait_for(websocket.recv(), timeout=TEST_TIMEOUT)
            assert welcome
            
            # Wait briefly but don't send anything
            await asyncio.sleep(2)
            
            # Connection should still be active
            assert websocket.open
            
            duration = (datetime.now() - start).total_seconds()
            return {"passed": True, "duration": duration}
    except Exception as e:
        return {"passed": False, "error": str(e)}


# ==============================================================================
# Main Test Suite
# ==============================================================================


async def run_all_tests() -> dict:
    """Run all E2E tests and return results"""
    results = E2ETestResults()
    
    print("\n" + "="*60)
    print("FRONTEND-BACKEND E2E TEST SUITE")
    print("="*60 + "\n")
    
    # HTTP Tests
    print("[HTTP ENDPOINTS]")
    
    test_result = await test_http_health_check()
    if test_result["passed"]:
        results.record_pass("HTTP Health Check", test_result.get("duration", 0))
    else:
        results.record_fail("HTTP Health Check", test_result["error"])
    
    test_result = await test_http_metadata_endpoint()
    if test_result["passed"]:
        results.record_pass("HTTP Metadata Endpoint", test_result.get("duration", 0))
    else:
        results.record_fail("HTTP Metadata Endpoint", test_result["error"])
    
    print("\n[WEBSOCKET ENDPOINTS]")
    
    # WebSocket Tests
    test_result = await test_websocket_connection()
    if test_result["passed"]:
        results.record_pass("WebSocket Connection", test_result.get("duration", 0))
    else:
        results.record_fail("WebSocket Connection", test_result["error"])
    
    test_result = await test_websocket_receive_welcome_message()
    if test_result["passed"]:
        results.record_pass("Receive Welcome Message", test_result.get("duration", 0))
    else:
        results.record_fail("Receive Welcome Message", test_result["error"])
    
    test_result = await test_websocket_send_and_receive()
    if test_result["passed"]:
        results.record_pass("Send and Receive Message", test_result.get("duration", 0))
    else:
        results.record_fail("Send and Receive Message", test_result["error"])
    
    test_result = await test_websocket_multiple_messages()
    if test_result["passed"]:
        results.record_pass("Multiple Messages", test_result.get("duration", 0))
    else:
        results.record_fail("Multiple Messages", test_result["error"])
    
    test_result = await test_websocket_connection_timeout()
    if test_result["passed"]:
        results.record_pass("Connection Timeout Handling", test_result.get("duration", 0))
    else:
        results.record_fail("Connection Timeout Handling", test_result["error"])
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    summary = results.summary()
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']}")
    
    if summary["timings"]:
        print("\n[TIMING RESULTS]")
        for test_name, duration in sorted(summary["timings"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {test_name}: {duration*1000:.2f}ms")
    
    if summary["errors"]:
        print("\n[ERRORS]")
        for error in summary["errors"]:
            print(f"  ✗ {error['test']}: {error['error']}")
    
    print("="*60 + "\n")
    
    return summary


if __name__ == "__main__":
    print("\n⚠️  Make sure the backend server is running on port 8000")
    print("Start it with: uvicorn api.main:app --host 0.0.0.0 --port 8000\n")
    
    # Run tests
    summary = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    exit(0 if summary["failed"] == 0 else 1)
