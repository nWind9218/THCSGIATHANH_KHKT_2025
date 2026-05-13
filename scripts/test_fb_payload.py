import time
import hmac
import hashlib
import json
import asyncio
import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Load env vars before importing app to ensure they are available
load_dotenv()

from api.main import app

async def mock_send_messenger_message(psid: str, text: str) -> None:
    print(f"\n[MOCK SEND] To: {psid}")
    print(f"[MOCK SEND] Message: {text}")

def test_fb_webhook(client, payload_type="text"):
    secret = os.getenv("FACEBOOK_APP_SECRET", "7d333a1734d2fb8ee4df4a6467310c0f")
    
    if payload_type == "text":
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "PAGE_ID",
                    "time": 123456789,
                    "messaging": [
                        {
                            "sender": {"id": "test_user_psid_123"},
                            "recipient": {"id": "PAGE_ID"},
                            "timestamp": 123456789,
                            "message": {"text": "Chào bạn, mình cần giúp đỡ về tâm lý."}
                        }
                    ]
                }
            ]
        }
    else: # postback
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "PAGE_ID",
                    "time": 123456789,
                    "messaging": [
                        {
                            "sender": {"id": "test_user_psid_123"},
                            "recipient": {"id": "PAGE_ID"},
                            "timestamp": 123456789,
                            "postback": {"payload": "START_COUNSELING", "title": "Bắt đầu tư vấn"}
                        }
                    ]
                }
            ]
        }

    body = json.dumps(payload).encode("utf-8")
    signature = hmac.new(
        secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256
    ).hexdigest()
    
    headers = {"x-hub-signature-256": f"sha256={signature}", "Content-Type": "application/json"}
    
    print(f"\n--- Testing Facebook Webhook with {payload_type} payload ---")
    
    # Mocking _send_messenger_message in api.facebook_webhook
    with patch("api.facebook_webhook._send_messenger_message", side_effect=mock_send_messenger_message):
        # Using the client as a context manager at a higher level to keep it alive
        try:
            response = client.post("/webhook/facebook", content=body, headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
            # The background task is now running...
        except Exception as e:
            print(f"Exception occurred during request: {e}")

if __name__ == "__main__":
    with TestClient(app) as client:
        # Test text message
        test_fb_webhook(client, payload_type="text")
        # Wait a bit for the first task
        time.sleep(8) 
        
        # Test postback
        test_fb_webhook(client, payload_type="postback")
        time.sleep(8)

        # Test emergency message
        print("\n--- Testing Emergency Message ---")
        payload_emergency = {
            "object": "page",
            "entry": [
                {
                    "id": "PAGE_ID",
                    "messaging": [
                        {
                            "sender": {"id": "test_user_psid_emergency"},
                            "message": {"text": "Em muốn tự tử, em mệt mỏi quá rồi."}
                        }
                    ]
                }
            ]
        }
        
        secret = os.getenv("FACEBOOK_APP_SECRET", "7d333a1734d2fb8ee4df4a6467310c0f")
        body = json.dumps(payload_emergency).encode("utf-8")
        signature = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256).hexdigest()
        headers = {"x-hub-signature-256": f"sha256={signature}", "Content-Type": "application/json"}
        
        with patch("api.facebook_webhook._send_messenger_message", side_effect=mock_send_messenger_message):
            response = client.post("/webhook/facebook", content=body, headers=headers)
            print(f"Status Code: {response.status_code}")
            time.sleep(10) # Give emergency flow more time (emails, etc)
