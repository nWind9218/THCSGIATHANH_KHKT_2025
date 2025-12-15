import os
from dotenv import load_dotenv
import httpx
import asyncio

load_dotenv()

PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")

async def test_token():
    """Test if Page Access Token is valid"""
    url = f"https://graph.facebook.com/v21.0/me"
    params = {"access_token": PAGE_ACCESS_TOKEN}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Token hợp lệ!")
            data = response.json()
            print(f"Page Name: {data.get('name')}")
            print(f"Page ID: {data.get('id')}")
        else:
            print("❌ Token không hợp lệ hoặc đã hết hạn!")

if __name__ == "__main__":
    asyncio.run(test_token())