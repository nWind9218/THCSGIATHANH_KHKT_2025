from typing import Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import os
from dotenv import load_dotenv
import httpx
import logging
import asyncio
from contextlib import asynccontextmanager
from utils.database import start_pooling, close_db_pools, get_redis_client
from workflow import workflow2
from workflow_demo import workflow_demo
from langsmith import Client, traceable
# Load environment variables
load_dotenv()


VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
GRAPH_API_URL = "https://graph.facebook.com/v21.0/me/messages"
REDIS_URL = os.getenv("REDIS_URL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
active_task: Dict[str, asyncio.Task] = {}
client = Client()

@asynccontextmanager
async def lifespan(app : FastAPI):
    logger.info("✅ START BUILDING...")
    await start_pooling()
    yield
    logger.info("🛑 SHUTDOWN")
    client.flush()
    await close_db_pools()

app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message":"Hello World"}
@app.post("/chatme")
async def chatMe(request: Request):
    try:
        body = await request.json()  # Thêm await
        message = body.get('message')
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Không gọi workflow_demo() như function, nó đã là workflow object
        final_response = await workflow_demo().ainvoke({
            'messages': message,
            'rag_results': '',
            'llm_response': '',
            'final_response': '',
            'error': ''
        }, config={"configurable": {"thread_id": "12321312321"}})
        
        return JSONResponse(content={
            "status": "success",
            "response": final_response['final_response']
        })
    except Exception as e:
        logger.error(f"❌ Error at /chatme: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/chat")
async def chat(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("✅ Webhook verified successfully")
        return PlainTextResponse(content=challenge)
    else:
        logger.warning(f"❌ Verification failed!")
        logger.warning(f"❌ Expected token: '{VERIFY_TOKEN}' (type: {type(VERIFY_TOKEN)})")
        logger.warning(f"❌ Received token: '{token}' (type: {type(token)})")
        raise HTTPException(status_code=403, detail="Verification failed")
    
@app.post("/chat")
async def receive_message(request: Request):    
    """Receive Messages from Webhook, then send messages to Agent to handle messages in background task

    Args:
        request (Request): Information necessary packed in each Request

    Raises:
        HTTPException: 500 - Internal Server Error

    Returns:
        "status": To Inform facebook that we received successfully
    """
    try:
        body = await request.json()
        logger. info(f"📩 Received: {body}")
        if body.get("object") == "page":
            for entry in body.get("entry", []):
                for event in entry.get("messaging", []):
                    sender_id = event.get("sender", {}).get("id")

                    if event.get("message"):
                        msg = event["message"].get("text")
                        if msg:
                            # logger.info(f"💬 Message from {sender_id}: {msg}")
                            asyncio.create_task(reset_task(1, sender_id, message=msg))
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
async def send_message(recipient_id: str, message_text: str):
    """Provide messages to AI workflow, and sending messages back to sender_id"""
    try:
        logger.info(f"✅ ĐÃ GỬI TIN NHẮN THÀNH CÔNG TỚI {recipient_id}")
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message_text}
        }
        
        headers = {"Content-Type": "application/json"}
        params = {"access_token": PAGE_ACCESS_TOKEN}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GRAPH_API_URL,
                json=payload,
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Message sent to {recipient_id}")
            else:
                logger.error(f"❌ Failed: {response.text}")
                
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
async def reset_task(seconds: int, sender_id: str, message: str):
    if sender_id in active_task:
        old_task = active_task[sender_id]
        old_task.cancel()
    logger.info(f"✅ CREATE TASK MỚI CHO {sender_id}")
    await stacking_messages(sender_id, message)
    new_task = asyncio.create_task(
        message_postback(seconds, sender_id)
    )
    active_task[sender_id] = new_task
@traceable(client=client)
async def message_postback(seconds: int, sender_id: str):
    try:
        logger.info(f"BẮT ĐẦU THIẾT LẬP HÀNG CHỜ CHO: {seconds} giây")
        await asyncio.sleep(seconds)
        logger.info(f"HẾT THỜI GIAN CHỜ, BÂY GIỜ BOT SẼ TIẾP TỤC CÔNG VIỆC!")
        messages = await clear_tasks(sender_id=sender_id)
        final_response = await workflow2().ainvoke({
            "conversation": {
                "user_id": sender_id,
                "messages": [
                    {
                        "content": messages,
                        "role": "user"
                    }
                ],
                "is_new_user": True  # Fix: thiếu field này
            }
        }, config={"configurable": {"thread_id": sender_id}})
        
        return await send_message(sender_id, final_response['response']['output'])
    except asyncio.CancelledError:
        logger.warning(f"Đã có tin nhắn mới từ {sender_id}!")
        raise
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
    finally:
        if sender_id in active_task:
            del active_task[sender_id]
            logger.info(f"{sender_id} đã được xóa bỏ khỏi list chờ")
async def clear_tasks(sender_id: str):
    try:
        db_client = await get_redis_client()
        logger.info(f"GET TOÀN BỘ TIN NHẮN CỦA {sender_id}")
        messages = ", ".join( await db_client.lrange(f"waiting:{sender_id}", 0, -1) )
        await db_client.delete(f"waiting:{sender_id}")
        logger.info(f"✅ XÓA TOÀN BỘ TIN NHẮN WAITING CỦA {sender_id} THÀNH CÔNG")
        return messages
    except Exception as e:
        logger.error("❌ Error at Clearing Tasks:", str(e))
async def stacking_messages(sender_id: str, message: str):
    """Stack tin nhắn vào Redis"""
    try:
        db_client = await get_redis_client()
        await db_client.rpush(f"waiting:{sender_id}", message)
        logger.info(f"✅ Stacked message for {sender_id}")
    except Exception as e:
        logger.error(f"❌ Error at Stacking Messages: {str(e)}")  

async def block_ai_workflow():
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)