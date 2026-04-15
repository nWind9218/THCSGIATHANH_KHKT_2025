from __future__ import annotations

import asyncio
import logging

from utils.database import close_db_pools, start_pooling
from workflow import workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_local_chat() -> None:
    await start_pooling()
    graph = workflow()

    print("Mimi local MVP chat")
    print("Type /exit to quit")

    user_id = input("User ID (default: local_student): ").strip() or "local_student"
    config = {"configurable": {"thread_id": user_id}}

    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                continue
            if user_text.lower() in {"/exit", "exit", "quit"}:
                break

            result = await graph.ainvoke(
                {
                    "user_id": user_id,
                    "messages": [{"role": "user", "content": user_text}],
                },
                config=config,
            )

            bot_text = result.get("response_text", "")
            if bot_text:
                print(f"Mimi: {bot_text}")
            else:
                print("Mimi: (no response)")

            if result.get("human_takeover"):
                print("[Notice] Conversation flagged for human takeover.")
    finally:
        await close_db_pools()


if __name__ == "__main__":
    asyncio.run(run_local_chat())
