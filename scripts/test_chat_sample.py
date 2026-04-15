from __future__ import annotations

import asyncio
import json

from utils.database import close_db_pools, start_pooling
from workflow import workflow


async def main() -> None:
    await start_pooling()
    graph = workflow()

    user_id = "local_integration_test"
    config = {"configurable": {"thread_id": user_id}}

    sample_inputs = [
        "Hom nay em met va hoi lo ve viec hoc.",
        "Em hay bi ap luc vi so diem kem.",
    ]

    outputs = []
    try:
        for message in sample_inputs:
            result = await graph.ainvoke(
                {
                    "user_id": user_id,
                    "messages": [{"role": "user", "content": message}],
                },
                config=config,
            )
            outputs.append(
                {
                    "input": message,
                    "intent_category": result.get("intent_category"),
                    "info_gap_status": result.get("info_gap_status"),
                    "is_emergency": result.get("is_emergency"),
                    "human_takeover": result.get("human_takeover"),
                    "response_text": result.get("response_text"),
                }
            )

        print(json.dumps({"chat_sample": outputs}, ensure_ascii=False, indent=2))
    finally:
        await close_db_pools()


if __name__ == "__main__":
    asyncio.run(main())
