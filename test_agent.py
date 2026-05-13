import asyncio
import logging
from utils.database import start_pooling, close_db_pools
from workflow import workflow

# Tắt bớt log rườm rà để tập trung vào kết quả
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

async def test_simple_interaction():
    print("🚀 Đang khởi tạo kết nối database và redis...")
    await start_pooling()
    
    try:
        graph = workflow()
        
        # Payload đơn giản: Một lời chào và chia sẻ nhẹ nhàng
        user_id = "test_student_001"
        user_input = "Chào Minh, dạo này mình thấy hơi áp lực chuyện học tập quá."
        
        print(f"\n👤 User: {user_input}")
        print("🤖 Minh đang suy nghĩ...")
        
        config = {"configurable": {"thread_id": user_id}}
        
        # Khởi chạy graph
        result = await graph.ainvoke(
            {
                "user_id": user_id,
                "messages": [{"role": "user", "content": user_input}],
            },
            config=config,
        )
        
        # Hiển thị kết quả
        print("\n" + "="*50)
        print(f"✨ Phản hồi từ Minh:\n{result.get('response_text')}")
        print("="*50)
        print(f"📊 Intent: {result.get('intent_category')}")
        print(f"📌 Topic hiện tại: {result.get('current_topic')}")
        print(f"🛡️ Khẩn cấp: {result.get('is_emergency')}")
        print(f"👤 Takeover: {result.get('human_takeover')}")
        print("="*50)

    except Exception as e:
        print(f"❌ Lỗi khi chạy agent: {e}")
    finally:
        await close_db_pools()

if __name__ == "__main__":
    asyncio.run(test_simple_interaction())
