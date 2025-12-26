import asyncio
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import asyncpg
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import redis
from redisvl.utils.vectorize import CustomTextVectorizer
from redisvl.extensions.cache.llm import SemanticCache
from utils.database import get_pg_connection, fetchall, fetch_one, get_redis_client

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

embedd = OllamaEmbeddings(model="bge-m3:latest",
                          base_url="http://localhost:11434")
REDIS_URL = os.getenv("REDIS_URL")
PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER = os.getenv("DB_USERNAME")
PG_PASS = os.getenv("DB_PASSWORD")

def embedding(text):
    return embedd.embed_query(text)

class AgentState(TypedDict):
    messages: str
    rag_results: str
    llm_response: str
    final_response: str
    error: str
    
async def search_with_rag(state: AgentState) -> AgentState:
    try:
        question = state['messages']

        question_vector = embedding(question)
        vector_str = '[' + ','.join(map(str, question_vector)) + ']'
        
        pg_pool = await get_pg_connection()
        
        query = """
            SELECT question, answer, bot_type,
            1 - (embedding <=> $1::vector) as similarity
            FROM bot_knowledge
            ORDER BY embedding <=> $1::vector
            LIMIT 3
        """
        
        results = await fetchall(query, vector_str)

        if not results:
            state["rag_results"] = ""
            return state
            
        formatted_results = []
        for row in results:
            formatted_results.append({
                "question": row[0],
                "answer": row[1],
                "similarity": float(row[3])
            })
        
        # ✅ Để LLM chọn câu trả lời tốt nhất
        selection_prompt = f"""Dựa vào câu hỏi của người dùng, hãy chọn câu trả lời PHÙ HỢP NHẤT từ các đáp án sau:

Câu hỏi: {question}

Các đáp án:
{json.dumps(formatted_results, ensure_ascii=False, indent=2)}

Chỉ trả về JSON với format: {{"selected_answer": "câu trả lời được chọn"}}"""

        messages = [
            SystemMessage(content="Bạn là chuyên gia phân tích và lựa chọn thông tin chính xác."),
            HumanMessage(content=selection_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        selected = json.loads(response.content)
        
        print("✅ STEP 1: LLM SELECTED ANSWER", selected)
        
        state["rag_results"] = selected["selected_answer"]
        
        return state
    except Exception as e:
        state["rag_results"] = ""
        state["error"] = str(e)
        print(f"Error In RAG :{str(e)}")
        return state
async def ask_llm_node(state: AgentState) -> AgentState:
    """Ask ChatGPT directly"""
    try:
        question = state["messages"]
        messages = [
            SystemMessage(content="Bạn là trợ lý ảo có tên gọi là Mimi, là trí thông minh nhân tạo của một shop hàng trực tuyến, bạn sẽ là người có trách nhiệm hỗ trợ customers để tư vấn về sản phẩm. Sản phẩm là một ứng dụng website chatbot online trực tuyến"),
            HumanMessage(content=question)
        ]
        response = await llm.ainvoke(messages)
        print("✅ STEP 2: ASK GPT\n", response)
        state["llm_response"] = response.content
        return state
    except Exception as e:
        state["error"] = str(e)
        return state

async def refine_response_node(state: AgentState) -> AgentState:
    """Refine the response for children"""
    try:
        question = state["messages"]
        # Use RAG result if available, otherwise use LLM response
        answer = state.get("rag_results") or state.get("llm_response", "")
        
        messages = [
            SystemMessage(content="""
            Hãy điều chỉnh câu trả lời theo yêu cầu sau:
            - Sử dụng ngôn ngữ thật ngắn gọn, dễ hiểu, giọng điệu nhí nhảnh, phù hợp lứa tuổi
            - Tránh thuật ngữ phức tạp
            - Sử dụng câu từ ngắn gọn, xúc tích
            - Thêm các emoji, để tạo sự hấp dẫn đối với người đọc
            - Thêm ví dụ minh họa nếu cần
            - Giữ ngữ điệu thân thiện, khuyến khích"""),
            HumanMessage(content=f"Câu hỏi: {question}\n\nCâu trả lời cần chỉnh sửa{answer}")
        ]
        response = await llm.ainvoke(messages)
        print("✅ STEP 3: REFINE BOT RESPONSE\n", response)
        state["final_response"] = response.content
        return state
    except Exception as e:
        state["error"] = str(e)
        return state
# Conditional edge function
def should_use_llm(state: AgentState) -> str:
    """Decide if we need to call LLM"""
    if state.get("rag_results"):
        return "refine"
    return "ask_llm"

def workflow_demo():
    workflow = StateGraph(AgentState)

    # workflow.add_node("search_rag", search_with_rag)
    workflow.add_node("ask_llm", ask_llm_node)
    workflow.add_node("refine", refine_response_node)

    workflow.set_entry_point("ask_llm")

    # workflow.add_conditional_edges(
    #     "search_rag",
    #     should_use_llm,
    #     {
    #         "ask_llm":"ask_llm",
    #         "refine": "refine" 
    #     }
    # )
    workflow.add_edge("ask_llm", "refine")
    workflow.add_edge("refine", END)

    return workflow.compile()
app = workflow_demo()

async def main():
    """✅ Test workflow - CẦN KHỞI TẠO POOL TRƯỚC"""
    print("🚀 Starting workflow test...\n")
    
    # ✅ QUAN TRỌNG: Khởi tạo pool trước khi dùng
    from utils.database import start_pooling, close_db_pools
    await start_pooling()
    
    try:
        while True:
            question = input("Nhập câu hỏi: ")
            if question == "end":
                break
                
            config = {"configurable": {"thread_id": "user_123"}}
            input_data = {
                'messages': question,
                'rag_results': '',
                'llm_response': '',
                'final_response': '',
                'error': ''
            }
            result = await app.ainvoke(input_data, config=config)
            
            print("\n📦 Final Response:", result.get("final_response"))
            
    finally:
        await close_db_pools()

if __name__ == "__main__":
    asyncio.run(main())