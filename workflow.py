import asyncio
import logging
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import redis
from redisvl.utils.vectorize import CustomTextVectorizer
from redisvl.extensions.cache.llm import SemanticCache
from utils.database import get_pg_connection, fetchall, fetch_one, get_redis_client, get_redis_checkpointer
from typing import Optional, List
from agent.state import UserEmotion
from SYSTEM_PROMPT.registry import prompt_registry
load_dotenv()

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

embedd = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
REDIS_URL = os.getenv("REDIS_URL")
PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER = os.getenv("DB_USERNAME")
PG_PASS = os.getenv("DB_PASSWORD")

async def embedding(text):
    """Async wrapper for embedding to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedd.embed_query, text)

class AgentState(TypedDict):
    messages: str
    emotion_state: UserEmotion
    history: Optional[List[str]]
    rag_results: str
    llm_response: str
    final_response: str
    error: str
    
async def search_with_rag(state: AgentState) -> AgentState:
    try:
        # ✅ Đảm bảo pool được khởi tạo (quan trọng cho LangGraph Studio)
        from utils.database import start_pooling
        try:
            pg_pool = await get_pg_connection()
        except Exception:
            # Nếu pool chưa có, khởi tạo tự động
            await start_pooling()
            pg_pool = await get_pg_connection()
        
        question = state['messages']
        
        question_vector = await embedding(question)
        vector_str = '[' + ','.join(map(str, question_vector)) + ']'
        
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
        
        # Load RAG selection prompt from registry
        rag_template = prompt_registry.get_function("rag_selection")
        selection_prompt = rag_template.format(
            question=question,
            formatted_results=json.dumps(formatted_results, ensure_ascii=False, indent=2)
        )

        messages = [
            SystemMessage(content="Bạn là chuyên gia phân tích và lựa chọn thông tin chính xác."),
            HumanMessage(content=selection_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        selected = json.loads(response.content)
        
        logger.info(f"✅ STEP 1: LLM SELECTED ANSWER {selected}")
        
        state["rag_results"] = selected["selected_answer"]
        
        return state
    except Exception as e:
        state["rag_results"] = ""
        state["error"] = str(e)
        logger.error(f"Error In RAG: {str(e)}")
        return state
async def ask_llm_node(state: AgentState) -> AgentState:
    """Ask ChatGPT directly"""
    try:
        question = state["messages"]
        
        # Load agent prompt from registry
        agent_prompt = prompt_registry.get_agent("mimi_school")
        
        messages = [
            SystemMessage(content=agent_prompt),
            HumanMessage(content=question)
        ]
        response = await llm.ainvoke(messages)
        logger.info(f"✅ STEP 2: ASK GPT\n{response}")
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
        
        # Load response refinement prompt from registry
        refine_template = prompt_registry.get_function("response_refinement")
        refine_prompt = refine_template.format(
            question=question,
            answer=answer
        )
        
        # Load agent prompt for system context
        agent_prompt = prompt_registry.get_agent("mimi_school")
        
        messages = [
            SystemMessage(content=agent_prompt + "\n\n" + refine_prompt),
            HumanMessage(content=f"Câu hỏi: {question}\n\nCâu trả lời cần chỉnh sửa: {answer}")
        ]
        response = await llm.ainvoke(messages)
        logger.info(f"✅ STEP 3: REFINE BOT RESPONSE\n{response}")
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

def workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("search_rag", search_with_rag)
    workflow.add_node("ask_llm", ask_llm_node)
    workflow.add_node("refine", refine_response_node)

    workflow.set_entry_point("search_rag")

    workflow.add_conditional_edges(
        "search_rag",
        should_use_llm,
        {
            "ask_llm":"ask_llm",
            "refine": "refine" 
        }
    )
    workflow.add_edge("ask_llm", "refine")
    workflow.add_edge("search_rag", END)

    return workflow.compile()
from agent.state import State
from agent.tools import response_and_update,route_after_decision,bot_planning, is_information_loaded, response_emergency, guest_risk_assesment,retrieve_risk_assessment,decide_next_step,get_user_information,summary_conv_history, get_emotion
from langgraph.checkpoint.memory import MemorySaver

def workflow2():
    workflow2 = StateGraph(State)

    workflow2.add_node("summary_conv_history", summary_conv_history)
    workflow2.add_node("get_emotion", get_emotion)
    workflow2.add_node("gen_response", response_and_update)
    workflow2.add_node("get_user_information", get_user_information)
    workflow2.add_node("decide_next_step", decide_next_step)
    workflow2.add_node("retrieve_risk_assessment", retrieve_risk_assessment)
    workflow2.add_node("guest_risk_assessment", guest_risk_assesment)
    workflow2.add_node("response_emergency", response_emergency)
    workflow2.add_node("bot_planning", bot_planning)

    workflow2.set_entry_point("summary_conv_history")

    workflow2.add_conditional_edges(
        "summary_conv_history",
        is_information_loaded,
        {
            "should_get_emotion": "get_emotion",
            "get_user_information": "get_user_information"
        }
    )
    workflow2.add_edge("get_user_information","get_emotion")
    workflow2.add_edge("get_emotion", "retrieve_risk_assessment")
    workflow2.add_edge("retrieve_risk_assessment", "guest_risk_assessment")
    workflow2.add_edge("guest_risk_assessment", "decide_next_step")

    workflow2.add_conditional_edges(
        "decide_next_step",
        route_after_decision,
        {
            "response_emergency": "response_emergency",
            "bot_planning": "bot_planning",
            "gen_response":"gen_response"
        }
    )
    workflow2.add_edge("bot_planning", "gen_response")

    workflow2.add_edge("response_emergency", END)
    workflow2.add_edge("gen_response", END)
    return workflow2.compile()

app = workflow2()
async def main():
    """✅ Test workflow - CẦN KHỞI TẠO POOL TRƯỚC"""
    logger.info("🚀 Starting workflow test...\n")
    
    # ✅ QUAN TRỌNG: Khởi tạo pool trước khi dùng
    from utils.database import start_pooling, close_db_pools
    await start_pooling()
    
    try:
        while True:
            question = input("Nhập câu hỏi: ")
            if question == "end":
                break
            
            config = {"configurable": {"thread_id": "user_123"}}
            # ✅ Chỉ truyền tin nhắn mới - LangGraph sẽ tự merge với checkpoint
            input_data = {
                "conversation": {
                    "messages": [{"content": question, "role": "user"}],
                    "user_id": "test_user"
                }
            }
            result = await app.ainvoke(input_data, config=config)
            
            logger.info(f"\n📦 Final Response: {result.get('response', {}).get('output', '')}")
            
    finally:
        await close_db_pools()

if __name__ == "__main__":
    asyncio.run(main())
