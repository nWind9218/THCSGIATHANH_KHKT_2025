import json
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import OllamaEmbeddings
from utils.database import start_pooling, get_pg_connection, fetchall, get_redis_client
from agent.state import State
embedd = OllamaEmbeddings(model="bge-m3:latest", base_url="http://127.0.0.1:11434"
)
import traceback
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
GRAPH_API_URL = "https://graph.facebook.com/v21.0/me/messages"
REDIS_URL = os.getenv("REDIS_URL")
PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER= os.getenv("DB_USERNAME")
PG_PASS= os.getenv("DB_PASSWORD")
async def embedding(text):
    """Async wrapper for embedding to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedd.embed_query, text)

async def get_user_information(state: State):
    """
        This node must be done first before get deep chat with person
        Short-term memory will be loaded all from long-term memory to gain insights
        
    """
    try:
        redis = await get_redis_client()
    except Exception :
        await start_pooling()
        redis = await get_redis_client()
    user_id = state["conversation"]["user_id"]
    messages = state["conversation"]["messages"]
    
    problem = state.get("user_emotion",{}).get("problem", None)
    query = """
        SELECT * FROM user_memory
        WHERE user_id = $1
    """
    user = await fetchall(query, user_id)
    if user:
        # Cho vào trong GraphRAG
        await redis.set(f"memory:{user_id}:preferences", user.preferences, ex=3*60*60)
        await redis.set(f"memory:{user_id}:hates", user.hates, ex=3*60*60)
    if (not user and not problem) or not problem:        
        return {
            **state,
            "conversation":{
                "user_id": user_id,
                "messages": messages,
                "is_new_user": True
            },
            "next_step":"clarify",
            "user_emotion":{
                "problem":"MUST CLARIFY MORE INFORMATION."
            }
        }
    return {**state}
    
async def update_graphrag(state: State):
    pass
async def update_cache(state: State):
    bot_plan = state["bot_plan"]
    is_new_user = state["conversation"].get("is_new_user", True)
    current_risk = state["risk"]
    current_emotion = state['user_emotion']
    user_id = state["conversation"]["user_id"]
    messages = [message for message in state["conversation"]["messages"][-4:] if message["role"] == "user"]
    try:
        redis = await get_redis_client()
    except Exception: 
        await start_pooling()
        redis = await get_redis_client()
    
    summary_data = await redis.get(f"summary:{user_id}")
    conversation_data = await redis.get(f"past:conversation:{user_id}")
    if conversation_data:
        try:
            conversation_json = json.loads(conversation_data)
            # Append messages mới vào history
            if "messages" in conversation_json:
                conversation_json["messages"].extend(messages)
            else:
                conversation_json["messages"] = messages
        except json.JSONDecodeError:
            # Nếu parse lỗi, tạo mới
            conversation_json = {"messages": messages}
    else:
        # Không có history cũ, tạo mới
        conversation_json = {"messages": messages}
    
    context_json = json.loads(summary_data)
    
    bot_plan_verified = {
        "problem": state["user_emotion"].get("problem",""),
        "bot_plan": state["bot_plan"],
        "confidence_score": state["confidence_score"],
        "risk":state.get("risk", {}),
        "language_signals": context_json.get("key_context")
    }
    payload = {
        "bot_plan": bot_plan,
        "is_new_user": is_new_user,
        "past_risk": current_risk,
        "past_emotion": current_emotion
    }
    # Điểm cần cải thiện
    # Tất cả tin nhắn giữa staff, user, và system sẽ được lưu lại và đánh giá RHLF thông qua phương thức post để gửi tới server khác để update long term memory về knowlege base
    # RHLF sẽ tạo một server khác riêng để cập nhật dữ liệu, và chúng ta sẽ gửi lại method post tới đó
    await redis.set(f"past:rhlf:{user_id}",json.dumps(bot_plan_verified, ensure_ascii=False), ex=60*60)
    await redis.set(f"past:notes:{user_id}", json.dumps(payload, ensure_ascii=False), ex=60*60)
    await redis.set(f"past:conversation:{user_id}",json.dumps(conversation_json, ensure_ascii=False), ex=60*60)
    return {}
def strip_markdown_json(content: str) -> str:
    """Remove markdown code block wrapper from LLM JSON response"""
    if not content:
        return "{}"
    
    # Remove ```json ... ``` or ``` ... ```
    content = content.strip()
    if content.startswith("```"):
        # Find first newline after ```
        first_newline = content.find('\n')
        if first_newline != -1:
            content = content[first_newline+1:]
        # Remove trailing ```
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    else: return content
async def get_emotion(state: State):
    """
        This node will be done initially to get first wanting from user,
        source of solutions later on 
        Design Doc: 
        Emotion detection is signal extraction, not diagnosis.
        Risk and action decisions must be separated.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    try:
        redis = await get_redis_client()
    except Exception :
        await start_pooling()
        redis = await get_redis_client()
    messages = state["conversation"]["messages"][-1]["content"]
    user_id = state['conversation']["user_id"]
    conv_summary = await redis.get(f"summary:{user_id}")
    if conv_summary:
        try:
            prev = json.loads(conv_summary)
            prev_intense = prev.get("intense")
        except:
            prev_intense = None
    emotion_prompt = f"""
        You are an EMOTION SIGNAL EXTRACTOR for an AI system.

        Your role:
        - Extract emotional signals ONLY from the CURRENT user message.
        - Urgency is NOT emotional — it reflects PROGRESSION over time and MUST be derived from INTENSE in conversation memory.

        Inputs:
        - Current user message
        - Conversation summary (authoritative memory, may be null)
        Current Intense:
        - CURRENT INTENSE
        Current Intense:
        "{prev_intense}"
        Current message:
        "{messages}"

        Conversation summary:
        {conv_summary}

        --------------------------------
        CORE RULES (VERY IMPORTANT):

        1. Analyze ONLY the current message for emotion.
        2. Do NOT diagnose mental health conditions.
        3. Do NOT determine crisis level.
        4. If emotional signals are weak, factual, or neutral → status = "uncertain".
        5. Prefer "uncertain" over guessing.

        --------------------------------
        INTENSE → URGENCY LOGIC (AUTHORITATIVE):

        INTENSE represents progression, NOT emotion.

        Minimum urgency MUST follow INTENSE:
        - starting → urgency = "normal"
        - growing → urgency = "watch"
        - request_high_urgency → urgency = "immediate"

        Rules:
        - Urgency can ONLY stay the same or increase.
        - Urgency MUST NEVER be lower than the INTENSE-based minimum.
        - The current message may ONLY increase urgency if it explicitly indicates immediacy.
        - If INTENSE is missing or undefined → assume "starting".

        --------------------------------
        SPECIAL CASE OVERRIDES (DO NOT VIOLATE INTENSE RULES):

        1. Greetings / Small Talk
        Examples: "Hi", "Hello", "Xin chào"
        - status: "joy"
        - problem: ""
        - self_harm: false
        - violence: false
        - urgency: INTENSE-based minimum
        - confidence_score: 0.9

        2. Casual identity questions
        Examples: "What's your name?", "Bạn là ai?"
        - status: "uncertain"
        - problem: ""
        - urgency: INTENSE-based minimum
        - confidence_score: 0.8

        3. Unclear or vague messages
        - status: "uncertain"
        - problem: "MUST CLARIFY MORE INFORMATION."
        - urgency: INTENSE-based minimum
        - confidence_score: 0.3

        --------------------------------
        OUTPUT FORMAT (STRICT):

        Return JSON ONLY with EXACT schema:
        {{
        "status": "joy | sadness | fear | disgust | anger | surprise | uncertain",
        "problem": "short phrase or empty string",
        "metadata": {{
            "trigger": "",
            "duration": "",
            "context": ""
        }},
        "self_harm": true | false,
        "violence": true | false,
        "urgency": "normal | watch | immediate",
        "confidence_score": 0.0
        }}

    """
    try:
        response = await llm.ainvoke(emotion_prompt)
        cleaned_content = strip_markdown_json(response.content)
        data = json.loads(cleaned_content)

        return {
            **state,
            "user_emotion": {
                "status": data.get("status", "uncertain"),
                "problem": data.get("problem", ""),
                "metadata": data.get("metadata", {
                    "trigger": "",
                    "duration": "",
                    "context": ""
                }),
            },
            "risk": {
                "self_harm": data.get("self_harm", False),
                "violence": data.get("violence", False),
                "urgency": "normal" if data.get("status", "uncertain") == "uncertain" else data.get("urgency")
            },
            "confidence_score": float(data.get("confidence_score", 0.3) )
        }

    except Exception as e:
        print(f"[get_emotion] Error: {str(e)}")
        return {
            "user_emotion": {
                "status": "uncertain",
                "problem": "",
                "metadata": {
                    "trigger": "",
                    "duration": "",
                    "context": ""
                },
                "crisis_level": "low",
            },
            "confidence_score": 0.1
        }
SIM_THRESHOLD = 0.78

async def retrieve_risk_assessment(state: State):
    query = """
        SELECT
            problem,
            solution,
            tone,
            must_not_do,
            level,
            language_signals,
            self_harm,
            violence,
            urgency,
            1 - (embedding <=> $1::vector) AS similarity
        FROM bot_knowledge
        ORDER BY embedding <=> $1::vector
        LIMIT 1
    """

    # ---- Ensure DB connection ----
    try:
        await get_pg_connection()
    except Exception:
        await start_pooling()
        await get_pg_connection()

    problem = state["user_emotion"].get("problem")

    # ---- Hard guard: no clear problem ----
    if not problem or problem == "MUST CLARIFY MORE INFORMATION.":
        return {
            **state,
            "user_emotion": {
                **state["user_emotion"],
                "is_new_problem": False
            }
        }

    vector = await embedding(problem)
    rows = await fetchall(query, str(vector))

    # ---- No retrieval result ----
    if not rows:
        return {
            **state,
            "user_emotion": {
                **state["user_emotion"],
                "is_new_problem": True
            }
        }

    result = rows[0]
    similarity = result.get("similarity", 0)

    # ---- 🔥 CORE SAFETY: similarity too low → IGNORE ----
    if similarity < SIM_THRESHOLD:
        return {
            **state,
            "user_emotion": {
                **state["user_emotion"],
                "is_new_problem": True
            },
            "rag_meta": {
                "ignored": True,
                "similarity": similarity
            }
        }

    # ---- Accepted retrieval ----
    crisis_dict = {
        "0": "low",
        "1": "medium",
        "2": "high",
        "3": "critical"
    }

    return {
        **state,
        "bot_plan": {
            "solution": result["solution"],
            "must_not_do": result["must_not_do"],
            "tone": result["tone"]
        },
        "risk": {
            "self_harm": result["self_harm"],
            "violence": result["violence"],
            "urgency": result.get("urgency", "normal")
        },
        "crisis_level": crisis_dict.get(str(result["level"]), "low"),
        "rag_meta": {
            "ignored": False,
            "similarity": similarity
        }
    }


async def guest_risk_assesment(state: State):
    should_guest = state["user_emotion"].get("is_new_problem", True)
    if should_guest:
        confidence_score = state["confidence_score"]
        risk_violence = state["risk"]["violence"]
        risk_self_harm = state["risk"]["self_harm"]
        emotion_status = state["user_emotion"]["status"]
        if risk_self_harm and confidence_score >= 0.7 and emotion_status in ["sadness", "fear"]:
            return {
                **state,
                "risk":{
                    **state["risk"],
                    "urgency":"immediate"
                },
                "user_emotion":{
                    **state["user_emotion"],
                    "crisis_level":"critical"
                }
            }
        elif (risk_self_harm or risk_violence) and confidence_score >= 0.6 and emotion_status in ["sadness", "fear", "anger"]:
            return {
                **state,
                "risk":{
                    **state["risk"],
                    "urgency":"watch"
                },
                "user_emotion":{
                    **state["user_emotion"],
                    "crisis_level":"high"
                }   
            }
        elif (risk_self_harm or risk_violence) and confidence_score >= 0.4 and emotion_status in ["sadness", "fear", "anger"]:
            # Can phai validate them cam xuc
            return {
                **state,
                "risk":{
                    **state["risk"],
                    "urgency":"watch"
                },
                "user_emotion":{
                    **state["user_emotion"],
                    "crisis_level":"medium"
                }
            }
        else:
            return {
                **state,
                "risk":{
                    **state["risk"],
                    "urgency": "normal"
                },
                "user_emotion":{
                    **state["user_emotion"],
                    "crisis_level":"low"
                }
            }
    else:
        return {**state, "crisis_level": "low"}
async def route_after_decision(state: State):
    urgency_res = await is_urgent(state)
    if urgency_res == "response_emergency":
        return "response_emergency"
    plan_res = await should_rotate_plan(state)
    return plan_res
async def bot_planning(state: State):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    try:
        redis = await get_redis_client()
    except Exception :
        await start_pooling()
        redis = await get_redis_client()
    user_id = state["conversation"]["user_id"]
    
    # Get user context
    user_message = state["conversation"]["messages"][-1]["content"]
    emotion_status = state["user_emotion"]["status"]
    problem = state["user_emotion"]["problem"]
    crisis_level = state["user_emotion"].get("crisis_level", "low")
    metadata = state["user_emotion"]["metadata"]
    confidence_score = state["confidence_score"]
    urgency = state["risk"]["urgency"]
    self_harm = state["risk"]["self_harm"]
    violence = state["risk"]["violence"]
    
    # Get conversation summary
    conv_summary = await redis.get(f"summary:{user_id}")
    summary_data = json.loads(conv_summary) if conv_summary else {}
    print("[bot_planning] PREVIOUS SUMM: ", summary_data)
    # Get user preferences if available
    user_preferences = await redis.get(f"memory:{user_id}:preferences")
    user_hates = await redis.get(f"memory:{user_id}:hates")
    plan_prompt = f"""
        You are a compassionate conversation planner for an AI assistant supporting teenagers (10-16 years old).
        Your task is to create a structured response plan based on the user's emotional state and context.

        ## INPUT DATA:

        **User Message:**
        "{user_message}"

        **Emotional Analysis:**
        - Status: {emotion_status}
        - Problem: {problem}
        - Crisis Level: {crisis_level}
        - Confidence Score: {confidence_score}
        - Trigger: {metadata.get('trigger', 'N/A')}
        - Duration: {metadata.get('duration', 'N/A')}
        - Context: {metadata.get('context', 'N/A')}

        **Risk Assessment:**
        - Urgency: {urgency}
        - Self-harm risk: {self_harm}
        - Violence risk: {violence}

        **Conversation Context:**
        {json.dumps(summary_data, ensure_ascii=False, indent=2) if summary_data else "No previous context"}

        **User Profile:**
        - Preferences: {user_preferences if user_preferences else "Unknown"}
        - Dislikes: {user_hates if user_hates else "Unknown"}

        ---

        ## YOUR TASK:

        Create a JSON Conversation plan that includes:

        1. **Solution Strategy**: What approach should the bot take?
        - Options: "empathize_first", "provide_guidance", "ask_clarifying_questions", "offer_resources", "gentle_redirect", "validate_feelings"

        2. **Tone**: How should the bot communicate?
        - Options: "warm_supportive", "calm_reassuring", "gentle_curious", "cheerful_encouraging", "serious_concerned", "playful_light"

        3. **Must Not Do**: What should the bot absolutely avoid?
        - Be specific based on the emotional state and risks
        RETURN JSON ONLY.
   """
    try:
        response = await llm.ainvoke(plan_prompt)
        cleaned_content = strip_markdown_json(response.content)
        plan_data = json.loads(cleaned_content)
        print(f"✅ RECEIVED [bot_planning]: {plan_data}")
        
        return {
            **state,
            "bot_plan": {
                "solution": plan_data.get("solution", "validate_feelings"),
                "tone": plan_data.get("tone", "warm_supportive"),
                "must_not_do": plan_data.get("must_not_do", []),
            }
        }
    except Exception as e:
        print(f"[bot_planning] Error: {str(e)}")
        return {
            **state,
            "bot_plan": {
                "solution": "validate_feelings",
                "tone": "warm_supportive",
                "must_not_do": ["dismiss feelings", "give medical advice", "make promises"],
                "conversation_goals": ["make user feel heard", "assess situation"],
                "next_steps": "continue_listening",
                "key_message_points": [
                    "Acknowledge their feelings",
                    "Let them know they're not alone",
                    "Ask gentle follow-up questions"
                ],
                "reasoning": "Safe fallback plan for validation and listening"
            }
        }
async def should_rotate_plan(state: State):
    bot_plan = state.get("bot_plan", None)
    solution = None
    tone = None
    must_not_do = None
    
    if bot_plan is not None:
        solution = bot_plan.get("solution")
        tone = bot_plan.get("tone")
        must_not_do = bot_plan.get("must_not_do")
    user_id = state["conversation"]["user_id"]
    try:
        redis = await get_redis_client()
    except Exception:
        await start_pooling()
        redis = await get_redis_client()

    key = await redis.get(f"summary:{user_id}")
    response = json.loads(key)
    if solution is None or tone is None or must_not_do is None or response["status"] == "changed":
        return "bot_planning"
    else:
        return "gen_response"
async def decide_next_step(state: State):
    crisis_level = state["user_emotion"].get("crisis_level","low")
    urgency = state["risk"]["urgency"]
    confidence_score = state["confidence_score"]
    self_harm = state["risk"]["self_harm"]
    violence = state["risk"]["violence"]
    problem = state["user_emotion"]["problem"]
    emotion_status = state["user_emotion"]["status"]
    if (urgency == "immediate" and crisis_level in ["critical", "high"]) and confidence_score >= 0.7:
        return {
            **state,
            "next_step":"escalate"
        }
    elif urgency == "watch" and crisis_level == "high" and confidence_score >= 0.6:
        return {
            **state,
            "next_step":"guide"
        }  
    elif crisis_level == 'medium' and confidence_score >= 0.5:
        if self_harm or violence:
            return {
                **state,
                "next_step":"guide"
            }
        else:
            return {
                **state,
                "next_step":"comfort"
            } 
    elif confidence_score < 0.5 or emotion_status == "uncertain":
        if not problem or problem == "MUST CLARIFY MORE INFORMATION.":
            return {
                **state,
                "next_step": "clarify"
            }
    else:
        return {
            **state,
            "next_step":"listen"
        }
async def summary_conv_history(state: State):
    """
    Conversation summary is intent memory, not transcript memory.
    Assistant messages are never part of long-term context.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2, 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    try:
        redis = await get_redis_client()
    except Exception:
        await start_pooling()
        redis = await get_redis_client()
    user_id = state["conversation"]["user_id"]
    messages = state["conversation"]["messages"]
    
    # Parse messages if string
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = []
    
    # Ensure messages is a list
    if not isinstance(messages, list):
        messages = []
    
    # Extract CHỈ user messages từ toàn bộ messages (đã được reducer giữ lại 5 cái gần nhất)
    recent_user_msgs = []
    for m in messages:
        if isinstance(m, dict):
            # Chỉ lấy user messages
            if m.get("role") == "user" and m.get("content"):
                recent_user_msgs.append(m["content"])
    
    # Nếu không lấy được tin nhắn nào, lấy tin nhắn cuối cùng
    if not recent_user_msgs and messages:
        last_msg = messages[-1]
        if isinstance(last_msg, dict) and "content" in last_msg:
            recent_user_msgs = [last_msg["content"]]
    
    lastest_message = " | ".join(recent_user_msgs) if recent_user_msgs else ""
    recent_msgs = await redis.get(f"past:conversation:{user_id}")
    print(f"[summary_conv_history] 📨 Recent user messages: {recent_msgs}")
    print(f"Curent messages: ", lastest_message)
    last_summary = await redis.get(f"summary:{user_id}")
    prev_intense = None
    if last_summary:
        try:
            prev = json.loads(last_summary)
            prev_intense = prev.get("intense")
        except:
            prev_intense = None

    new_summary_prompt = f"""
        Nhiệm vụ của bạn là Cập nhật trí nhớ về Ý ĐỊNH + CONTEXT của người dùng trở nên thật ngắn gọn, súc tích, diễn đạt được context mà cuộc trò chuyện đã trải qua
        INPUT:
        - Previous summary:{json.dumps(last_summary, ensure_ascii=False) if last_summary else "None"}
        - Previous Intense Level: {prev_intense}
        - Recent messages: {recent_msgs}
        - Current user messages: {json.dumps(lastest_message, ensure_ascii=False)}

        Task:
        - Decide whether the user's intent is CONTINUING or CHANGED. 
        - If continuing: enrich or slightly update the summary.
        - If changed: replace the summary to reflect the new intent.
        
        Rules:
        - Focus on user intent and context only.
        - If previous status is CONTINUING, current status is CHANGED, replace EVERYTHING
        - Ignore assistant messages.
        - Keep it concise and durable.
        - Do NOT guess emotions if unclear.
        - If previous summary IS NOT DEFINED, status MUST AUTOMATICALLY be set to CHANGED
        - INTENSE must be based on previous summary, RECENT MESSAGES, CURRENT USER MESSAGES, if
        - "intense" indicates the urgency or progression level of the user's intent,
        INTENSE DECISION RULES (STRICT):

        You MUST decide intense by PROGRESSION, not emotion.

        1. starting:
        - First time this intent appears
        - OR previous summary does not exist
        - OR user speaks in general / exploratory terms

        2. growing:
        - Same intent appears AGAIN
        - OR user adds more details / examples
        - OR user asks follow-up questions on same topic
        - Previous intense MUST be starting or growing

        3. request_high_urgency:
        - User explicitly requests immediate action, help, or resolution
        - OR language indicates "cannot wait", "right now", "urgent"
        - OR repetition with frustration + demand for action
        - Previous intense MUST be growing

        IMPORTANT:
        - Intense can ONLY stay the same or increase.
        - NEVER decrease intense.

        {{
            "intent": "",
            "main_topic": "",
            "intense": starting → growing → request_high_urgency
            "status": "continuing | changed",
            "key_context": []
            "confidence_score": float in range [0.0, 1.0]
        }}
        RETURN JSON ONLY 
        """
    try:
        response = await llm.ainvoke(new_summary_prompt)
        print(response.content)
        cleaned_content = strip_markdown_json(response.content)
        summary_data = json.loads(cleaned_content)
        await redis.set(
            f"summary:{user_id}",
            json.dumps({
                "intent":summary_data.get("intent",""),
                "intense":summary_data.get("intense",""),
                "main_topic":summary_data.get("main_topic",""),
                "status": summary_data.get("status","continuing"),
                "key_context":summary_data.get("key_context",[])
            }, ensure_ascii=False),
            ex=60*60
        )
        return {
            **state,
            "confidence_score": summary_data.get("confidence_score", 0.3)
        }
    except Exception as e:
        print(f"[summary_conv_history] Error: {str(e)}")
        print(f"[summary_conv_history] Error type: {type(e).__name__}")
        print(f"[summary_conv_history] Error details: {repr(e)}")
        print(f"[summary_conv_history] Traceback:\n{traceback.format_exc()}")
        return state
async def is_information_loaded(state: State):
    user_id = state["conversation"]["user_id"]
    redis = await get_redis_client() 
    if await redis.get(f"memory:{user_id}:preferences"):
        return "should_get_emotion"
    else:
        return "get_user_information" 
async def should_get_emotion(state: State):
    try:
        redis = await get_redis_client()
    except Exception :
        await start_pooling()
        redis = await get_redis_client()
    try:
        user_id = state["conversation"]["user_id"]
        last_summary = await redis.get(f"summary:{user_id}")
        summary_data = json.loads(last_summary)
        if summary_data["status"] == "changed":
            return "get_emotion"
        else:
            return "retrieve_risk_assessment"
    except Exception as e:
        print(f"Lỗi tại node should_get_emotion: {str(e)}")
        return "get_emotion"
async def pickup_intervenor(state: State):
    """
        This node is used with condtional edge from get_emotion
        If state contains intevenor_type, then start going in compatible intervenor's subgraph
        Else return output to user (case user ask something isn't related to problem (like Hello or something else))
    """
    intervenor_type = state["user_emotion"]["intervention_type"]
    if not intervenor_type:
        return "refine"
    return intervenor_type
async def generate_response(state: State):
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    try:
        redis = await get_redis_client()
    except Exception:
        await start_pooling()
        redis = await get_redis_client()
    user_id = state["conversation"]["user_id"]
    
    # Extract all context
    user_message = state["conversation"]["messages"][-1]["content"]
    emotion_status = state["user_emotion"]["status"]
    problem = state["user_emotion"]["problem"]
    crisis_level = state["user_emotion"].get("crisis_level", "low")
    next_step = state.get("next_step", "listen")
    
    # Get bot plan
    bot_plan = state.get("bot_plan", {})
    solution = bot_plan.get("solution", "validate_feelings")
    tone = bot_plan.get("tone", "warm_supportive")
    must_not_do = bot_plan.get("must_not_do", [])
    
    # Get conversation summary
    conv_summary = await redis.get(f"summary:{user_id}")
    summary_data = json.loads(conv_summary) if conv_summary else {}
    
    # Get user preferences
    user_preferences = await redis.get(f"memory:{user_id}:preferences")
    user_hates = await redis.get(f"memory:{user_id}:hate")
    
    if next_step == "clarify":
        system_instruction = """Bạn là Mimi - trợ lý ảo thân thiện của trường THCS Gia Thanh.
        Nhiệm vụ: Đặt câu hỏi MỞ để hiểu rõ hơn vấn đề em.

        Quy tắc:
        - Dùng giọng điệu tò mò, thân thiện
        - Hỏi 1-2 câu ngắn gọn
        - Không giả định vấn đề
        - Không áp đặt cảm xúc
        - Dùng emoji nhẹ nhàng 😊

        Ví dụ tốt:
        "Mình có thể kể thêm về chuyện đó được không? 🤔"
        "Em cảm thấy như thế nào về việc này? 💭"
    """

    elif next_step == "guide":
        system_instruction = f"""Bạn là Mimi - trợ lý ảo hỗ trợ học đường của trường THCS Gia Thanh.
        Nhiệm vụ: Hướng dẫn em qua vấn đề với sự đồng cảm và chiến lược cụ thể.

        Cảm xúc của em: {emotion_status}
        Vấn đề: {problem}
        Mức độ: {crisis_level}

        Phương pháp tiếp cận: {solution}
        Giọng điệu: {tone}

        Tuyệt đối TRÁNH:
        {chr(10).join(f"- {item}" for item in must_not_do)}

        Hướng dẫn phản hồi:
        1. Thừa nhận cảm xúc (1 câu ngắn)
        2. Đề xuất 1-2 bước hành động CỤ THỂ
        3. Hỏi em muốn thử bước nào trước
        4. Dùng ví dụ từ ngữ cảnh học đường
        5. Kết thúc khuyến khích

        Độ dài: 3-5 câu ngắn
        Emoji: Dùng 2-3 emoji phù hợp (nếu mức crisis_level không phải là high hoặc critical)
    """

    else:  # listen, comfort
        system_instruction = f"""Bạn là Mimi - trợ lý ảo đồng hành cùng học sinh trường THCS Gia Thanh.
        Nhiệm vụ: Lắng nghe, thấu hiểu, và đồng hành cùng em.

        Cảm xúc của em: {emotion_status}
        Vấn đề: {problem if problem else "Chưa rõ"}

        Phương pháp: {solution}
        Giọng điệu: {tone}

        KHÔNG được làm:
        {chr(10).join(f"- {item}" for item in must_not_do) if must_not_do else "- Phán xét, đưa ra lời khuyên y tế"}

        Quy tắc phản hồi:
        - Phản ánh cảm xúc em đang trải qua
        - Xác nhận cảm xúc của em là hợp lý
        - Đặt 1 câu hỏi mở để em chia sẻ thêm
        - Giữ giọng điệu ấm áp, không áp đặt
        - Độ dài: 2-4 câu ngắn
        - Emoji: 1-2 emoji nhẹ nhàng
    """
    
    generation_prompt = f"""{system_instruction}

**NGỮ CẢNH CUỘC TRÒ CHUYỆN:**
    {json.dumps(summary_data, ensure_ascii=False, indent=2) if summary_data else "Cuộc trò chuyện mới"}

**TIN NHẮN CỦA EM:**
    "{user_message}"

**HỒ SƠ EM:**
- Sở thích: {user_preferences.decode() if user_preferences else "Chưa biết"}
- Không thích: {user_hates.decode() if user_hates else "Chưa biết"}

---

**YÊU CẦU CUỐI:**
- Gọi em bằng "em" hoặc "bạn"
- Tự xưng là "Mimi" hoặc "mình"
- Phù hợp lứa tuổi 10-16
- Ngôn ngữ Tiếng Việt tự nhiên
- Tránh thuật ngữ phức tạp
- GIỮ NGẮN GỌN (không quá 5 câu)

FINALLY: RETURN EXACT JSON FORMAT:
{{
    "response": ""
}}
RETURN JSON ONLY."""
    
    try:
        response = await llm.ainvoke(generation_prompt)
        cleaned_content = strip_markdown_json(response.content)
        data = json.loads(cleaned_content)

        return {
            **state,
            "response": {
                "output": data.get("response", "Xin lỗi, Mimi hiện không thể hỗ trợ bạn lúc này!")
            }
        }

    except Exception as e:
        print(f"[generate_response] Error: {str(e)}")
        return {
            **state,
            "response": {
                "output": "Xin lỗi, Mimi hiện không thể hỗ trợ bạn lúc này!"
            }
        }

async def is_urgent(state: State):
    urgency = state["risk"]["urgency"]
    crisis_level = state["user_emotion"].get("crisis_level", "low")
    if urgency == "immediate" and crisis_level in ["high", "critical"]: 
        return "response_emergency"
    else:
        return "should_rotate_plan"
async def response_emergency(state: State):
    return {
        **state,
        "response":{
            "output":"""
                Mình rất tiếc khi nghe bạn đang trải qua điều này 💔
                Có vẻ như lúc này bạn đang cảm thấy rất khó khăn, và cảm giác đó hoàn toàn không sai.
                
                Bạn không cần phải đối mặt với chuyện này một mình. Nếu có thể, bạn hãy thử chia sẻ với người lớn mà bạn tin tưởng như bố mẹ, thầy cô, hoặc người thân nhé.
                Mình có thể giúp bạn liên hệ với cô Thúy và cô Trâm để chúng ta cùng nhau vượt qua vấn đề của bạn nhé!
                
                Điều quan trọng nhất là: **bạn xứng đáng được lắng nghe và được giúp đỡ** 🌱
                Mimi vẫn ở đây để lắng nghe bạn
            """
        }
    }

async def empathy_node(state: State):
    pass
async def pickup_mood(state:State):
    """
    """
    pass
async def suggest_to_help(state: State):
    pass
async def give_options(state: State):
    """
        This is for Guidance LLM
    """
    pass
async def clarify(state: State):
    pass
async def gentle_expand(state: State):
    pass
async def solve_emergency(state: State):
    pass