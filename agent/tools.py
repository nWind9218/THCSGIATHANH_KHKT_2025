import json
import os
import asyncio
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import OllamaEmbeddings
from utils.database import get_pg_connection, fetchall, get_redis_client
from agent.state import State

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
embedd = OllamaEmbeddings(model="bge-m3:latest", base_url=OLLAMA_HOST
)
import traceback

logger = logging.getLogger(__name__)
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
GRAPH_API_URL = "https://graph.facebook.com/v21.0/me/messages"
REDIS_URL = os.getenv("REDIS_URL")
PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER= os.getenv("DB_USERNAME")
PG_PASS= os.getenv("DB_PASSWORD")

# Email configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMERGENCY_EMAIL_RECIPIENTS = os.getenv("EMERGENCY_EMAIL_RECIPIENTS", "evkingwind9218@gmail.com,thanhtam852009@gmail.com").split(",")
async def embedding(text):
    """Async wrapper for embedding to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedd.embed_query, text)

async def send_emergency_email(user_id: str, user_message: str, emotion_status: str, problem: str, urgency: str):
    """Gửi email thông báo khẩn cấp tới các giáo viên phụ trách"""
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.warning("⚠️ Email configuration not set. Skipping email notification.")
        return False
    
    try:
        # Tạo email content
        subject = f"🚨 CẢNH BÁO KHẨN CẤP - Học sinh cần hỗ trợ ngay lập tức"
        
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #dc3545; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ background-color: #f8f9fa; padding: 20px; margin-top: 20px; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; }}
                .info-row {{ margin: 10px 0; }}
                .label {{ font-weight: bold; color: #495057; }}
                .value {{ color: #212529; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🚨 CẢNH BÁO KHẨN CẤP</h2>
                    <p>Hệ thống AI Mimi phát hiện học sinh cần hỗ trợ tâm lý khẩn cấp</p>
                </div>
                
                <div class="content">
                    <h3>Thông tin học sinh:</h3>
                    <div class="info-row">
                        <span class="label">User ID:</span>
                        <span class="value">{user_id}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">Vấn đề đang gặp phải:</span>
                        <span class="value">{problem}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">Trạng thái cảm xúc:</span>
                        <span class="value">{emotion_status}</span>
                    </div>
                    <div class="info-row">
                        <span class="label">Mức độ khẩn cấp:</span>
                        <span class="value" style="color: #dc3545; font-weight: bold;">{urgency.upper()}</span>
                    </div>
                    
                    <div class="warning">
                        <h4>⚠️ Nội dung tin nhắn:</h4>
                        <p>"{user_message}"</p>
                    </div>
                    
                    <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 15px; margin: 15px 0;">
                        <h4>📋 Hành động cần thực hiện:</h4>
                        <ul>
                            <li>Liên hệ ngay với học sinh hoặc gia đình</li>
                            <li>Đánh giá mức độ rủi ro trực tiếp</li>
                            <li>Xem xét cần thiết can thiệp chuyên môn</li>
                            <li>Thông báo cho Ban Giám hiệu nếu cần</li>
                        </ul>
                    </div>
                    
                    <p style="color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                        <strong>Lưu ý:</strong> Email này được gửi tự động từ hệ thống AI Mimi. 
                        Vui lòng xử lý trong thời gian sớm nhất.
                    </p>
                </div>
                
                <div class="footer">
                    <p>Email này được gửi từ Hệ thống Trợ lý AI Mimi - THCS Gia Thanh</p>
                    <p>Thời gian: {asyncio.get_event_loop().time()}</p>
                    <p>Hotline hỗ trợ: Cô Thúy (0962186108) | Cô Trâm (0915266338)</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Tạo message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = SMTP_USERNAME
        message["To"] = ", ".join(EMERGENCY_EMAIL_RECIPIENTS)
        
        # Attach HTML content
        html_part = MIMEText(html_body, "html", "utf-8")
        message.attach(html_part)
        
        # Gửi email bất đồng bộ
        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            username=SMTP_USERNAME,
            password=SMTP_PASSWORD,
            start_tls=True,
        )
        
        logger.info(f"✅ Đã gửi email khẩn cấp cho user {user_id} tới {len(EMERGENCY_EMAIL_RECIPIENTS)} người nhận")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi gửi email khẩn cấp: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def get_user_information(state: State):
    """
        This node must be done first before get deep chat with person
        Short-term memory will be loaded all from long-term memory to gain insights
        
    """
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
    
async def update_cache(state: State):
    bot_plan = state["bot_plan"]
    is_new_user = state["conversation"].get("is_new_user", True)
    current_risk = state["risk"]
    current_emotion = state['user_emotion']
    user_id = state["conversation"]["user_id"]
    # Lấy tất cả messages từ state (đã được reducer giới hạn 5 messages)
    current_messages = state["conversation"]["messages"]
    redis = await get_redis_client()
    
    summary_data = await redis.get(f"summary:{user_id}")
    conversation_data = await redis.get(f"past:conversation:{user_id}")
    
    # Merge messages cũ và mới, giữ tối đa 5 messages gần nhất
    if conversation_data:
        try:
            conversation_json = json.loads(conversation_data)
            # Lấy messages cũ
            old_messages = conversation_json.get("messages", [])
            # Merge với messages mới
            all_messages = old_messages + current_messages
            # Chỉ giữ 5 messages gần nhất
            conversation_json["messages"] = all_messages[-5:]
        except json.JSONDecodeError:
            # Nếu parse lỗi, tạo mới với 5 messages gần nhất
            conversation_json = {"messages": current_messages[-5:]}
    else:
        # Không có history cũ, tạo mới với 5 messages gần nhất
        conversation_json = {"messages": current_messages[-5:]}

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
    await redis.set(f"current:state:{user_id}", json.dumps(state, ensure_ascii=False), ex=60*60)
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
        model="gpt-4.1-mini",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY")
    )
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
       ### TASK
        Analyze the emotional signal of the LATEST message provided below.

        ### INPUT
        - Message: "{messages}"
        - Previous Context Intense: "{prev_intense}"

        ### CLASSIFICATION GUIDELINES
        1. **Status**: joy, sadness, fear, disgust, anger, surprise, uncertain.
        - If message is vague (e.g., "hmm", "ok") -> status: "uncertain".
        - If greeting -> status: "joy".
        2. **Urgency**:
        - "normal": Casual conversation, venting without danger.
        - "watch": Deep sadness, anger, mentions of hopelessness.
        - "immediate": Explicit self-harm (cutting, suicide), violence, or clear crisis.
        3 **Problem (CRITICAL - MUST BE IN VIETNAMESE)**: 
           - Extract the specific subject, struggle, or topic the user is talking about in **VIETNAMESE** (Tiếng Việt).
           - Examples of valid problems:
                                                                                                             * "khó khăn trong vấn đề toán học","mất gốc toán hình","cảm thấy áp lực trong hbọc tập","áp lực thi cử", ..v.v (Academic)
             * "cảm thấy bị bạn bè tẩy chay", "gặp khó khăn trong kết nối với bạn bè", "khó khăn để hòa đồng", "khó khăn trong giao tiếp với phụ huynh", v.v (Social)
             * "cảm thấy cô đơn", "crush không tích", "cãi nhau với bố mẹ" (Emotional), v.v
             * "cần lời khuyên trong học tập", "cần sự hướng dẫn để trở nên thành công", v.v (Request)
           - ONLY use "MUST CLARIFY" if the message is purely a greeting (e.g. "Hello") or completely vague (e.g. "I'm sad" without reason).

        ### SAFETY RULES
        - Do NOT diagnose mental disorders.
        - If user mentions "die", "kill", "suicide", "hurt myself" -> Urgency MUST be "immediate".
        - Urgency cannot be lower than Previous Context Intense (unless topic changed).

        ### OUTPUT JSON
        {{
            "status": "...",
            "problem": "...",
            "metadata": {{ "trigger": "...", "context": "..." }},
            "self_harm": true/false,
            "violence": true/false,
            "urgency": "normal | watch | immediate",
            "confidence_score": 0.0 to 1.0
    }}
    """
    try:
        response = await llm.ainvoke(emotion_prompt)
        cleaned_content = strip_markdown_json(response.content)
        data = json.loads(cleaned_content)
        response_urgency = data.get("urgency", "normal")
        logger.info(f"✅ CAUGHT PROBLEM: {data.get('problem', '')}")
        if data.get("self_harm") or data.get("violence"):
            response_urgency = "immediate"
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
                "urgency": response_urgency
            },
            "confidence_score": float(data.get("confidence_score", 0.3) )
        }

    except Exception as e:
        logger.error(f"❌ [get_emotion] Error: {str(e)}")
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
    # ⚠️ KHÔNG save solution ở đây - chỉ save khi đã dùng thật trong generate_response
    return {
        **state,
        "bot_plan": {
            "solution": result["solution"],
            "must_not_do": result["must_not_do"],
            "tone": result["tone"]
        },
        "user_emotion":{
            **state["user_emotion"],
            "is_new_problem": False,
            "crisis_level": crisis_dict.get(str(result["level"]), "low"),
        },
        "risk": {
            "self_harm": result["self_harm"],
            "violence": result["violence"],
            "urgency": result.get("urgency", "normal")
        },
        "rag_meta": {
            "ignored": False,
            "similarity": similarity
        },
        "confidence_score": 1
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
        return {**state}
async def route_after_decision(state: State):
    urgency_res = await is_urgent(state)
    if urgency_res == "response_emergency":
        return "response_emergency"
    else: 
        plan_res = await should_rotate_plan(state)
        return plan_res
async def bot_planning(state: State):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
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
    logger.info(f"[bot_planning] PREVIOUS SUMM: {summary_data}")
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
        - Options: "warm_supportive", "calm_reassuring", "gentle_curious", "cheerful_encouraging", "serious_concerned", "playful_light", "encouraging_sister", "gentle_protective"

        3. **Must Not Do**: What should the bot absolutely avoid?
        - Be specific based on the emotional state and risks
        RETURN JSON ONLY.
   """
    try:
        response = await llm.ainvoke(plan_prompt)
        cleaned_content = strip_markdown_json(response.content)
        plan_data = json.loads(cleaned_content)
        logger.info(f"✅ RECEIVED [bot_planning]: {plan_data}")
        
        return {
            **state,
            "bot_plan": {
                "solution": plan_data.get("solution", "validate_feelings"),
                "tone": plan_data.get("tone", "warm_supportive"),
                "must_not_do": plan_data.get("must_not_do", []),
            }
        }
    except Exception as e:
        logger.error(f"[bot_planning] Error: {str(e)}")
        return {
            **state,
            "bot_plan": {
                "solution": "validate_feelings",
                "tone": "warm_supportive",
                "must_not_do": ["dismiss feelings", "give medical advice", "make promises"]
            }
        }
async def should_rotate_plan(state: State):
    bot_plan = state.get("bot_plan", None)
    user_id = state["conversation"]["user_id"]
    redis = await get_redis_client()
    key = await redis.get(f"summary:{user_id}")
    response = json.loads(key)
    print("✅[SHOULD ROTATE PLAN] RECEIVED",bot_plan)
    if bot_plan is None: #or response["status"] == "changed":
        return "bot_planning"
    else:
        return "gen_response"
async def decide_next_step(state: State):
    redis = await get_redis_client()
    crisis_level = state["user_emotion"].get("crisis_level","low")
    user_id = state["conversation"]["user_id"]
    urgency = state["risk"]["urgency"]
    confidence_score = state["confidence_score"]
    self_harm = state["risk"]["self_harm"]
    violence = state["risk"]["violence"]
    problem = state["user_emotion"]["problem"]
    emotion_status = state["user_emotion"]["status"]
    summary_data = json.loads(await redis.get(f"summary:{user_id}"))
    if summary_data.get("intense") == "growing" and problem:
        return {**state, "next_step": "guide"}
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
    Lấy state từ Redis và cập nhật nếu là lần thứ 2 người dùng chạy.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2, 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    redis = await get_redis_client()
    user_id = state["conversation"]["user_id"]
    messages = state["conversation"]["messages"]
    
    # ===== Lấy state cũ từ Redis (nếu có) =====
    previous_state_data = await redis.get(f"current:state:{user_id}")
    is_returning_user = False
    
    if previous_state_data:
        try:
            previous_state = json.loads(previous_state_data)
            is_returning_user = True
            logger.info(f"[summary_conv_history] ✅ Tìm thấy state cũ cho user {user_id}")
            
            # Merge các thông tin từ state cũ nếu cần thiết
            if "user_emotion" in previous_state and not state.get("user_emotion"):
                state["user_emotion"] = previous_state["user_emotion"]
            
            if "risk" in previous_state and not state.get("risk"):
                state["risk"] = previous_state["risk"]
            
            if "bot_plan" in previous_state and not state.get("bot_plan"):
                state["bot_plan"] = previous_state["bot_plan"]
                
            logger.info(f"[summary_conv_history] 🔄 Đã merge thông tin từ state cũ")
        except json.JSONDecodeError as e:
            logger.warning(f"[summary_conv_history] ⚠️ Không parse được state cũ: {e}")
            is_returning_user = False
    else:
        logger.info(f"[summary_conv_history] ℹ️ User {user_id} lần đầu tiên hoặc không có state cũ")
    
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
    logger.info(f"[summary_conv_history] 📨 Recent user messages: {recent_msgs}")
    logger.info(f"[summary_conv_history] 📝 Current messages: {lastest_message}")
    logger.info(f"[summary_conv_history] 🔢 Is returning user: {is_returning_user}")
    last_summary = await redis.get(f"summary:{user_id}")
    prev_intense = None
    if last_summary:
        try:
            prev = json.loads(last_summary)
            prev_intense = prev.get("intense")
        except:
            prev_intense = None

    new_summary_prompt = f"""
    ### SYSTEM ROLE
    You are the MEMORY MANAGER. Update the conversation summary based on logic below.

    ### INPUT
    - Previous Summary: {json.dumps(last_summary, ensure_ascii=False) if last_summary else "None"}
    - User Message: "{lastest_message}"
    
    ### LOGIC FOR "STATUS" & "INTENSE" (CRITICAL)
    
    1. **CONTINUING (Most Common)**:
       - User answers a previous question (e.g., Bot: "Which part?", User: "Trigonometry").
       - User gives more details on the SAME broad topic (e.g., Math -> Geometry -> Trigonometry).
       - User is venting/complaining about the SAME issue.
       -> ACTION: Set status = "continuing". Increase 'intense' (starting -> growing).

    2. **CHANGED**:
       - User explicitly stops the current topic (e.g., "Enough about math, let's talk about games").
       - User starts a completely UNRELATED topic.
       -> ACTION: Set status = "changed". Reset 'intense' = "starting".

    ### LOGIC RULES
    1. IF Previous Summary is NULL -> Status = "changed", Intense = "starting".
    2. IF User switches topic -> Status = "changed", Intense = "starting".
    3. IF User continues topic -> Status = "continuing".
       - AND user provides more detail -> Intense = "growing".
       - AND user screams/cries/demands -> Intense = "request_high_urgency".
    4. **IMPORTANT**: Intense can NEVER decrease within the same topic.

    ### OUTPUT JSON
    {{
        "reasoning": "Explain why it is CONTINUING or CHANGED. Did user answer a question?",
        "intent": "Update intent (e.g., seeking help with trigonometry)",
        "main_topic": "...",
        "intense": "starting | growing | request_high_urgency",
        "status": "continuing | changed",
        "key_context": ["User struggles with Trigonometry basics"],
        "confidence_score": 0.95
    }}
    """
    try:
        response = await llm.ainvoke(new_summary_prompt)
        logger.debug(f"[summary_conv_history] Response: {response.content}")
        cleaned_content = strip_markdown_json(response.content)
        summary_data = json.loads(cleaned_content)
        
        # Lưu summary mới vào Redis
        summary_to_save = {
            "intent": summary_data.get("intent",""),
            "intense": summary_data.get("intense",""),
            "main_topic": summary_data.get("main_topic",""),
            "status": summary_data.get("status","continuing"),
            "key_context": summary_data.get("key_context",[]),
            "is_returning_user": is_returning_user
        }
        
        await redis.set(
            f"summary:{user_id}",
            json.dumps(summary_to_save, ensure_ascii=False),
            ex=60*60
        )
        
        # Cập nhật state mới với thông tin returning user
        updated_state = {
            **state,
            "confidence_score": summary_data.get("confidence_score", 0.3)
        }
        
        # Nếu là returning user, thêm flag vào conversation
        if is_returning_user:
            updated_state["conversation"]["is_new_user"] = False
        
        logger.info(f"[summary_conv_history] ✅ Đã cập nhật summary cho user {user_id}")
        logger.info(f"[summary_conv_history] 📊 Status: {summary_data.get('status')}, Intense: {summary_data.get('intense')}")
        
        return updated_state
    except Exception as e:
        logger.error(f"[summary_conv_history] Error: {str(e)}")
        logger.error(f"[summary_conv_history] Error type: {type(e).__name__}")
        logger.error(f"[summary_conv_history] Error details: {repr(e)}")
        logger.error(f"[summary_conv_history] Traceback:\n{traceback.format_exc()}")
        return state
async def is_information_loaded(state: State):
    user_id = state["conversation"]["user_id"]
    redis = await get_redis_client() 
    if await redis.get(f"memory:{user_id}:preferences"):
        return "should_get_emotion"
    else:
        return "get_user_information" 
async def should_get_emotion(state: State):
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
        logger.error(f"Lỗi tại node should_get_emotion: {str(e)}")
        return "get_emotion"
async def generate_response(state: State):
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
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
    
    
    # Get RAG solution (full text từ database, không phải strategy name)
    rag_solution = bot_plan.get("solution", "")
    
    # Get last solution đã dùng (để tránh lặp lại)
    last_solution_bytes = await redis.get(f"rag:solution:{user_id}")
    last_solution = last_solution_bytes if last_solution_bytes else None
    
    # Nếu solution này đã dùng rồi, bỏ qua (để LLM tự sáng tạo)
    if last_solution and rag_solution and last_solution.strip() == rag_solution.strip():
        rag_solution = ""  # Để trống để LLM tự nghĩ

    if next_step == "clarify":
        system_instruction = """Bạn là Mimi - trợ lý ảo thân thiện của trường THCS Gia Thanh.
        Nhiệm vụ: Đặt câu hỏi MỞ để hiểu rõ hơn vấn đề em.

        Quy tắc:
        - Dùng giọng điệu tò mò, thân thiện
        - Hỏi 1-2 câu ngắn gọn
        - Không giả định vấn đề
        - Không áp đặt cảm xúc
        - Dùng emoji phù hợp DỰA TRÊN CẢM XÚC hiện tại của EM.

        Ví dụ tốt:
        "Mình có thể kể thêm về chuyện đó được không? 🤔"
        "Em cảm thấy như thế nào về việc này? 💭"
    """

    elif next_step == "guide":
        system_instruction = f"""You are Mimi, a supportive, empathetic, and gentle older sister (big sister figure) for a middle school student (10-15 years old).
    ### CÁCH TRẢ LỜI
        - Đừng chỉ copy-paste bí kíp. Hãy biến nó thành lời thủ thỉ.
        - Thay vì nói: "Bước 1 là quan sát", hãy nói: "Mẹo nhỏ nè, em thử nghía xem bạn ấy đang cầm món đồ gì hay hay không..."
        - Luôn kết thúc bằng một câu hỏi kích thích hành động nhỏ: "Mai em có dám thử không?", "Em thấy chiêu này có khả thi không?"
    ### CURRENT SITUATION
    - User Emotion: {emotion_status} (Crisis Level: {crisis_level})
    - Problem: {problem}
    - Strategy: {solution}
    - Tone: {tone}
    ### REMEMBER
    NHIỆM VỤ:
        1. Đừng hỏi "phần nào" nữa nếu user nói "không hiểu gì cả".
        2. Hãy đề xuất một bước nhỏ nhất, dễ nhất.### LOGIC RULES
    1. IF Previous Summary is NULL -> Status = "changed", Intense = "starting".
    2. IF User switches topic -> Status = "changed", Intense = "starting".
    3. IF User continues topic -> Status = "continuing".
       - AND user provides more detail -> Intense = "growing".
       - AND user screams/cries/demands -> Intense = "request_high_urgency".
    4. **IMPORTANT**: Intense can NEVER decrease within the same topic.
    ### INSTRUCTIONS
    1. **Style**: Use natural Vietnamese for Gen Z/Alpha. Use words like "nhỉ", "nè", "ui", "đâu á", "thương ghê". 
    2. **Structure**:
       - Step 1: Validate feelings (e.g., "Nghe chuyện này Mimi thấy thương em quá/...").
       - Step 2: Gentle advice (based on Strategy).
       - Step 3: Check-in (e.g., "Em thấy sao về ý này?").
    3. **Constraints**:
       - MAX 3-4 sentences. Keep it short like a chat message.
       - Use 1-2 soft emojis (🌿, ☁️, ✨, 💙). Avoid "loud" emojis (😂, 🤣) unless the user is happy.
       - NEVER sound like a textbook or a robot.
       - {f"AVOID: {', '.join(must_not_do)}" if must_not_do else ""}
    """

    else: 
        system_instruction = f"""Bạn là Mimi - trợ lý ảo đồng hành cùng học sinh trường THCS Gia Thanh.
        Nhiệm vụ: Lắng nghe, thấu hiểu, và đồng hành cùng em.

        Cảm xúc của em: {emotion_status}
        Vấn đề: {problem if problem else "Chưa rõ"}

        Phương pháp: {solution}
        Giọng điệu: {tone}

        ###KHÔNG được làm:
        {chr(10).join(f"- {item}" for item in must_not_do) if must_not_do else "- Phán xét, đưa ra lời khuyên y tế"}
        ### VARIETY RULES
        - KHÔNG lặp lại câu mở đầu giống hệt tin nhắn trước.
        - Nếu tin nhắn trước đã "Validate feelings" rồi, tin nhắn này hãy đi thẳng vào vấn đề.
        - Đừng lúc nào cũng "Mimi thấy thương em quá". Hãy dùng: "Chà, ca này khó nhỉ", "Mimi hiểu mà", "Cố lên nhé", v.v.
        ###Quy tắc phản hồi:
        - Phản ánh cảm xúc em đang trải qua
        - Xác nhận cảm xúc của em là hợp lý
        - Đặt 1 câu hỏi mở để em chia sẻ thêm
        - Giữ giọng điệu ấm áp, không áp đặt
        - Độ dài: 2-4 câu ngắn
        - Emoji: 1-2 emoji nhẹ nhàng
    """
    
    generation_prompt = f"""{system_instruction}
### SOURCE KNOWLEDGE
    Nếu có thông tin dưới đây, hãy DÙNG NÓ để đưa ra lời khuyên cụ thể, đừng trả lời chung chung:
    [Kiến thức chuyên gia]: {rag_solution}
**NGỮ CẢNH CUỘC TRÒ CHUYỆN:**
    {json.dumps(summary_data, ensure_ascii=False, indent=2) if summary_data else "Cuộc trò chuyện mới"}

**TIN NHẮN CỦA EM:**
    "{user_message}"

---

**YÊU CẦU CUỐI:**
- Gọi em bằng "em" hoặc "bạn"
- Tự xưng là "Mimi" hoặc "mình"
- Phù hợp lứa tuổi 10-16
- Ngôn ngữ Tiếng Việt tự nhiên
- Tránh thuật ngữ phức tạp
- GIỮ NGẮN GỌN NHƯNG PHẢI SÚC TÍCH

FINALLY: RETURN EXACT JSON FORMAT:
{{
    "response": ""
}}
RETURN JSON ONLY."""
    
    try:
        response = await llm.ainvoke(generation_prompt)
        cleaned_content = strip_markdown_json(response.content)
        data = json.loads(cleaned_content)
        
        # ✅ Lưu solution vừa dùng để tránh lặp lại lần sau
        if rag_solution:  # Chỉ save nếu có dùng RAG solution
            await redis.set(f"rag:solution:{user_id}", rag_solution, ex=5*60)

        return {
            **state,
            "response": {
                "output": data.get("response", "Xin lỗi, Mimi hiện không thể hỗ trợ bạn lúc này!")
            }
        }

    except Exception as e:
        logger.error(f"[generate_response] Error: {str(e)}")
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
async def response_and_update(state: State):
    update_mem_task = update_cache(state)
    response_task = generate_response(state)
    _, response_state = await asyncio.gather(update_mem_task, response_task)
    return response_state
async def response_emergency(state: State):
    """Node xử lý tình huống khẩn cấp - gửi response và thông báo email"""
    
    # Lấy thông tin từ state
    user_id = state["conversation"]["user_id"]
    user_message = state["conversation"]["messages"][-1]["content"]
    emotion_status = state["user_emotion"]["status"]
    problem = state["user_emotion"]["problem"]
    urgency = state["risk"]["urgency"]
    
    # Gửi email thông báo khẩn cấp (không chờ kết quả để không block workflow)
    asyncio.create_task(
        send_emergency_email(
            user_id=user_id,
            user_message=user_message,
            emotion_status=emotion_status,
            problem=problem,
            urgency=urgency
        )
    )
    
    logger.warning(f"🚨 EMERGENCY triggered for user {user_id} - Email notification sent")
    
    return {
        **state,
        "response":{
            "output":"""
Mình rất tiếc khi nghe bạn đang trải qua điều này 💔
Có vẻ như lúc này bạn đang cảm thấy rất khó khăn, và cảm giác đó hoàn toàn không sai.

Bạn không cần phải đối mặt với chuyện này một mình. Nếu có thể, bạn hãy thử chia sẻ với người lớn mà bạn tin tưởng như bố mẹ, thầy cô, hoặc người thân nhé.
Mình có thể giúp bạn liên hệ với:
- *cô Thúy (hotline: 0962186108)* 
- *cô Trâm (hotline: 0915266338)* 
- *Tổng đài Quốc gia Bảo vệ Trẻ em (111)*
để chúng ta cùng nhau vượt qua vấn đề của bạn nhé!

Điều quan trọng nhất là: *bạn xứng đáng được lắng nghe và được giúp đỡ* 🌱
Mimi vẫn ở đây để lắng nghe bạn.
            """
        }
    }
