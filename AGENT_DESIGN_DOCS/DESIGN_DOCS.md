Kế Hoạch Triển Khai Kỹ Thuật — Teen Counseling Chatbot MVP (v2)
1. Tổng Quan Kiến Trúc
                        ┌─────────────────────────────────┐
                        │         Web App (React)         │
                        │  [Chat UI học sinh] [UI giáo viên]│
                        └──────────────┬──────────────────┘
                                       │ WebSocket
                        ┌──────────────▼──────────────────┐
                        │        FastAPI Backend          │
                        │   /chat (WS) · /admin (WS)      │
                        └──┬───────────┬─────────────────┘
                           │           │
              ┌────────────▼──┐   ┌────▼──────────────┐
              │  LangGraph    │   │   Redis (Upstash)  │
              │  Workflow     │   │  session · takeover│
              └────────┬──────┘   └───────────────────┘
                       │
              ┌────────▼──────────────────────────────┐
              │         PostgreSQL — Supabase          │
              │  psychology_kb (pgvector)              │
              │  user_memory_kb (pgvector, per-user)   │
              └───────────────────────────────────────┘
Real-time layer: WebSocket cho cả học sinh lẫn giáo viên, dùng Redis Pub/Sub làm message broker giữa các connection.

2. State Definition
pythonclass CounselingState(TypedDict):
    messages: Annotated[list, add_messages]   # Sliding window 20 messages gần nhất
    user_id: str

    # Context
    current_topic: str           # Chủ đề đang thảo luận, cập nhật liên tục

    # Luồng xử lý
    intent_category: str         # "out_of_scope" | "simple" | "complex"
    info_gap_status: str         # "missing" | "sufficient"

    # RAG — 2 nguồn
    kb_guidelines: str           # Từ psychology_kb (kiến thức chung)
    user_memory: str             # Từ user_memory_kb (hồ sơ học sinh này)

    # Reasoning
    reasoning_scratchpad: list[str]

    # HITL
    is_emergency: bool
    human_takeover: bool

3. Nodes, Edges & Tools
Sơ Đồ Luồng
[START]
   │
   ▼
[load_memory_node]
   │
   ├─ human_takeover == True ──────────────────────────────► [END]
   │
   ▼
[safety_gateway_node]
   │
   ├─ is_emergency == True ──► [handoff_human_node] ────────► [END]
   │
   ▼
[intent_routing_node]
   │
   ├─ "out_of_scope" ──► [out_of_scope_node] ──► [save_memory_node] ► [END]
   ├─ "simple"       ──► [simple_response_node] ► [save_memory_node] ► [END]
   └─ "complex"      ──► [info_gap_assessor_node]
                               │
                 ┌─────────────┴──────────────┐
            "missing"                   "sufficient"
                 │                            │
                 ▼                            ▼
   [clarification_node]         [deep_reasoning_node]
                 │               (CoT + ReAct + RAG)
                 └──────────────┬─────────────┘
                                ▼
                       [save_memory_node]
                                │
                              [END]

Chi Tiết Từng Node
Node 1: load_memory_node
python"""
Làm 3 việc theo thứ tự:

1. Kiểm tra Redis key: chat:{user_id}:takeover
   → Nếu "true": set human_takeover=True, return ngay (graph END)

2. Tải lịch sử chat gần nhất từ Redis:
   chat:{user_id}:history → last 20 messages

3. Tải user_memory từ Supabase (user_memory_kb):
   - Embed câu hỏi hiện tại
   - Query top-3 chunks liên quan đến user_id này
   - Đây là "hồ sơ dài hạn" của học sinh: các vấn đề cũ, tính cách, hoàn cảnh
"""
Redis keys layout:
chat:{user_id}:history   → JSON list[{role, content}]   TTL: 7 ngày
chat:{user_id}:topic     → string                        TTL: 7 ngày
chat:{user_id}:takeover  → "true"/"false"                TTL: 30 ngày

Node 2: safety_gateway_node
python"""
Gọi LLM (gpt-4o-mini, fast) với prompt cực kỳ focused:

System: "Bạn là bộ lọc an toàn cho chatbot tư vấn học sinh Việt Nam.
         Chỉ trả lời YES hoặc NO."

User: "Tin nhắn sau có chứa dấu hiệu học sinh muốn tự làm hại
       bản thân hoặc người khác không?
       Tin nhắn: {last_message}"

→ YES → is_emergency = True
→ NO  → is_emergency = False

Không dùng model lớn ở đây — ưu tiên tốc độ và độ chính xác binary.
"""

Node 3: handoff_human_node
python"""
Thực hiện 3 bước:

1. Sinh phản hồi an ủi + hotline:
   "Mình rất lo cho bạn khi nghe điều này. Bạn không đơn độc đâu nhé.
    Nếu bạn đang cần giúp đỡ ngay, hãy gọi đường dây 1800 599 920
    (miễn phí, 24/7). Mình cũng đang nhờ người hỗ trợ bạn ngay bây giờ."

2. Gọi Tool: notify_human_admin
   → Push notification / email tới giáo viên trực
   → Payload: {user_id, tóm tắt 3 messages cuối, mức độ: KHẨN CẤP}

3. Set Redis: chat:{user_id}:takeover = "true"
   → Mọi message sau từ học sinh đều được forward thẳng tới giáo viên
   → Bot hoàn toàn im lặng
"""
Tool: notify_human_admin
pythondef notify_human_admin(user_id: str, summary: str) -> bool:
    # 1. Gửi email qua Resend API
    # 2. Push lên Redis channel: admin:alerts → giáo viên đang online nhận ngay
    # 3. Ghi log vào Supabase bảng emergency_logs
    pass

Node 4: intent_routing_node
python"""
Gọi LLM với context: lịch sử + tin nhắn mới + current_topic

Phân loại đồng thời 2 việc:

[A] Intent:
  - out_of_scope: giải bài, code, hỏi thông tin không liên quan tâm lý
  - simple: chào hỏi, xã giao, cảm xúc thoáng qua ("hôm nay mệt quá")
  - complex: vấn đề cần tư vấn sâu (áp lực, cô đơn, gia đình, định hướng)

[B] Topic update:
  - Nhìn vào 3-5 messages gần nhất để detect xem chủ đề có thay đổi không
  - Cập nhật current_topic vào State
  - VD: "áp lực thi cử" → "mâu thuẫn với bạn bè"

Output JSON: {"intent": "complex", "topic": "áp lực từ bố mẹ"}
"""

Node 5: out_of_scope_node
python"""
Phản hồi thân thiện, không phán xét, redirect nhẹ nhàng.

VD: "Ồ chuyện đó thì mình không giỏi lắm 😅
     Nhưng nếu bạn có điều gì đang băn khoăn trong lòng,
     mình luôn ở đây để lắng nghe nha!"

Không được nói cứng nhắc như "tôi chỉ hỗ trợ vấn đề tâm lý".
"""

Node 6: simple_response_node
python"""
Dùng: messages + current_topic + user_memory (để cá nhân hóa)

Không cần RAG psychology_kb — vấn đề đơn giản, phản hồi tự nhiên.

VD học sinh: "Hôm nay mệt quá"
Bot: "Ôi, mệt kiểu nào vậy bạn — mệt người hay mệt đầu óc?
      (biết user_memory: hay bị áp lực thi → hỏi thêm thay vì giả định)"
"""

Node 7: info_gap_assessor_node
python"""
Dùng Chain of Thought để đánh giá:

Scratchpad:
  Thought 1: "Vấn đề học sinh đang nêu là gì?"
             → "Em nói buồn và không muốn đi học"
  Thought 2: "Mình đã biết gì?"
             → "Cảm xúc: buồn. Hành vi: né tránh học."
  Thought 3: "Còn thiếu gì để tư vấn hiệu quả?"
             → "Chưa biết nguyên nhân: do bạn bè, thầy cô, hay gia đình?"
  Verdict: "missing"

Tiêu chí "sufficient" — ĐẠT KHI CÓ ÍT NHẤT:
  ✓ Hiểu cảm xúc cốt lõi
  ✓ Biết ít nhất 1 nguyên nhân hoặc bối cảnh
  (Không cần hoàn hảo — đủ để cho lời khuyên có ích)

Ghi toàn bộ scratchpad vào State để deep_reasoning_node tái sử dụng.
"""

Node 8: clarification_node
python"""
Đọc reasoning_scratchpad → biết đang thiếu gì → hỏi đúng điểm đó.

Rules:
  - Chỉ hỏi 1 câu mỗi lần
  - Câu hỏi mở, không dẫn dắt
  - Tone như người bạn, không như bác sĩ

VD: "Bạn kể thêm cho mình nghe được không —
     chuyện đó liên quan đến ai không?"
"""

Node 9: deep_reasoning_node
python"""
ReAct loop với 2 công cụ RAG:

--- Vòng lặp ---

Thought: "Dựa trên scratchpad, vấn đề cốt lõi là [X].
          Cần tìm phương pháp tâm lý phù hợp cho lứa tuổi teen Việt Nam."

Action 1: search_psychology_kb(query="[X] thanh thiếu niên")
Observation 1: [kb_guidelines — phương pháp chuẩn từ KB chung]

Action 2: search_user_memory(user_id, query="[X]")
Observation 2: [user_memory — bối cảnh riêng của học sinh này]

Answer: Tổng hợp → viết phản hồi
  - Nội dung: dựa trên kb_guidelines (không bịa)
  - Cá nhân hóa: dùng user_memory (biết hoàn cảnh em)
  - Tone: nhẹ nhàng, tuổi teen, tiếng Việt tự nhiên (bao gồm tiếng lóng phù hợp)
  - Format: KHÔNG liệt kê bullet. Viết như đang nói chuyện.
"""
Tool: search_psychology_kb
pythondef search_psychology_kb(query: str, top_k: int = 3) -> str:
    """
    1. Embed query → OpenAI text-embedding-3-small
    2. pgvector cosine similarity search trên bảng psychology_kb
    3. Return top_k chunks dạng plain text
    """
Tool: search_user_memory
pythondef search_user_memory(user_id: str, query: str, top_k: int = 3) -> str:
    """
    1. Embed query
    2. pgvector search trên bảng user_memory_kb, filter WHERE user_id = ?
    3. Return các đoạn liên quan từ lịch sử dài hạn của học sinh
    """

Node 10: save_memory_node
python"""
Làm 2 việc:

[A] Redis — cập nhật short-term:
  - Append messages mới vào chat:{user_id}:history
  - Giữ sliding window 20 messages (xóa cũ hơn)
  - Ghi current_topic mới

[B] Supabase — cập nhật long-term (async, không block response):
  - Nếu conversation chứa thông tin mới về học sinh
    (hoàn cảnh, tính cách, vấn đề recurring)
    → Chunk + embed → upsert vào user_memory_kb
  - Dùng LLM nhỏ để detect xem có info đáng lưu không
    ("Em hay bị bố mẹ so sánh với anh trai" → đáng lưu)
    ("Ừ, mình hiểu rồi" → không cần lưu)
"""

4. HITL Real-time Architecture
Đây là phần phức tạp nhất — thiết kế chi tiết để tránh over-engineer:
Học sinh                 FastAPI Server              Giáo viên
    │                         │                          │
    │──── WS connect ─────────►│                          │
    │                         │◄─── WS connect (admin) ──│
    │                         │                          │
    │                    Redis Pub/Sub                   │
    │                  channel: room:{user_id}           │
    │                         │                          │
    │──── message ────────────►│                          │
    │            [Bot đang active]                        │
    │            LangGraph xử lý                         │
    │◄──── bot response ───────│                          │
    │                         │──── forward to admin ───►│
    │                         │    (read-only mode)       │
    │                         │                          │
    │         [Emergency → human_takeover = True]        │
    │                         │                          │
    │──── message ────────────►│                          │
    │            [Bot im lặng]                            │
    │                         │──── forward to teacher ─►│
    │◄──────────────────────── teacher response ──────────│
Giáo viên có 2 mode:

Observer mode (mặc định): nhận bản sao toàn bộ conversation, không can thiệp
Active mode (sau emergency): nhận message từ học sinh, reply trực tiếp

Redis Pub/Sub implementation:
python# Khi học sinh gửi message
await redis.publish(f"room:{user_id}", json.dumps({
    "from": "student",
    "content": message,
    "bot_active": not state["human_takeover"]
}))

# Giáo viên subscribe
await redis.subscribe(f"room:{user_id}")

5. Psychology KB Pipeline
Nguồn dữ liệu & Xử lý
Nguồn 1: Website crawl          Nguồn 2: .md files biên soạn
   (beautifulsoup / scrapy)         (đã sẵn sàng)
         │                               │
         ▼                               ▼
   [Cleaning & Chunking]          [Chunking trực tiếp]
   chunk_size=500 tokens          chunk_overlap=50 tokens
         │                               │
         └──────────────┬────────────────┘
                        ▼
              [OpenAI Embedding]
           text-embedding-3-small
                        │
                        ▼
              [Supabase pgvector]
              bảng psychology_kb
Script seed KB (scripts/seed_kb.py):
python"""
CLI tool chạy một lần để nạp dữ liệu ban đầu.

Usage:
  python seed_kb.py --source website --url https://example.com
  python seed_kb.py --source md --path ./docs/

Steps:
  1. Parse & clean content
  2. Chunk (RecursiveCharacterTextSplitter)
  3. Batch embed (OpenAI, batch size 100)
  4. Upsert vào Supabase
"""
Database Schema
sql-- Psychology KB chung
CREATE TABLE psychology_kb (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic       TEXT,
    content     TEXT,
    embedding   VECTOR(1536),
    source_url  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Memory dài hạn per-user
CREATE TABLE user_memory_kb (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    content     TEXT,           -- VD: "Hay bị bố mẹ so sánh với anh trai"
    embedding   VECTOR(1536),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Log khẩn cấp
CREATE TABLE emergency_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    summary     TEXT,
    handled_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Index vector search
CREATE INDEX ON psychology_kb
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX ON user_memory_kb
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON user_memory_kb (user_id);

6. Cấu Trúc Thư Mục
teen-counseling-bot/
├── graph/
│   ├── state.py            # CounselingState
│   ├── nodes.py            # Tất cả node functions
│   ├── tools.py            # search_psychology_kb, search_user_memory,
│   │                       # notify_human_admin
│   ├── edges.py            # Conditional edge functions
│   └── workflow.py         # Compile & cache LangGraph graph
├── memory/
│   ├── redis_client.py     # Upstash wrapper (history, takeover, pubsub)
│   └── supabase_client.py  # pgvector search + upsert
├── prompts/                # Prompt templates tách riêng để dễ iterate
│   ├── safety.py
│   ├── intent.py
│   ├── info_gap.py
│   └── deep_reasoning.py
├── api/
│   ├── main.py             # FastAPI app
│   ├── chat_ws.py          # WebSocket endpoint học sinh (/ws/chat/{user_id})
│   └── admin_ws.py         # WebSocket endpoint giáo viên (/ws/admin/{user_id})
├── scripts/
│   └── seed_kb.py          # CLI seed psychology KB
└── .env