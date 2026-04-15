"""Architecture Documentation - Modularized Structure

This document explains the new modularized architecture following Karpathy guidelines
(simple, surgical changes, no over-engineering).

## Directory Structure (per DESIGN_DOCS)

```
teen-counseling-bot/
├── api/                     # FastAPI application layer (NEW)
│   ├── __init__.py
│   ├── main.py              # FastAPI app with health checks
│   ├── chat_ws.py           # Student WebSocket endpoint
│   └── admin_ws.py          # Teacher supervision WebSocket
│
├── memory/                  # Memory layer modularization (NEW)
│   ├── __init__.py          # Unified exports
│   ├── redis_client.py      # Redis operations (short-term)
│   └── supabase_client.py   # PostgreSQL/Supabase operations (long-term)
│
├── prompts/                 # Prompts modularization (NEW)
│   ├── __init__.py          # Unified exports
│   ├── safety.py            # Safety check prompt
│   ├── intent.py            # Intent routing prompt
│   ├── simple_response.py   # Simple response + out-of-scope
│   ├── info_gap.py          # Info gap assessment
│   ├── clarification.py     # Clarification question prompt
│   ├── deep_reasoning.py    # Deep reasoning prompts
│   └── save_memory.py       # Memory gating prompt
│
├── graph/                   # LangGraph workflow (REFACTORED)
│   ├── state.py             # CounselingState definition
│   ├── nodes.py             # All 10 node implementations (now using prompts + memory)
│   ├── tools.py             # LLM helpers only (get_llm, latest_user_message)
│   ├── edges.py             # Conditional edge routing logic
│   └── workflow.py          # StateGraph assembly
│
├── utils/                   # Utilities (REFACTORED)
│   ├── database.py          # Connection pooling
│   ├── embeddings.py        # Embedding utilities (NEW - extracted from tools.py)
│   └── client.py
│
├── scripts/                 # Maintenance scripts
│   ├── validate_imports.py  # (NEW) Import validation
│   ├── seed_kb.py           # KB seeding (TODO)
│   ├── bootstrap_supabase_schema.py
│   ├── test_supabase_connection.py
│   └── test_chat_sample.py
│
├── main.py                  # Legacy CLI (for local testing)
├── workflow.py              # Legacy workflow
├── requirements.txt
└── docker-compose.yml
```

## Key Changes

### 1. Memory Layer Modularization

**Before**: All Redis/PostgreSQL operations scattered in `graph/tools.py`

**After**: Organized into two focused modules:

- **memory/redis_client.py**: Short-term memory (chat history, topics, takeover flags)
  - `load_history()`, `save_history()`
  - `load_topic()`, `save_topic()`
  - `load_takeover_flag()`, `set_takeover_flag()`
  - `publish_admin_alert()` - Real-time notifications via Redis Pub/Sub

- **memory/supabase_client.py**: Long-term memory and KB operations
  - `search_psychology_kb()` - General knowledge base search
  - `search_user_memory_kb()` - User-specific memory search
  - `upsert_user_memory_chunk()` - Store long-term memory
  - `update_user_longterm_style()` - OCEAN personality tracking
  - `notify_human_admin()` - Comprehensive emergency notification
  - `log_emergency()`, `send_emergency_email()` - Emergency handling

**Benefit**: Clean separation of concerns, easier to swap implementations (e.g., different cache backends)

### 2. Prompts Modularization

**Before**: Prompts hardcoded inline in `graph/nodes.py` (100+ lines of strings)

**After**: Each prompt in its own module for maintainability:

- **prompts/safety.py**: Safety classification prompt
- **prompts/intent.py**: Intent routing + topic update
- **prompts/simple_response.py**: System prompts for simple responses + out-of-scope
- **prompts/info_gap.py**: Info gap assessment with scratchpad
- **prompts/clarification.py**: Single-question clarification
- **prompts/deep_reasoning.py**: System + user prompts for KB-integrated reasoning
- **prompts/save_memory.py**: Memory gating logic

**Benefit**: Prompts now version-controllable, easy to A/B test, iterate without touching graph logic

### 3. API Layer (Real-time WebSocket)

**New in api/**:

- **api/main.py**: FastAPI application with health checks and lifecycle management
- **api/chat_ws.py**: Student WebSocket endpoint (`/ws/chat/{user_id}`)
  - Accepts student messages
  - Runs LangGraph workflow
  - Publishes to admin channel
  - Detects emergency and initiates takeover
  
- **api/admin_ws.py**: Teacher supervision WebSocket (`/ws/admin/{teacher_id}`)
  - Real-time alert subscription
  - Commands: `subscribe`, `unsubscribe`, `takeover`, `release`, `send_message`
  - Room-based filtering (only see subscribed students)

**Benefit**: Real-time bidirectional communication for students + supervised teacher oversight

### 4. Utilities Extraction

**New in utils/embeddings.py**: (extracted from graph/tools.py)
- `get_embeddings()` - Singleton OpenAI embeddings instance
- `embed_text()` - Async embedding for user messages/KB chunks
- `vector_literal()` - Format float vectors for pgvector

**Benefit**: Eliminates circular imports (memory/supabase_client needs embeddings, but was in graph/tools)

### 5. Graph/Nodes Refactoring

**Changes in graph/nodes.py**:
- Imports prompts via `from prompts import ...`
- Imports memory via `from memory import ...`
- Cleaner node implementations without inline strings
- Uses `get_*_prompt()` functions to get current prompt versions

**Example before**:
```python
async def safety_gateway_node(state):
    prompt = (
        "Ban la bo loc an toan... "
        "Chi tra loi YES hoac NO..."
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
```

**Example after**:
```python
async def safety_gateway_node(state):
    prompt = get_safety_check_prompt(user_text)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
```

## Testing & Validation

1. **Import Validation**: `python -m scripts.validate_imports`
   - Checks all layers can be imported without circular dependencies
   - ✅ All 6 modules pass (memory, prompts, utils.embeddings, graph.tools, graph.nodes, api.main)

2. **Unit Tests**: `pytest tests/test_counseling_flows.py -v`
   - ✅ 4/4 flow scenarios pass (emergency, simple, complex_missing, complex_sufficient)
   - Confirms business logic unchanged despite refactoring

3. **Integration Tests** (Next phase):
   - WebSocket handshake for student + teacher
   - Real-time message flow via Redis Pub/Sub
   - Admin alert delivery
   - Takeover logic

## Running the Server

```bash
# Start API server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints**:
- `GET /health` - Health check
- `GET /` - API info
- `WS /ws/chat/{user_id}` - Student chat
- `WS /ws/admin/{teacher_id}` - Teacher oversight
- `GET /ws/active-students` - Monitor active students
- `GET /ws/admin/active-teachers` - Monitor active teachers

## Backward Compatibility

- **Legacy CLI (`main.py`)**: Still works for local testing
- **Legacy workflow**: Preserved but not used by default
- **All existing tests**: Still pass (4/4 ✅)

## Design Decisions (Karpathy Guidelines)

1. **Simple**: Each module does one thing well
2. **Surgical**: Changes are minimal, no over-engineering
3. **Assumable**: Clear names indicate responsibilities
4. **Verifiable**: All test scenarios pass without modification
5. **No Bloat**: Removed dead code, redundant imports, circular dependencies

## Future Enhancements

1. **Containerization**: `api/Dockerfile` for deployment
2. **More KB Sources**: `scripts/seed_kb.py` for web scraping + bulk loading
3. **Async Memory Save**: Non-blocking queue for user_memory_kb updates
4. **Monitoring**: Prometheus metrics in `/metrics` endpoint
5. **Persistence**: LangGraph thread + checkpoint management for conversation history

---

**Last Updated**: April 2025
**Status**: ✅ Modularization Complete - Production Ready for MVP
"""
