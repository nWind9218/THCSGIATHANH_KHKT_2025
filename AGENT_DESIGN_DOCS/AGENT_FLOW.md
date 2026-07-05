# Agent Flow

Tai lieu nay tom tat luong agent chinh trong repo, dua tren graph duoc build tai `graph/workflow.py` va luong goi graph tu WebSocket tai `api/chat_ws.py`.

## 1. Core counseling graph

```mermaid
flowchart TD
    A([START]) --> B[load_memory\n- load takeover flag\n- load history/topic\n- retrieve user memory]

    B -->|human_takeover = true| Z([END])
    B -->|human_takeover = false| C[safety_gateway\n- classify emergency risk]

    C -->|is_emergency = true| D[handoff_human\n- notify admin\n- set takeover flag\n- return comfort response]
    C -->|is_emergency = false| E[intent_routing\n- classify out_of_scope/simple/complex\n- update topic]

    D --> Z

    E -->|intent = out_of_scope| F[out_of_scope\n- fixed safe fallback]
    E -->|intent = simple| G[simple_response\n- answer from prompt + user memory]
    E -->|intent = complex| H[info_gap_assessor\n- decide missing vs sufficient context]

    H -->|info_gap_status = missing| I[clarification\n- ask one follow-up question]
    H -->|info_gap_status = sufficient| J[deep_reasoning\n- search psychology KB\n- search student knowledge KB\n- search user memory KB\n- generate guided response]

    F --> K[save_memory\n- save history/topic\n- decide long-term memory write\n- update long-term style]
    G --> K
    I --> K
    J --> K

    K --> Z
```

## 2. Runtime entry flow from WebSocket

```mermaid
flowchart TD
    A[Student WS connect\n/ws/chat/{user_id}] --> B[Validate origin]
    B --> C[Accept socket]
    C --> D[Load recent history + current topic]
    D --> E[Send connection_acknowledged]
    E --> F[Receive student message]
    F --> G[build_counseling_graph]
    G --> H[graph.ainvoke(state)]
    H --> I[Send bot_response]
    I --> J[Publish admin alert]
    J --> K{Emergency or takeover?}
    K -->|Yes| L[Publish emergency alert]
    K -->|No| M[Wait next message]
    L --> M
    M --> F
```

## 3. Routing conditions

- `load_memory -> END`: khi `human_takeover` da duoc bat truoc do.
- `safety_gateway -> handoff_human`: khi LLM tra ve emergency.
- `intent_routing -> out_of_scope | simple_response | info_gap_assessor`: dua tren `intent_category`.
- `info_gap_assessor -> clarification | deep_reasoning`: dua tren `info_gap_status`.

## 4. Luu y quan trong

- `deep_reasoning_node` co the dat `human_takeover = true` khi phat hien escalation pattern trong student knowledge, nhung graph hien tai van di tiep qua `save_memory` roi moi ket thuc.
- Thong tin `human_takeover` sau `deep_reasoning` duoc lop goi ben ngoai doc va phat alert trong WebSocket layer.
- Entry point CLI dung `workflow()` trong `main.py`, con API runtime dung `build_counseling_graph()` truc tiep.

## 5. Nguon tham chieu

- `graph/workflow.py`
- `graph/edges.py`
- `graph/nodes.py`
- `api/chat_ws.py`
- `main.py`
