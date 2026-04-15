# Conversation Summary Update Prompt

**Purpose**: Update conversation memory with user intent and context  
**Version**: 1.0  
**Language**: Mixed (English structure + Vietnamese analysis)  
**Target Output**: JSON with intent, topic, intense level, and context  
**Last Updated**: 2026-04-08

---

## Core Task

Update the conversation memory to capture user's INTENT + CONTEXT in concise form, tracking progression over time.

---

## Key Concepts

### Intent vs Emotion
- **Intent**: What the user wants/needs from the conversation
- **Emotion**: How they feel (covered separately by emotion_extraction)
- **Focus**: INTENT ONLY, not emotion

### Intense (Progression Indicator)
Tracks how the situation is progressing:

- **starting**: First time this intent appears OR topic is entirely new
- **growing**: Same intent appears again with more details/examples OR follow-up questions
- **request_high_urgency**: Explicit demand for action, frustration with waiting

**Important**: Intense can ONLY stay same or increase, NEVER decrease

---

## Input Variables (to be populated at runtime)

- `{last_summary}`: Previous summary (may be null)
- `{prev_intense}`: Previous intense level
- `{recent_msgs}`: Recent messages in conversation
- `{lastest_message}`: Current user message
- `{is_returning_user}`: Boolean - is this a returning user?

---

## Update Logic

### Decision: CONTINUING vs CHANGED

**CONTINUING**:
- Same topic/intent as before
- Building on previous message
- Asking follow-up questions
- Example: Student discusses math problems → asks more about geometry

**CHANGED**:
- Completely new topic
- Shift in what they need
- Different from previous intent
- Example: "I struggled with bullying" → now asking "How do I make friends?"
- **Note**: If previous summary doesn't exist, MUST be "CHANGED"

### Intense Progression Rules

**Rule 1 - starting**:
- When: First time this intent appears
- OR: Previous summary does not exist
- OR: User speaks in general/exploratory terms
- When to keep: Still exploring, no repetition yet

**Rule 2 - growing**:
- When: Same intent appears AGAIN
- OR: User adds more details/examples
- OR: Follow-up questions on same topic
- Condition: Previous intense MUST be "starting" or already "growing"
- Special: FOR RETURNING USERS: If they come back with same problem = strong signal for "growing"

**Rule 3 - request_high_urgency**:
- When: Explicit request for immediate action/help
- OR: Language indicates "cannot wait", "right now", "urgent"
- OR: Repetition with frustration + demand for action
- Condition: Previous intense MUST be "growing"
- Special: FOR RETURNING USERS: If been in "growing" before and return, escalate to this

**Critical Rule**: 
- Intense can ONLY stay the same or increase
- NEVER decrease intense
- For returning users, be more sensitive to progression signs

---

## Output Schema

```json
{
  "intent": "What does the user want/need (concise)",
  "main_topic": "Main topic/subject area",
  "intense": "starting | growing | request_high_urgency",
  "status": "continuing | changed",
  "key_context": [
    "important detail 1",
    "important detail 2",
    "important detail 3"
  ],
  "confidence_score": 0.0 to 1.0,
  "is_returning_user": true | false
}
```

---

## Prompt Template

```
Nhiệm vụ của bạn là Cập nhật trí nhớ về Ý ĐỊNH + CONTEXT của người dùng trở nên thật ngắn gọn, súc tích, diễn đạt được context mà cuộc trò chuyện đã trải qua

USER STATUS: {"Returning user (Lần thứ 2+ chạy)" if is_returning_user else "New user (Lần đầu tiên)"}

INPUT:
- Previous summary: {json.dumps(last_summary, ensure_ascii=False) if last_summary else "None"}
- Previous Intense Level: {prev_intense}
- Recent messages: {recent_msgs}
- Current user messages: {json.dumps(lastest_message, ensure_ascii=False)}
- Is Returning User: {is_returning_user}

Task:
- Decide whether the user's intent is CONTINUING or CHANGED. 
- If continuing: enrich or slightly update the summary.
- If changed: replace the summary to reflect the new intent.
- If returning user: Consider previous context more heavily and look for progression patterns.

Rules:
- Focus on user intent and context only.
- If previous status is CONTINUING, current status is CHANGED, replace EVERYTHING
- Ignore assistant messages.
- Keep it concise and durable.
- Do NOT guess emotions if unclear.
- If previous summary IS NOT DEFINED, status MUST AUTOMATICALLY be set to CHANGED
- For returning users, pay extra attention to how their situation has evolved
- "intense" indicates the urgency or progression level of the user's intent

INTENSE DECISION RULES (STRICT):

You MUST decide intense by PROGRESSION, not emotion.

1. starting: First time intent appears OR no previous summary OR general/exploratory terms
2. growing: Same intent appears again OR more details added OR follow-up questions
3. request_high_urgency: Explicit action request OR "cannot wait" language OR frustration + demand

IMPORTANT:
- Intense can ONLY stay the same or increase.
- NEVER decrease intense.
- For returning users, be more sensitive to signs of progression.

RETURN JSON ONLY
```

---

## Examples

### Example 1: New User, Starting

**Input**:
- previous_summary: None
- prev_intense: None
- current_message: "Em cảm thấy khó khăn trong bài học toán hình"

**Output**:
```json
{
  "intent": "Cần sự hỗ trợ với toán hình",
  "main_topic": "Academic - Mathematics",
  "intense": "starting",
  "status": "changed",
  "key_context": ["Struggling with geometry", "First time mentioning"],
  "confidence_score": 0.95,
  "is_returning_user": false
}
```

### Example 2: Returning User, Progression

**Input**:
- previous_summary: {"intent": "Khó khăn với toán hình", "intense": "starting"}
- prev_intense: "starting"
- current_message: "Vẫn còn khó với hình, em đã làm bài tập nhưng không hiểu cách chứng minh tam giác. Em chiều nay có tiết dạy thêm"

**Output**:
```json
{
  "intent": "Cần lời giải thích chi tiết về cách chứng minh tam giác trong toán hình",
  "main_topic": "Academic - Geometry Proofs",
  "intense": "growing",
  "status": "continuing",
  "key_context": [
    "Already attempted homework",
    "Specific struggle: triangle proofs",
    "Has tutoring session this afternoon",
    "Looking for detailed explanation"
  ],
  "confidence_score": 0.92,
  "is_returning_user": true
}
```

### Example 3: Returning User, Topic Shift

**Input**:
- previous_summary: {"intent": "Khó khăn với toán hình", "intense": "growing"}
- prev_intense: "growing"
- current_message: "Toán hình hôm nay hiểu rồi! Nhưng em lại đang lo lắng vì theo sau em trong lớp luôn chế giễu em"

**Output**:
```json
{
  "intent": "Cần hỗ trợ xử lý tình huống bị bạn bè chế giễu",
  "main_topic": "Social - Peer Relationships",
  "intense": "starting",
  "status": "changed",
  "key_context": [
    "New topic - bullying/mockery from classmate",
    "Specific behavior: classmate behind following and making fun",
    "Previous academic issue resolved",
    "First mention of social issue"
  ],
  "confidence_score": 0.88,
  "is_returning_user": true
}
```

---

## Context Categories

### Main Topics
- Academic: Điểm số, quy trình học, khó khăn với môn học
- Social: Bạn bè, nhóm bạn, mâu thuẫn với đồng học
- Family: Mâu thuẫn với gia đình, sự hỗ trợ
- Emotional: Cảm xúc, tâm trạng, cảm thấy cô đơn
- Career/Future: Lựa chọn nghề, planc tương lai
- Health: Giấc ngủ, tập thể dục, dinh dưỡng

---

## Usage

```python
from SYSTEM_PROMPT.registry import prompt_registry

summary_prompt = prompt_registry.get_function("summary_update")
# Format with conversation data
formatted_prompt = summary_prompt.format(
    last_summary=previous_summary,
    prev_intense=prev_intense_level,
    recent_msgs=recent_conversation,
    lastest_message=current_message,
    is_returning_user=is_returning
)
# Use in LLM call to update summary
```

---

## Notes

- Always return valid JSON
- confidence_score reflects certainty of intent detection
- key_context should be concise bullet points
- For returning users, always consider previous history
- Summary should be durable enough to guide future messages without rereading full history
