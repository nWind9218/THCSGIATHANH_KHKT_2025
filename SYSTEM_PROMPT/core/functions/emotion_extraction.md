# Emotion Signal Extraction Prompt

**Purpose**: Extract emotional signals from current user message  
**Version**: 1.0  
**Language**: Mixed (English rules + Vietnamese analysis)  
**Target Output**: JSON with emotion status, problem, urgency, safety flags  
**Last Updated**: 2026-04-08

---

## Core Task

Analyze the emotional signal of the LATEST message provided.

### Key Principle
- Extract emotional signals ONLY from the CURRENT user message
- Urgency is NOT emotional — it reflects PROGRESSION over time
- Urgency MUST be derived from INTENSE in conversation memory

---

## Input Variables (to be populated at runtime)

- `{messages}`: Current user message
- `{prev_intense}`: Previous intensity level (starting, growing, request_high_urgency)
- `{conv_summary}`: Conversation summary (may be null)

---

## Classification Guidelines

### 1. Status (Emotional State)
- **joy**: Positive, happy, content
- **sadness**: Sad, melancholic, down
- **fear**: Anxious, scared, worried
- **disgust**: Disgusted, repulsed
- **anger**: Angry, frustrated, irritated
- **surprise**: Shocked, surprised
- **uncertain**: Unclear, ambiguous, neutral

**Rules:**
- If message is vague (e.g., "hmm", "ok") → status = "uncertain"
- If greeting → status = "joy"

### 2. Urgency (Progression-Based)
- **normal**: Casual conversation, venting without danger
- **watch**: Deep sadness, anger, mentions of hopelessness
- **immediate**: Explicit self-harm (cutting, suicide), violence, or clear crisis

**Minimum Urgency Rule based on INTENSE:**
- starting → urgency = "normal"
- growing → urgency = "watch"
- request_high_urgency → urgency = "immediate"

**Important:**
- Urgency can ONLY stay the same or increase
- Urgency MUST NEVER be lower than INTENSE-based minimum
- The current message may ONLY increase urgency if it explicitly indicates immediacy
- If INTENSE is missing or undefined → assume "starting"

### 3. Problem (CRITICAL - MUST BE IN VIETNAMESE)
Extract the specific subject, struggle, or topic the user is talking about.

**Valid Problem Categories:**

**Academic:**
- "khó khăn trong vấn đề toán học"
- "mất gốc toán hình"
- "cảm thấy áp lực trong học tập"
- "áp lực thi cử"

**Social:**
- "cảm thấy bị bạn bè tẩy chay"
- "gặp khó khăn trong kết nối với bạn bè"
- "khó khăn để hòa đồng"
- "khó khăn trong giao tiếp với phụ huynh"

**Emotional:**
- "cảm thấy cô đơn"
- "crush không tích"
- "cãi nhau với bố mẹ"

**Request:**
- "cần lời khuyên trong học tập"
- "cần sự hướng dẫn để trở nên thành công"

**ONLY use "MUST CLARIFY MORE INFORMATION."** if:
- Message is purely a greeting (e.g. "Hello")
- Message is completely vague (e.g. "I'm sad" without reason)

---

## Safety Rules (CRITICAL)

1. Do NOT diagnose mental disorders
2. Do NOT determine crisis level (that comes later)
3. If user mentions suicide-related keywords ("die", "kill", "suicide", "hurt myself") → **Urgency MUST be "immediate"**
4. If emotional signals are weak, factual, or neutral → status = "uncertain"
5. Prefer "uncertain" over guessing

---

## Output Schema (STRICT JSON)

```json
{
  "status": "joy | sadness | fear | disgust | anger | surprise | uncertain",
  "problem": "short phrase or empty string",
  "metadata": {
    "trigger": "what triggered the emotion",
    "duration": "how long they've felt this",
    "context": "additional context"
  },
  "self_harm": true | false,
  "violence": true | false,
  "urgency": "normal | watch | immediate",
  "confidence_score": 0.0 to 1.0
}
```

---

## Usage

```python
from SYSTEM_PROMPT.registry import prompt_registry

emotion_prompt = prompt_registry.get_function("emotion_extraction")
# Format with variables:
formatted_prompt = emotion_prompt.format(
    messages=user_message,
    prev_intense=previous_intensity,
    conv_summary=conversation_summary
)
# Use in: llm.ainvoke(emotion_prompt)
```

---

## Notes

- Always return valid JSON
- confidence_score should reflect certainty of classification
- metadata captures important context for downstream nodes
- self_harm and violence flags trigger immediate escalation
