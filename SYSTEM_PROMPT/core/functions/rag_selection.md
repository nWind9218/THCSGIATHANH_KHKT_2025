# RAG Selection Prompt

**Purpose**: LLM selects best answer from RAG knowledge base results  
**Version**: 1.0  
**Language**: Vietnamese  
**Target Output**: JSON with selected answer  
**Last Updated**: 2026-04-08

---

## Core Task

Select the most appropriate answer from knowledge base results based on user question.

---

## Input Variables (to be populated at runtime)

- `{question}`: User's original question
- `{formatted_results}`: JSON array of candidates from RAG
  ```json
  [
    {
      "question": "Candidate question from KB",
      "answer": "Corresponding answer",
      "similarity": 0.95
    }
  ]
  ```

---

## Selection Criteria

1. **Relevance**: How well does the answer address the user's question?
2. **Completeness**: Does the answer fully address the problem?
3. **Appropriateness**: Is it suitable for a 10-16 year old student?
4. **Safety**: Does it align with school support guidelines?

---

## Prompt Template

```
Dựa vào câu hỏi của người dùng, hãy chọn câu trả lời PHÙ HỢP NHẤT từ các đáp án sau:

Câu hỏi: {question}

Các đáp án:
{formatted_results}

Chỉ trả về JSON với format:
{
  "selected_answer": "câu trả lời được chọn",
  "reasoning": "lý do lựa chọn"
}
```

---

## System Message

**Role**: You are an expert information analyst and selector.  
**Responsibility**: Select the most accurate and relevant information.

---

## Output Schema

```json
{
  "selected_answer": "the chosen answer text",
  "reasoning": "brief explanation of why this was selected (optional)"
}
```

---

## Usage

```python
from SYSTEM_PROMPT.registry import prompt_registry

rag_prompt = prompt_registry.get_function("rag_selection")
# Format with RAG results
formatted_prompt = rag_prompt.format(
    question=user_question,
    formatted_results=json.dumps(rag_results, ensure_ascii=False)
)
# Use in conversation with LLM
```

---

## Important Notes

- Always return valid JSON
- Only select from provided candidates (don't generate new answers)
- Consider student safety in all selections
- Prioritize answers that are explicitly relevant to the question
