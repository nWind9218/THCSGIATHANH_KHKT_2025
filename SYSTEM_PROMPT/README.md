# SYSTEM_PROMPT Module Documentation

**Version**: 1.0.0  
**Purpose**: Centralized system prompt management for Mimi chatbot  
**Language**: Mixed (Vietnamese content + English infrastructure)  
**Last Updated**: 2026-04-08

---

## Overview

The `SYSTEM_PROMPT` module provides centralized management of all system prompts used throughout the Mimi application. This ensures consistency, maintainability, and easy versioning of prompts.

### Key Benefits
✅ **Centralized**: All prompts in one place  
✅ **Versioned**: Easy to track changes  
✅ **Documented**: Each prompt includes purpose, usage, and examples  
✅ **Cached**: Efficient prompt loading with caching  
✅ **Organized**: Clear categorization (agents, functions, supporting)  
✅ **Maintainable**: Easy to update without touching code

---

## Directory Structure

```
SYSTEM_PROMPT/
├── core/
│   ├── agents/                          # Agent personalities
│   │   └── mimi_school.md              # Main school support agent
│   │
│   ├── functions/                       # Task-specific prompts
│   │   ├── emotion_extraction.md        # Extract emotion signals
│   │   ├── rag_selection.md            # Select best RAG answer
│   │   ├── response_refinement.md      # Make responses kid-friendly
│   │   ├── bot_planning.md             # Plan response strategy
│   │   └── README.md                   # Function prompts docs
│   │
│   └── supporting/                      # Guidelines & templates
│       ├── response_guidance.md         # Response structure template
│       ├── summary_update.md            # Conversation summary guidelines
│       └── README.md                    # Supporting docs
│
├── __init__.py                          # Module initialization and exports
├── loader.py                            # PromptLoader class
├── registry.py                          # PromptRegistry class (main API)
└── README.md                            # This file
```

---

## Quick Start

### 1. Get Agent Prompt

```python
from SYSTEM_PROMPT.registry import prompt_registry
from langchain_core.messages import SystemMessage, HumanMessage

# Get Mimi school guidance prompt
agent_prompt = prompt_registry.get_agent("mimi_school")

# Use in LLM call
messages = [
    SystemMessage(content=agent_prompt),
    HumanMessage(content=user_input)
]
response = llm.ainvoke(messages)
```

### 2. Get Function Prompt

```python
# Get emotion extraction prompt
emotion_prompt = prompt_registry.get_function("emotion_extraction")

# Format with your variables
formatted = emotion_prompt.format(
    messages=user_message,
    prev_intense="starting",
    conv_summary=summary
)

# Use with LLM
response = llm.ainvoke(emotion_prompt)
```

### 3. List Available Prompts

```python
# See what's available
prompts = prompt_registry.list_prompts()
print(prompts)
# Output:
# {
#   'agents': ['mimi_school'],
#   'functions': ['emotion_extraction', 'rag_selection', ...],
#   'supporting': ['response_guidance', 'summary_update']
# }
```

### 4. Validate All Prompts

```python
# Check if all prompts load correctly
is_valid = prompt_registry.validate_all()
if is_valid:
    print("✅ All prompts valid")
```

---

## Prompt Categories

### Agents (`core/agents/`)

**Purpose**: Define bot personality and role  
**When to use**: For initial system message in LLM calls  
**Count**: 1 (currently)

| Name | Purpose |
|------|---------|
| `mimi_school` | Main Mimi agent for school support |

---

### Functions (`core/functions/`)

**Purpose**: Task-specific prompts for processing  
**When to use**: In workflow nodes that need specific LLM tasks  
**Count**: 4

| Name | Purpose | Input Variables |
|------|---------|-----------------|
| `emotion_extraction` | Extract emotion from message | messages, prev_intense, conv_summary |
| `rag_selection` | Select best answer from RAG | question, formatted_results |
| `response_refinement` | Make response kid-friendly | question, answer |
| `bot_planning` | Plan response strategy | Various context variables |

---

### Supporting (`core/supporting/`)

**Purpose**: Guidelines and templates for response generation  
**When to use**: As reference for response creation  
**Count**: 2

| Name | Purpose |
|------|---------|
| `response_guidance` | Template for structured responses |
| `summary_update` | Guidelines for keeping conversation summary |

---

## Usage Patterns

### Pattern 1: Simple Agent Usage

```python
from SYSTEM_PROMPT.registry import prompt_registry

agent = prompt_registry.get_agent("mimi_school")
# Use in: SystemMessage(content=agent)
```

### Pattern 2: Function Prompt with Formatting

```python
from SYSTEM_PROMPT.registry import prompt_registry

template = prompt_registry.get_function("emotion_extraction")
formatted = template.format(
    messages=user_msg,
    prev_intense=prev_level,
    conv_summary=summary
)
# Use in LLM call
```

### Pattern 3: Batch Load All Prompts

```python
from SYSTEM_PROMPT.registry import prompt_registry

# Load everything at initialization
all_prompts = {
    "agents": prompt_registry.get_all_agents(),
    "functions": prompt_registry.get_all_functions(),
    "supporting": prompt_registry.get_all_supporting()
}
```

### Pattern 4: Custom Registry Instance

```python
from SYSTEM_PROMPT.registry import PromptRegistry
from SYSTEM_PROMPT.loader import PromptLoader

# Create custom instance with specific path
loader = PromptLoader(base_path="/custom/path")
registry = PromptRegistry(loader=loader)

prompt = registry.get_agent("mimi_school")
```

---

## Integration with Code

### Before (Hardcoded Prompts)

```python
# agent/tools.py - OLD WAY
emotion_prompt = f"""
### TASK
Analyze the emotional signal...
... (50+ lines of prompt) ...
"""
response = await llm.ainvoke(emotion_prompt)
```

### After (Using Registry)

```python
# agent/tools.py - NEW WAY
from SYSTEM_PROMPT.registry import prompt_registry

emotion_template = prompt_registry.get_function("emotion_extraction")
formatted_prompt = emotion_template.format(
    messages=messages,
    prev_intense=prev_intense,
    conv_summary=conv_summary
)
response = await llm.ainvoke(formatted_prompt)
```

---

## API Reference

### PromptRegistry

#### Methods

```python
# Get single prompts
get_agent(agent_name: str) -> str
get_function(function_name: str) -> str
get_supporting(prompt_name: str) -> str

# Get all prompts by category
get_all_agents() -> Dict[str, str]
get_all_functions() -> Dict[str, str]
get_all_supporting() -> Dict[str, str]

# Utility
list_prompts() -> Dict[str, List[str]]
clear_cache() -> None
validate_all() -> bool
get_info() -> Dict
```

#### Global Instance

```python
from SYSTEM_PROMPT.registry import prompt_registry

# Already initialized, ready to use
prompt = prompt_registry.get_agent("mimi_school")
```

---

## Adding New Prompts

### Step 1: Create Markdown File

Create file in appropriate directory:
- `SYSTEM_PROMPT/core/agents/{name}.md`
- `SYSTEM_PROMPT/core/functions/{name}.md`
- `SYSTEM_PROMPT/core/supporting/{name}.md`

### Step 2: Add Metadata Header

```markdown
# Prompt Name

**Purpose**: What this prompt does
**Version**: 1.0
**Language**: en/vi/mixed
**Last Updated**: 2026-04-08

---

## Content starts here...
```

### Step 3: Use in Code

```python
from SYSTEM_PROMPT.registry import prompt_registry

# Automatically available!
new_prompt = prompt_registry.get_function("new_function_name")
```

---

## Maintenance

### Updating Prompts

1. Edit the `.md` file in `core/` directory
2. Clear cache (optional): `prompt_registry.clear_cache()`
3. Changes take effect immediately on next load

### Versioning

- Update version in prompt file header
- Update SYSTEM_PROMPT version in `__init__.py`
- Document changes in Git commit

### Testing

```python
# Validate after changes
is_valid = prompt_registry.validate_all()
assert is_valid, "Prompt validation failed"

# Get info for debugging
info = prompt_registry.get_info()
print(info)
```

---

## Troubleshooting

### Prompt Not Found

```
FileNotFoundError: Prompt file not found: ...
```

**Solution**: Check that the markdown file exists in the correct subdirectory

### Format String Error

```
KeyError: 'variable_name'
```

**Solution**: Ensure all required variables are provided in `.format()` call

### Slow Loading

**Solution**: Use `get_all_*()` methods at startup to pre-cache

---

## Future Enhancements

- [ ] Multi-language support (Vietnamese/English/Chinese)
- [ ] Version history tracking
- [ ] A/B testing framework
- [ ] Prompt performance metrics
- [ ] Collaborative editing interface
- [ ] Auto-translation support

---

## Contributing

To add or modify prompts:

1. Create/edit `.md` file in appropriate `core/` directory
2. Follow format and metadata conventions
3. Test with `prompt_registry.validate_all()`
4. Document changes in commit message

---

## License & Attribution

All prompts are specific to Mimi chatbot for THCS Gia Thanh.

**Contact**: [Project maintainers]
