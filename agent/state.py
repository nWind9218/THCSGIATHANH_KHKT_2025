from typing import Optional
from typing_extensions import TypedDict, List
class Conversation(TypedDict):
    conversation_id: str
    messages: list
    history: Optional[list]
    bot_type: str
class AIParameter(TypedDict):
    model: str
    prompt: str
    frequency_penalty: float
    max_tokens: int
    presence_penalty: float
    temperature: float
    top_p: float
class Response(TypedDict):
    message: str
class State(TypedDict):
    conversation: Conversation
    ai_parameter: AIParameter
    response: Response
    cache_hit: bool
    status: bool