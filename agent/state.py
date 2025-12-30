from typing import Optional, Literal, Dict, Any
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
class UserEmotion(TypedDict):
    status: Literal["Joy", "Sadness","Fear","Anger","Disgust","Surprise"]
    crisis_level: Literal[""]
    problem: str
    need_llm: Literal["Therapeutic Help","Guidance Help","Emergency Help"]
    metadata: Dict[str, Any]
class UserEmotionMetadata(TypedDict):
    duration: str
    trigger: str 
    context: str
    notes: str