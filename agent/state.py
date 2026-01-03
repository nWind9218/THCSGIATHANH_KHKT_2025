from typing import Optional, Literal, Dict, Any
from typing_extensions import TypedDict, List
class AIParameter(TypedDict):
    model: str
    prompt: str
    frequency_penalty: float
    max_tokens: int
    presence_penalty: float
    temperature: float
    top_p: float
class Message(TypedDict): 
    role: Literal['agent', 'staff','user']
    content: str
class Conversation(TypedDict):
    conversation_id: str
    user_id: str
    messages: List[Message]
    is_new_user: bool
class Response(TypedDict):
    output: Optional[str]
class UserEmotionMetadata(TypedDict):
    duration: str
    trigger: str 
    context: str
class UserEmotion(TypedDict):
    status: Literal["joy", "sadness", "fear", "anger", "uncertain"]
    crisis_level: Literal["low", "medium", "high", "critical"]
    problem: str
    is_new_problem: bool
    metadata: UserEmotionMetadata
class RiskAssessment(TypedDict):
    self_harm: bool
    violence: bool
    urgency: Literal["normal", "watch", "immediate"]
class BotPlan:
    solution: str
    tone: str
    must_not_do: str
    
class State(TypedDict):
    conversation: Conversation
    user_emotion: UserEmotion
    bot_plan:  BotPlan
    risk: RiskAssessment
    confidence_score: float
    response: Response # Response của các node
    next_step: Literal["listen", "clarify", "comfort","guide", "escalate"]
    