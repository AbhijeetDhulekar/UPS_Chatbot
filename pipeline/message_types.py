# models/message_types.py
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class MessageRole(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"

class MessageType(str, Enum):
    QUERY = "query"
    RESPONSE = "response"
    VALIDATION = "validation"
    FEEDBACK = "feedback"
    ERROR = "error"
    DEBUG = "debug"

class BaseMessage(BaseModel):
    """Base message class following LangChain message format"""
    role: MessageRole
    content: str
    type: MessageType = MessageType.QUERY
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class SystemMessage(BaseMessage):
    """System message for instructions"""
    role: MessageRole = MessageRole.SYSTEM
    
    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)

class HumanMessage(BaseMessage):
    """Human/user message"""
    role: MessageRole = MessageRole.HUMAN
    
    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)

class AIMessage(BaseMessage):
    """AI/assistant message"""
    role: MessageRole = MessageRole.AI
    sources: List[Dict] = Field(default_factory=list)
    confidence: float = 0.0
    validation_status: Optional[str] = None
    
    def dict(self) -> Dict:
        data = super().dict()
        data.update({
            "sources": self.sources,
            "confidence": self.confidence,
            "validation_status": self.validation_status
        })
        return data

class ValidationMessage(BaseMessage):
    """Validation agent message"""
    role: MessageRole = MessageRole.TOOL
    type: MessageType = MessageType.VALIDATION
    is_valid: bool = False
    feedback: Optional[str] = None
    corrected_content: Optional[str] = None
    
    def dict(self) -> Dict:
        data = super().dict()
        data.update({
            "is_valid": self.is_valid,
            "feedback": self.feedback,
            "corrected_content": self.corrected_content
        })
        return data