# memory/conversation_memory.py
import redis
import json
from typing import List, Dict, Optional
from datetime import datetime
from config import Config
from debug.debugger import debugger

class ConversationMemory:
    """
    Short-term memory for conversation history using Redis.
    Stores recent messages per session with a TTL.
    """

    def __init__(self):
        """Initialize Redis connection for conversation memory."""
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True
        )
        self.ttl = Config.CONVERSATION_TTL

    def add_message(self, session_id: str, message: dict):
        """
        Add a message to the conversation history.

        Args:
            session_id: Unique session identifier.
            message: Dictionary with at least 'role' and 'content'.
        """
        key = f"session:{session_id}:history"

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Store as JSON
        self.redis_client.rpush(key, json.dumps(message))
        self.redis_client.expire(key, self.ttl)

        debugger.log("MEMORY", {"action": "add_message", "session": session_id})

    def get_history(self, session_id: str, limit: int = 5) -> List[dict]:
        """
        Retrieve the most recent messages from conversation history.

        Args:
            session_id: Unique session identifier.
            limit: Maximum number of messages to return.

        Returns:
            List of message dictionaries in chronological order.
        """
        key = f"session:{session_id}:history"

        # Get last N messages
        messages = self.redis_client.lrange(key, -limit, -1)

        history = []
        for msg_json in messages:
            try:
                history.append(json.loads(msg_json))
            except json.JSONDecodeError:
                continue

        debugger.log("MEMORY", {"action": "get_history", "session": session_id, "count": len(history)})
        return history

    def format_for_context(self, session_id: str, limit: int = 3) -> str:
        """
        Format conversation history as a string suitable for LLM context.

        Args:
            session_id: Unique session identifier.
            limit: Number of recent messages to include.

        Returns:
            Formatted conversation string, or empty string if no history.
        """
        history = self.get_history(session_id, limit)

        if not history:
            return ""

        formatted = "Previous conversation:\n"
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted += f"{role.upper()}: {content}\n"

        return formatted

    def clear_history(self, session_id: str):
        """Delete all conversation history for a session."""
        key = f"session:{session_id}:history"
        self.redis_client.delete(key)
        debugger.log("MEMORY", {"action": "clear_history", "session": session_id})