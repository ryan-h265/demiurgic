"""Conversation manager for multi-turn chat with context management."""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """Single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None

    def to_dict(self) -> Dict:
        """Convert message to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tool_calls=data.get("tool_calls"),
            tool_results=data.get("tool_results"),
        )


class ConversationManager:
    """Manages conversation history with context window handling."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_context_length: int = 8192,
        max_history_turns: int = 20,
    ):
        """Initialize conversation manager.

        Args:
            system_prompt: Optional system prompt to set behavior
            max_context_length: Maximum context window in tokens (approximate)
            max_history_turns: Maximum number of turns to keep in history
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_length = max_context_length
        self.max_history_turns = max_history_turns
        self.messages: List[Message] = []
        self.total_tokens_estimate = 0

    def _default_system_prompt(self) -> str:
        """Default system prompt for coding assistant behavior."""
        # This will be enhanced with tool schemas when tools are available
        return """You are a helpful coding assistant powered by ChatGLM3. You can:

1. Write and explain code in multiple programming languages
2. Debug and fix code issues
3. Use tools to execute code, read/write files, and run commands
4. Break down complex tasks into steps
5. Provide clear explanations and best practices

When you need to use a tool, output a JSON code block like this:

```json
{
  "tool": "tool_name",
  "parameters": {
    "param": "value"
  }
}
```

Be proactive with tool usage - don't just describe what to do, actually do it!"""

    def add_message(
        self,
        role: MessageRole,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
    ) -> Message:
        """Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            tool_calls: Optional tool calls made by assistant
            tool_results: Optional results from tool execution

        Returns:
            Created Message object
        """
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        self.messages.append(message)

        # Rough token estimate (4 chars ≈ 1 token)
        self.total_tokens_estimate += len(content) // 4

        # Trim history if needed
        self._trim_history()

        return message

    def _trim_history(self) -> None:
        """Trim conversation history to stay within limits."""
        # Keep only last N turns (user + assistant pair = 1 turn)
        if len(self.messages) > self.max_history_turns * 2:
            # Always keep system messages
            system_msgs = [m for m in self.messages if m.role == "system"]
            recent_msgs = [m for m in self.messages if m.role != "system"][-self.max_history_turns * 2:]
            self.messages = system_msgs + recent_msgs

            # Recalculate token estimate
            self.total_tokens_estimate = sum(len(m.content) // 4 for m in self.messages)

    def get_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get messages formatted for model input.

        Args:
            include_system: Whether to include system prompt

        Returns:
            List of message dictionaries
        """
        messages = []

        if include_system and self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        for msg in self.messages:
            if msg.role == "system" and not include_system:
                continue
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return messages

    def get_last_user_message(self) -> Optional[Message]:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []
        self.total_tokens_estimate = 0

    def save_to_file(self, filepath: Path) -> None:
        """Save conversation to JSON file.

        Args:
            filepath: Path to save conversation
        """
        data = {
            "system_prompt": self.system_prompt,
            "max_context_length": self.max_context_length,
            "max_history_turns": self.max_history_turns,
            "messages": [m.to_dict() for m in self.messages],
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path) -> "ConversationManager":
        """Load conversation from JSON file.

        Args:
            filepath: Path to load conversation from

        Returns:
            Loaded ConversationManager instance
        """
        with open(filepath) as f:
            data = json.load(f)

        manager = cls(
            system_prompt=data.get("system_prompt"),
            max_context_length=data.get("max_context_length", 8192),
            max_history_turns=data.get("max_history_turns", 20),
        )

        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            manager.messages.append(msg)

        return manager

    def get_context_summary(self) -> Dict:
        """Get summary of current conversation context.

        Returns:
            Dictionary with context statistics
        """
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in self.messages if m.role == "assistant"),
            "tool_messages": sum(1 for m in self.messages if m.role == "tool"),
            "estimated_tokens": self.total_tokens_estimate,
            "context_usage_percent": (self.total_tokens_estimate / self.max_context_length) * 100,
        }


__all__ = ["ConversationManager", "Message", "MessageRole"]
