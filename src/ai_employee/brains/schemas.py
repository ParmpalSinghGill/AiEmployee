"""Provider-neutral message & result schemas used by every Brain.

These deliberately mirror the common denominator across OpenAI/DeepSeek/Gemini/GLM
so that swapping providers never changes the shapes business logic sees.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(BaseModel):
    """A single chat message."""

    role: Role
    content: str = ""
    # When role == tool, identifies which tool call this responds to.
    tool_call_id: str | None = None
    # When role == assistant and the model requested tools.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.system, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.user, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=Role.assistant, content=content)

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str | None = None) -> Message:
        return cls(role=Role.tool, content=content, tool_call_id=tool_call_id, name=name)


class ToolSpec(BaseModel):
    """JSON-schema description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResult(BaseModel):
    """The normalized result of a (non-streaming) chat call."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    # Provider-specific extras (e.g. DeepSeek reasoning_content) live here.
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


Message.model_rebuild()
