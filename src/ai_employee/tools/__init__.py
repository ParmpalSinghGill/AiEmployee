"""Tool layer — a registry of capabilities agents can invoke.

Tools are provider-agnostic: each declares a Pydantic args schema and an async
``run``. The registry exposes them to any Brain as OpenAI-style tool specs and
enforces the human-approval policy from settings.
"""

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.invocation_history import (
    begin_task_tool_history,
    clear_task_tool_history,
)
from ai_employee.tools.registry import ToolRegistry, default_registry

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "default_registry",
    "begin_task_tool_history",
    "clear_task_tool_history",
]
