"""Brain layer — the provider-independent LLM abstraction.

Business logic depends ONLY on :class:`Brain` and the schemas in this package.
Concrete providers (DeepSeek, OpenAI, Gemini, GLM, mock) live in
``ai_employee.brains.providers`` and are selected purely from configuration via
:func:`get_brain`.
"""

from ai_employee.brains.base import Brain
from ai_employee.brains.factory import get_brain, get_embedder
from ai_employee.brains.schemas import (
    ChatResult,
    Message,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "Brain",
    "get_brain",
    "get_embedder",
    "Message",
    "Role",
    "ChatResult",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
