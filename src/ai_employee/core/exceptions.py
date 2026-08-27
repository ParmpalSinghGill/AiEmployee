"""Framework exception hierarchy.

A single root (:class:`AIEmployeeError`) lets callers catch everything the
framework raises, while subclasses allow precise handling per layer.
"""

from __future__ import annotations

from typing import Any


class AIEmployeeError(Exception):
    """Base class for all framework errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"{self.message} | {self.details}"
        return self.message


class ConfigurationError(AIEmployeeError):
    """Invalid or missing configuration (e.g. provider selected but no API key)."""


class BrainError(AIEmployeeError):
    """An LLM provider call failed or returned something unusable."""


class AgentError(AIEmployeeError):
    """An agent failed to complete its step."""


class ToolError(AIEmployeeError):
    """A tool raised during execution. Carries the tool name for diagnostics."""

    def __init__(self, message: str, *, tool: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.tool = tool


class MemoryError(AIEmployeeError):  # noqa: A001 - intentional domain name
    """A memory store operation failed."""


class VerificationError(AIEmployeeError):
    """Output failed verification and could not be repaired within the retry budget."""


class BudgetExceeded(AIEmployeeError):
    """The run hit its token budget — no further LLM calls are allowed."""

    def __init__(self, spent: int, budget: int) -> None:
        super().__init__(
            f"Token budget exhausted: {spent} of {budget} tokens used. "
            "Raise the budget or re-run without one."
        )
        self.spent = spent
        self.budget = budget


class ApprovalRequired(AIEmployeeError):
    """Raised when an operation needs human approval before it can proceed.

    The orchestrator catches this to surface an approval request rather than
    silently performing a sensitive action.
    """

    def __init__(self, operation: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(f"Human approval required for operation: {operation}")
        self.operation = operation
        self.payload = payload or {}
