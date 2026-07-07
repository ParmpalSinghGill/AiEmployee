"""Tool base classes.

A :class:`Tool` pairs a Pydantic args model with an async ``run``. The args model
*is* the JSON schema we advertise to the LLM, so tool calling stays type-safe end
to end.
"""

from __future__ import annotations

import abc
import time
from typing import Any, ClassVar, get_type_hints

from pydantic import BaseModel, ValidationError

from ai_employee.brains.schemas import ToolSpec
from ai_employee.core.exceptions import ApprovalRequired, ToolError
from ai_employee.core.logging import get_logger

log = get_logger("tool")


class ToolResult(BaseModel):
    """Uniform result envelope returned by every tool."""

    ok: bool = True
    output: Any = None
    error: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = {}

    def as_text(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if isinstance(self.output, dict) and self.output.get("status") == "duplicate":
            import json

            return json.dumps(self.output, ensure_ascii=False, default=str)
        if isinstance(self.output, str):
            return self.output
        import json

        try:
            return json.dumps(self.output, default=str, ensure_ascii=False)
        except TypeError:
            return str(self.output)


class _EmptyArgs(BaseModel):
    pass


class Tool(abc.ABC):
    """Base class for all tools.

    Subclasses set :pyattr:`name`, :pyattr:`description`, an ``Args`` Pydantic
    model, and implement :meth:`_run`.
    """

    name: ClassVar[str] = "tool"
    description: ClassVar[str] = ""
    #: Category for discovery (filesystem, code, web, data, comms, devops…).
    category: ClassVar[str] = "general"
    #: Approval key checked against ``settings.approval_required``.
    approval_key: ClassVar[str | None] = None
    #: The Pydantic model describing this tool's arguments.
    Args: ClassVar[type[BaseModel]] = _EmptyArgs

    # ── Public API ───────────────────────────────────────────────────────────
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Validate args, run, and wrap the result. Never raises for tool-level
        failures — those come back as ``ToolResult(ok=False)`` so the agent loop
        can decide how to react."""
        started = time.perf_counter()
        try:
            args = self.Args(**kwargs)
        except ValidationError as exc:
            return ToolResult(ok=False, error=f"invalid arguments: {exc}")
        try:
            result = await self._run(args)
            if not isinstance(result, ToolResult):
                result = ToolResult(output=result)
        except ApprovalRequired:
            raise
        except ToolError as exc:
            result = ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - tools must not crash the loop
            log.warning("tool.exception", tool=self.name, error=str(exc))
            result = ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    @abc.abstractmethod
    async def _run(self, args: BaseModel) -> ToolResult | Any:
        """Do the work. Receives a validated instance of ``Args``."""

    def requires_approval(self, arguments: dict[str, Any]) -> str | None:
        """Return the approval key required for this specific call, or None.

        Default: the class-level :pyattr:`approval_key`. Override to make approval
        depend on the arguments (e.g. a shell tool that exempts read-only commands).
        """
        return self.approval_key

    # ── LLM interop ──────────────────────────────────────────────────────────
    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.Args.model_json_schema(),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.name}>"


def _check_signature(tool: type[Tool]) -> None:  # pragma: no cover - dev aid
    get_type_hints(tool._run)
