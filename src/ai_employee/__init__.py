"""AI Employee Framework.

A production-grade, provider-independent multi-agent framework that behaves like a
highly skilled employee: it accepts high-level goals, plans the work, delegates to
specialized agents, uses tools autonomously, learns reusable skills, and improves
over time.

The public surface is intentionally small — most usage goes through:

    from ai_employee import AIEmployee, get_settings

    employee = AIEmployee()
    result = await employee.run("Summarize the latest sales report and email it to the team")
"""

from __future__ import annotations

from ai_employee.config import Settings, get_settings

__all__ = ["AIEmployee", "Settings", "get_settings", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    # Lazy import so `import ai_employee` stays cheap and does not pull the heavy
    # agent/LangGraph stack unless the orchestrator is actually used.
    if name == "AIEmployee":
        from ai_employee.core.employee import AIEmployee

        return AIEmployee
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
