"""Truncation helpers for explainability traces.

The dashboard's "Explain" view shows the real input/output of every component
(LLM calls, tools, planner, verifier). To keep the event stream — which is polled
and persisted as JSON — bounded, large strings are clipped here before logging.
The clip marker keeps the original length visible so nothing looks silently lost.

How much is kept is configurable in ``.env``:

* ``EXPLAIN_MAX_CHARS``      — LLM messages, the step task/system prompt, answers
* ``EXPLAIN_TOOL_MAX_CHARS`` — tool arguments and tool output

Set either to ``0`` for **no truncation** (show everything). The defaults are
generous so the Explain view is complete out of the box, while still capping a
runaway tool that prints megabytes.
"""

from __future__ import annotations

from typing import Any

#: Fallback per-string clip length if settings can't be read.
MAX_CHARS = 1500


def _limit(name: str, fallback: int) -> int:
    """Read a clip limit from settings; 0 means "no truncation", None → fallback."""
    try:
        from ai_employee.config.settings import get_settings

        value = int(getattr(get_settings(), name))
    except Exception:  # noqa: BLE001 - tracing must never break on config issues
        return fallback
    return value


def clip(text: Any, limit: int | None = None) -> str:
    """Return ``text`` as a string, truncated to ``limit`` chars with a marker.

    ``limit`` defaults to ``EXPLAIN_MAX_CHARS``; ``0`` disables truncation.
    """
    s = "" if text is None else str(text)
    if limit is None:
        limit = _limit("explain_max_chars", MAX_CHARS)
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + f"… (+{len(s) - limit} more chars — raise EXPLAIN_MAX_CHARS to see more)"


def clip_obj(obj: Any, limit: int | None = None) -> Any:
    """Recursively clip every string inside a JSON-able structure.

    Lists are capped at 200 items so a runaway collection can't bloat an event.
    """
    if isinstance(obj, str):
        return clip(obj, limit)
    if isinstance(obj, dict):
        return {k: clip_obj(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        clipped = [clip_obj(v, limit) for v in list(obj)[:200]]
        if len(obj) > 200:
            clipped.append(f"… (+{len(obj) - 200} more items)")
        return clipped
    return obj


def tool_clip(text: Any, limit: int | None = None) -> str:
    """Clip a tool argument/output string to ``EXPLAIN_TOOL_MAX_CHARS`` (0 = all)."""
    if limit is None:
        limit = _limit("explain_tool_max_chars", MAX_CHARS)
    return clip(text, limit)


__all__ = ["MAX_CHARS", "clip", "clip_obj", "tool_clip"]
