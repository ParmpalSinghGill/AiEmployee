"""Truncation helpers for explainability traces.

The dashboard's "Explain" view shows the real input/output of every component
(LLM calls, tools, planner, verifier). To keep the event stream — which is polled
and persisted as JSON — bounded, large strings are clipped here before logging.
The clip marker keeps the original length visible so nothing looks silently lost.
"""

from __future__ import annotations

from typing import Any

#: Default per-string clip length for trace fields.
MAX_CHARS = 1500


def clip(text: Any, limit: int = MAX_CHARS) -> str:
    """Return ``text`` as a string, truncated to ``limit`` chars with a marker."""
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"… (+{len(s) - limit} more chars)"


def clip_obj(obj: Any, limit: int = MAX_CHARS) -> Any:
    """Recursively clip every string inside a JSON-able structure.

    Lists are capped at 50 items so a runaway collection can't bloat an event.
    """
    if isinstance(obj, str):
        return clip(obj, limit)
    if isinstance(obj, dict):
        return {k: clip_obj(v, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        clipped = [clip_obj(v, limit) for v in list(obj)[:50]]
        if len(obj) > 50:
            clipped.append(f"… (+{len(obj) - 50} more items)")
        return clipped
    return obj
