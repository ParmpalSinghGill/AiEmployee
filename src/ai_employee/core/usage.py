"""Token accounting.

A single, provider-agnostic place to total the tokens every LLM call consumes.
Recording happens at the ``Brain`` choke point (see :mod:`ai_employee.brains.metering`),
so business logic never touches it and every agent is counted uniformly.

Usage is accumulated into a :class:`TokenMeter` bound to the current async context
via a :class:`~contextvars.ContextVar`. Wrap a unit of work in :func:`track_usage`
to get a fresh meter whose total covers every brain call made inside it::

    with track_usage() as meter:
        await orchestrator.run(goal)
    print(meter.total_tokens, meter.by_purpose)

Outside a ``track_usage`` block, :func:`record_usage` is a cheap no-op.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

from ai_employee.brains.schemas import Usage

_current_meter: contextvars.ContextVar[TokenMeter | None] = contextvars.ContextVar(
    "token_meter", default=None
)
#: (provider, model) of the brain making the current call — set by MeteredBrain so
#: ``record_usage`` can attribute tokens to a specific brain in the persistent ledger.
_brain_label: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "brain_label", default=("unknown", "unknown")
)

#: Optional persistent sink: ``fn(provider, model, usage, purpose)``. Registered at
#: startup (see the usage ledger). Kept as a callback so ``core`` doesn't depend on
#: ``observability``.
_sink: Any = None


def _purpose() -> str:
    """Attribute a call to a component from bound context vars (same scheme the
    explainability tracer uses): planner / verifier / agent:<name> / node."""
    ctx = structlog.contextvars.get_contextvars()
    if ctx.get("phase") == "verify":
        return "verifier"
    node = ctx.get("node")
    if node == "plan":
        return "planner"
    agent = ctx.get("agent")
    if agent:
        return f"agent:{agent}"
    if node:
        return str(node)
    return "llm"


class TokenMeter:
    """Accumulates token usage, with a per-purpose (per-agent) breakdown."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0
        self.by_purpose: dict[str, dict[str, int]] = {}

    def record(self, usage: Usage, purpose: str = "llm") -> None:
        total = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += total
        self.calls += 1
        bucket = self.by_purpose.setdefault(
            purpose,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        )
        bucket["prompt_tokens"] += usage.prompt_tokens
        bucket["completion_tokens"] += usage.completion_tokens
        bucket["total_tokens"] += total
        bucket["calls"] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "by_purpose": {k: dict(v) for k, v in self.by_purpose.items()},
        }


def register_usage_sink(fn: Any) -> None:
    """Install the persistent ledger sink ``fn(provider, model, usage, purpose)``."""
    global _sink
    _sink = fn


@contextmanager
def brain_label(provider: str, model: str) -> Iterator[None]:
    """Mark which brain is making calls inside this block (for ledger attribution)."""
    token = _brain_label.set((provider, model))
    try:
        yield
    finally:
        _brain_label.reset(token)


def record_usage(usage: Usage, purpose: str | None = None) -> None:
    """Add ``usage`` to the active meter and the persistent ledger.

    The in-context meter is updated only inside a :func:`track_usage` block; the
    ledger sink (if registered) fires for *every* call so cumulative per-brain
    usage is captured regardless of any active meter.
    """
    p = purpose or _purpose()
    meter = _current_meter.get()
    if meter is not None:
        meter.record(usage, p)
    if _sink is not None:
        import contextlib

        provider, model = _brain_label.get()
        with contextlib.suppress(Exception):  # accounting must never break a call
            _sink(provider, model, usage, p)


@contextmanager
def track_usage() -> Iterator[TokenMeter]:
    """Bind a fresh :class:`TokenMeter` for the duration of the block."""
    meter = TokenMeter()
    token = _current_meter.set(meter)
    try:
        yield meter
    finally:
        _current_meter.reset(token)


__all__ = [
    "TokenMeter",
    "track_usage",
    "record_usage",
    "register_usage_sink",
    "brain_label",
]
