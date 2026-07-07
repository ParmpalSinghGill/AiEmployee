"""Transparent instrumentation wrapper around any :class:`Brain`.

Emits structured ``brain.request`` / ``brain.response`` events so the dashboard
can show the *exact* input and output of every LLM call — the planner's
decomposition, each agent's reasoning, the verifier's review, the final
synthesis — without any call site importing a provider or even knowing it is
being traced.

This honours the project's golden rule: business logic still depends only on the
``Brain`` interface. The wrapper delegates everything to the wrapped brain and
mirrors its identity/capability flags, so it is indistinguishable to callers.
The single choke point (all reasoning flows through ``Brain``) is what makes
whole-pipeline explainability possible from one place.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

import structlog
from pydantic import BaseModel

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import ChatResult, Message, ToolSpec
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip, clip_obj

TModel = TypeVar("TModel", bound=BaseModel)

log = get_logger("brain")

#: Only the last N messages of a (growing) conversation are dumped per call.
_MAX_MESSAGES = 16


def _purpose() -> str:
    """Best-effort label of *what* this call is for, from bound context vars.

    The orchestrator binds ``node`` / ``step`` / ``agent`` (and the verifier binds
    ``phase=verify``) so each LLM call can be attributed to the right component in
    the trace without threading an argument through every call site.
    """
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


def _dump_messages(messages: Sequence[Message]) -> list[dict]:
    out: list[dict] = []
    for m in list(messages)[-_MAX_MESSAGES:]:
        item: dict = {"role": m.role.value, "content": clip(m.content)}
        if m.tool_calls:
            item["tool_calls"] = [
                {"name": c.name, "args": clip_obj(c.arguments)} for c in m.tool_calls
            ]
        if m.name:
            item["name"] = m.name
        out.append(item)
    return out


class ExplainBrain(Brain):
    """Wraps a brain, logging the I/O of every ``chat`` / ``structured_output`` call."""

    def __init__(self, inner: Brain) -> None:
        self._inner = inner
        # Mirror identity & capabilities so callers can't tell the difference.
        self.name = inner.name
        self.model_name = inner.model_name
        self.supports_tools = inner.supports_tools
        self.supports_streaming = inner.supports_streaming
        self.supports_structured_output = inner.supports_structured_output
        self.supports_embeddings = inner.supports_embeddings
        self.supports_reasoning = inner.supports_reasoning

    # Delegate any attribute we didn't explicitly mirror to the wrapped brain.
    def __getattr__(self, item: str):  # pragma: no cover - passthrough
        return getattr(self._inner, item)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        reasoning: bool = False,
    ) -> ChatResult:
        purpose = _purpose()
        log.info(
            "brain.request",
            purpose=purpose,
            kind="chat",
            model=model or self.model_name,
            tools=[t.name for t in (tools or [])],
            messages=_dump_messages(messages),
        )
        result = await self._inner.chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            reasoning=reasoning,
        )
        log.info(
            "brain.response",
            purpose=purpose,
            kind="chat",
            content=clip(result.content),
            reasoning=clip((result.extra or {}).get("reasoning_content", "")),
            tool_calls=[{"name": c.name, "args": clip_obj(c.arguments)} for c in result.tool_calls],
            finish_reason=result.finish_reason,
            usage=result.usage.model_dump(),
        )
        return result

    async def structured_output(
        self,
        messages: Sequence[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> TModel:
        purpose = _purpose()
        log.info(
            "brain.request",
            purpose=purpose,
            kind="structured",
            schema=schema.__name__,
            model=model or self.model_name,
            messages=_dump_messages(messages),
        )
        result = await self._inner.structured_output(
            messages, schema, temperature=temperature, model=model
        )
        try:
            payload = clip_obj(result.model_dump())
        except Exception:  # noqa: BLE001 - tracing must never break the call
            payload = {}
        log.info(
            "brain.response",
            purpose=purpose,
            kind="structured",
            schema=schema.__name__,
            output=payload,
        )
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._inner.stream(
            messages, temperature=temperature, max_tokens=max_tokens, model=model
        ):
            yield chunk

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._inner.embed(texts)

    async def aclose(self) -> None:
        await self._inner.aclose()
