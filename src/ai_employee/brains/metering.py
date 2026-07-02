"""Always-on token metering wrapper around any :class:`Brain`.

Records the token usage of every ``chat`` call into the active
:class:`~ai_employee.core.usage.TokenMeter` (no-op when none is active). This is
the counterpart to :class:`~ai_employee.brains.instrument.ExplainBrain`: where
that captures *I/O* for the trace, this captures *cost*. Both are transparent —
identity and capability flags are mirrored — so callers and the golden rule
("business logic depends only on ``Brain``") are unaffected.

``structured_output`` is left to delegate to the wrapped brain: providers either
use the base prompt-based implementation (which records its own nested ``chat``)
or a native JSON mode that records its usage at the provider — either way the
tokens are counted once, without double counting here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pydantic import BaseModel

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import ChatResult, Message, ToolSpec
from ai_employee.core.usage import brain_label, record_usage

TModel = TypeVar("TModel", bound=BaseModel)


class MeteredBrain(Brain):
    """Wraps a brain, totalling token usage from every ``chat`` call."""

    def __init__(self, inner: Brain) -> None:
        self._inner = inner
        self.name = inner.name
        self.model_name = inner.model_name
        self.supports_tools = inner.supports_tools
        self.supports_streaming = inner.supports_streaming
        self.supports_structured_output = inner.supports_structured_output
        self.supports_embeddings = inner.supports_embeddings
        self.supports_reasoning = inner.supports_reasoning

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
        with brain_label(self._inner.name, model or self._inner.model_name):
            result = await self._inner.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                reasoning=reasoning,
            )
            record_usage(result.usage)
        return result

    async def structured_output(
        self,
        messages: Sequence[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> TModel:
        with brain_label(self._inner.name, model or self._inner.model_name):
            return await self._inner.structured_output(
                messages, schema, temperature=temperature, model=model
            )

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


__all__ = ["MeteredBrain"]
