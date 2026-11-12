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

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import ChatResult, Message, ToolSpec
from ai_employee.core import telemetry
from ai_employee.core.exceptions import BudgetExceeded
from ai_employee.core.usage import active_meters, brain_label, record_usage

TModel = TypeVar("TModel", bound=BaseModel)


def _enforce_budget() -> None:
    """Refuse the call if any active meter has spent its token budget.

    Enforcement lives here — outside the model — so no prompt can talk the
    system past its cap: once exhausted, every further LLM call raises.
    """
    for meter in active_meters():
        if meter.budget and meter.total_tokens >= meter.budget:
            raise BudgetExceeded(meter.total_tokens, meter.budget)


def _record_llm(sp: Any, labels: dict[str, str], started: float, result: ChatResult) -> None:
    """Export one finished LLM call: span attributes + latency/token metrics."""
    if not telemetry.is_enabled():
        return
    usage = result.usage
    telemetry.set_attrs(
        sp,
        **{
            "gen_ai.response.model": result.model,
            "gen_ai.response.finish_reason": result.finish_reason,
            "gen_ai.usage.input_tokens": usage.prompt_tokens if usage else None,
            "gen_ai.usage.output_tokens": usage.completion_tokens if usage else None,
            "llm.tool_calls": len(result.tool_calls),
        },
    )
    telemetry.count("llm_calls", status="ok", **labels)
    telemetry.observe("llm_duration", time.perf_counter() - started, **labels)
    if usage:
        telemetry.count("llm_tokens", usage.prompt_tokens, type="input", **labels)
        telemetry.count("llm_tokens", usage.completion_tokens, type="output", **labels)


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
        _enforce_budget()
        model_name = model or self._inner.model_name
        labels = {"provider": self._inner.name, "model": model_name, "kind": "chat"}
        started = time.perf_counter()
        with (
            brain_label(self._inner.name, model_name),
            telemetry.span(
                f"llm.chat {model_name}",
                **{
                    "gen_ai.system": self._inner.name,
                    "gen_ai.request.model": model_name,
                    "gen_ai.request.max_tokens": max_tokens,
                    "llm.messages": len(messages),
                    "llm.tools": len(tools or ()),
                    "llm.reasoning": reasoning,
                },
            ) as sp,
        ):
            try:
                result = await self._inner.chat(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    reasoning=reasoning,
                )
            except Exception:
                telemetry.count("llm_calls", status="error", **labels)
                telemetry.observe("llm_duration", time.perf_counter() - started, **labels)
                raise
            record_usage(result.usage)
            _record_llm(sp, labels, started, result)
        return result

    async def structured_output(
        self,
        messages: Sequence[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> TModel:
        _enforce_budget()
        model_name = model or self._inner.model_name
        with (
            brain_label(self._inner.name, model_name),
            telemetry.span(
                f"llm.structured_output {schema.__name__}",
                **{
                    "gen_ai.system": self._inner.name,
                    "gen_ai.request.model": model_name,
                    "llm.schema": schema.__name__,
                },
            ),
        ):
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
