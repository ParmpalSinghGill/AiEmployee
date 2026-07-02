"""The :class:`Brain` abstraction.

Every provider implements this identical interface. Business logic, agents, and
tools depend on ``Brain`` — never on a vendor SDK — so the provider can be swapped
through configuration alone.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pydantic import BaseModel

from ai_employee.brains.schemas import ChatResult, Message, ToolSpec

TModel = TypeVar("TModel", bound=BaseModel)


class Brain(abc.ABC):
    """Provider-independent LLM interface.

    Capability flags let callers degrade gracefully (e.g. fall back to a JSON
    prompt when ``supports_structured_output`` is False, or to another embedder
    when ``supports_embeddings`` is False).
    """

    #: Stable provider identifier, e.g. "deepseek".
    name: str = "brain"
    #: Human-readable active chat model, e.g. "qwen3:8b" (for /config & the UI).
    model_name: str = ""

    # ── Capability flags (override per provider) ─────────────────────────────
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_structured_output: bool = True
    supports_embeddings: bool = False
    supports_reasoning: bool = False

    # ── Core operations ──────────────────────────────────────────────────────
    @abc.abstractmethod
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
        """Single completion. If ``reasoning`` and the provider has a reasoning
        model, that model is used."""

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield content chunks. Default falls back to a single ``chat`` call."""
        result = await self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, model=model
        )
        yield result.content

    async def structured_output(
        self,
        messages: Sequence[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> TModel:
        """Return a validated instance of ``schema``.

        Default implementation prompts for JSON and validates — providers with
        native structured output should override for reliability.
        """
        import json

        from ai_employee.brains.schemas import Message as _Msg
        from ai_employee.brains.schemas import Role

        instruction = (
            "Respond ONLY with a single JSON object that conforms to this JSON schema. "
            "No prose, no markdown fences.\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        augmented = [*messages, _Msg(role=Role.system, content=instruction)]
        result = await self.chat(augmented, temperature=temperature, model=model)
        # Count tokens for this nested call: a metering wrapper above us never sees
        # it (we called our own chat, not the wrapper's), so record it here.
        from ai_employee.core.usage import record_usage

        record_usage(result.usage)
        payload = _extract_json(result.content)
        return schema.model_validate(payload)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return an embedding vector per input text."""
        raise NotImplementedError(
            f"Provider {self.name!r} does not support embeddings; "
            "configure EMBEDDING_PROVIDER to a provider that does."
        )

    # ── Convenience ──────────────────────────────────────────────────────────
    async def complete(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        """Shorthand: prompt-in, text-out."""
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(prompt))
        result = await self.chat(messages, **kwargs)
        return result.content

    async def aclose(self) -> None:
        """Release any underlying client resources. Override if needed."""
        return None


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    import json
    import re

    text = text.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced object in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
