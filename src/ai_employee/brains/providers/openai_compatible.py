"""Base Brain for every OpenAI-compatible provider.

DeepSeek, OpenAI, GLM/Zhipu and Gemini (via its OpenAI-compatible endpoint) all
speak the same Chat Completions dialect, so they share one robust implementation.
Subclasses only declare base URL, key, model names and capability flags.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import (
    ChatResult,
    Message,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from ai_employee.core.exceptions import BrainError
from ai_employee.core.logging import get_logger

log = get_logger("brain.openai_compatible")


class OpenAICompatibleBrain(Brain):
    """Concrete, reusable Brain over the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        chat_model: str,
        reasoning_model: str | None = None,
        embedding_model: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        default_temperature: float = 0.2,
        default_max_tokens: int = 2048,
        name: str = "openai_compatible",
        strip_think: bool = False,
        no_think: bool = False,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise BrainError(
                "The 'openai' package is required for OpenAI-compatible providers."
            ) from exc

        if not api_key:
            log.warning("brain.no_api_key", provider=name)

        self.name = name
        self.model_name = chat_model
        self._chat_model = chat_model
        self._reasoning_model = reasoning_model
        self._embedding_model = embedding_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._max_retries = max_retries
        self._strip_think = strip_think
        self._no_think = no_think
        self.supports_reasoning = reasoning_model is not None
        self.supports_embeddings = embedding_model is not None
        self._client = AsyncOpenAI(
            api_key=api_key or "missing",
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # we handle retries with tenacity for uniform backoff
        )

    # ── Chat ─────────────────────────────────────────────────────────────────
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
        chosen_model = model or (
            self._reasoning_model if reasoning and self._reasoning_model else self._chat_model
        )
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": self._maybe_no_think([_to_openai_message(m) for m in messages], reasoning),
            "max_tokens": max_tokens or self._default_max_tokens,
        }
        # Reasoning models reject the temperature param on some providers.
        if not reasoning:
            kwargs["temperature"] = (
                temperature if temperature is not None else self._default_temperature
            )
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
            kwargs["tool_choice"] = "auto"

        resp = await self._call(self._client.chat.completions.create, **kwargs)
        result = _parse_completion(resp)
        if self._strip_think:
            result = _separate_think(result)
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model or self._chat_model,
            messages=self._maybe_no_think([_to_openai_message(m) for m in messages]),
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens or self._default_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def structured_output(self, messages, schema, *, temperature=None, model=None):
        """Use native JSON mode when available, else fall back to the base prompt."""
        import json

        json_messages = [_to_openai_message(m) for m in messages]
        json_messages.append(
            {
                "role": "system",
                "content": (
                    "Return a JSON object conforming to this schema:\n"
                    + json.dumps(schema.model_json_schema())
                ),
            }
        )
        json_messages = self._maybe_no_think(json_messages)
        try:
            resp = await self._call(
                self._client.chat.completions.create,
                model=model or self._chat_model,
                messages=json_messages,
                temperature=temperature if temperature is not None else self._default_temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            if resp.usage:
                from ai_employee.core.usage import record_usage

                record_usage(
                    Usage(
                        prompt_tokens=resp.usage.prompt_tokens or 0,
                        completion_tokens=resp.usage.completion_tokens or 0,
                        total_tokens=resp.usage.total_tokens or 0,
                    )
                )
            return schema.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001 - fall back to prompt-based parsing
            log.debug("structured_output.json_mode_failed", error=str(exc))
            return await super().structured_output(
                messages, schema, temperature=temperature, model=model
            )

    # ── Embeddings ───────────────────────────────────────────────────────────
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._embedding_model:
            return await super().embed(texts)
        resp = await self._call(
            self._client.embeddings.create,
            model=self._embedding_model,
            input=list(texts),
        )
        return [d.embedding for d in resp.data]

    def _maybe_no_think(self, openai_messages: list[dict], reasoning: bool = False) -> list[dict]:
        """Append a ``/no_think`` directive (honored by Qwen3 and similar) to keep
        local generation fast, unless reasoning was explicitly requested."""
        if self._no_think and not reasoning:
            return [*openai_messages, {"role": "system", "content": "/no_think"}]
        return openai_messages

    async def aclose(self) -> None:
        await self._client.close()

    # ── Internals ────────────────────────────────────────────────────────────
    async def _call(self, fn, **kwargs):
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )
        async def _inner():
            return await fn(**kwargs)

        try:
            return await _inner()
        except _RETRYABLE as exc:
            raise BrainError(f"{self.name} request failed after retries: {exc}") from exc


def _retryable_types() -> tuple[type[Exception], ...]:
    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        return (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
    except ImportError:  # pragma: no cover
        return (Exception,)


_RETRYABLE = _retryable_types()


def _to_openai_message(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role.value}
    if m.role == Role.tool:
        out["content"] = m.content
        out["tool_call_id"] = m.tool_call_id
        return out
    out["content"] = m.content
    if m.tool_calls:
        import json

        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    if m.name:
        out["name"] = m.name
    return out


# DeepSeek's native tool-call markup, which occasionally arrives as plain text in
# `content` rather than as structured `tool_calls`. The sentinel uses full-width
# bars (U+FF5C), sometimes rendered with ASCII pipes by intermediate proxies.
_BAR = r"[｜|]"
_INLINE_INVOKE_RE = re.compile(
    rf"<{_BAR}*\s*DSML{_BAR}*invoke\s+name=\"(?P<name>[\w.-]+)\"\s*>(?P<body>.*?)"
    rf"(?=<{_BAR}*\s*DSML{_BAR}*invoke|<\s*/{_BAR}*\s*DSML{_BAR}*tool_calls|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_INLINE_PARAM_RE = re.compile(
    rf"<{_BAR}*\s*DSML{_BAR}*parameter\s+name=\"(?P<key>[\w.-]+)\""
    rf"(?P<attrs>[^>]*)>(?P<value>.*?)<\s*/{_BAR}*\s*DSML{_BAR}*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_INLINE_ANY_RE = re.compile(rf"<\s*/?{_BAR}*\s*DSML{_BAR}*[^>]*>", re.IGNORECASE)


def _has_inline_tool_markup(text: str) -> bool:
    return bool(text) and "DSML" in text and "invoke" in text


def _coerce_param(raw: str, attrs: str) -> Any:
    """Best-effort typing for a leaked parameter value.

    The markup carries hints like ``string="true"``; otherwise fall back to JSON
    so lists/objects/numbers survive, and finally to the trimmed string.
    """
    import json

    value = raw.strip()
    if re.search(r'string\s*=\s*"true"', attrs or "", re.IGNORECASE):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _recover_inline_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Parse leaked DSML markup into tool calls; return them and the cleaned text."""
    calls: list[ToolCall] = []
    for i, m in enumerate(_INLINE_INVOKE_RE.finditer(content)):
        name = m.group("name")
        args: dict[str, Any] = {}
        for p in _INLINE_PARAM_RE.finditer(m.group("body") or ""):
            args[p.group("key")] = _coerce_param(p.group("value"), p.group("attrs"))
        calls.append(ToolCall(id=f"recovered_{i}", name=name, arguments=args))

    if not calls:
        return [], content

    # Drop everything from the first sentinel onward, then clear stray tags so the
    # model's genuine prose (if any) survives as the message content.
    start = content.find("<")
    prose = content[:start] if start > 0 else ""
    cleaned = _INLINE_ANY_RE.sub("", prose).strip()
    log.warning(
        "brain.recovered_inline_tool_calls", count=len(calls), tools=[c.name for c in calls]
    )
    return calls, cleaned


def _parse_completion(resp: Any) -> ChatResult:
    import json

    choice = resp.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

    content = msg.content or ""
    # Some DeepSeek responses emit their native tool-call markup as plain text
    # instead of structured tool_calls. Without recovery the agent loop sees no
    # call, treats the raw markup as its final answer, and the step fails
    # verification. Parse the markup back into real calls and strip it from
    # the visible content.
    if not tool_calls and _has_inline_tool_markup(content):
        recovered, cleaned = _recover_inline_tool_calls(content)
        if recovered:
            tool_calls = recovered
            content = cleaned

    usage = Usage()
    if resp.usage:
        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens or 0,
            completion_tokens=resp.usage.completion_tokens or 0,
            total_tokens=resp.usage.total_tokens or 0,
        )

    extra: dict[str, Any] = {}
    # DeepSeek reasoner (and some local servers) expose chain-of-thought separately.
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        extra["reasoning_content"] = reasoning

    return ChatResult(
        content=content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        model=resp.model,
        usage=usage,
        extra=extra,
    )


def _separate_think(result: ChatResult) -> ChatResult:
    """Move inline ``<think>…</think>`` reasoning (Qwen3 et al.) out of the visible
    answer and into ``extra['reasoning_content']`` so agents see clean output."""
    import re

    content = result.content
    if "<think>" not in content:
        return result
    thinks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Handle an unterminated <think> (truncated reasoning) by dropping everything after it.
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    if thinks:
        result.extra.setdefault("reasoning_content", "\n".join(t.strip() for t in thinks))
    result.content = cleaned
    return result
