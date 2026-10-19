"""Base Brain for every OpenAI-compatible provider.

DeepSeek, OpenAI, GLM/Zhipu and Gemini (via its OpenAI-compatible endpoint) all
speak the same Chat Completions dialect, so they share one robust implementation.
Subclasses only declare base URL, key, model names and capability flags.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

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
        rate_limit_retries: int = 6,
        rate_limit_base_wait: float = 5.0,
        rate_limit_max_wait: float = 60.0,
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
        self._max_retries = max(1, max_retries)
        self._rate_limit_retries = max(self._max_retries, rate_limit_retries)
        self._rate_limit_base_wait = max(1.0, rate_limit_base_wait)
        self._rate_limit_max_wait = max(self._rate_limit_base_wait, rate_limit_max_wait)
        self._strip_think = strip_think
        self._no_think = no_think
        self.supports_reasoning = reasoning_model is not None
        self.supports_embeddings = embedding_model is not None
        self._client = AsyncOpenAI(
            api_key=api_key or "missing",
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # retries/backoff are handled in _call for uniform logs
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
        """Call the provider, with visible cooldowns for rate limits.

        A 429/quota error is treated differently from a transient connection
        error: it gets a longer, capped exponential backoff (and honors the
        provider's ``Retry-After`` when present) so a per-minute quota can
        recover instead of failing the step after a few seconds. Each sleep is
        logged as ``brain.rate_limited`` with a ``cooling_id`` and ``retry_at``;
        the task manager turns that into a visible "cooling down" status. The
        matching ``brain.rate_limit_cleared`` lets the dashboard resume.
        """
        cooling_id = uuid.uuid4().hex[:8]
        model = kwargs.get("model") or self._chat_model
        last_exc: Exception | None = None
        cooling = False
        # Independent budgets: a burst of 429s must not spend the connection-error
        # retries (and vice versa). Each backoff sequence also counts only its own
        # attempts, so a connection error after several cooldowns still waits 1s,
        # not 2^N seconds.
        general_left = self._max_retries
        rate_left = self._rate_limit_retries
        general_used = 0
        rate_used = 0
        total_attempts = 0

        while general_left > 0 or rate_left > 0:
            total_attempts += 1
            try:
                result = await fn(**kwargs)
            except _RETRYABLE as exc:
                last_exc = exc
                rate_limited = _is_rate_limit(exc)
                if rate_limited:
                    rate_used += 1
                    rate_left -= 1
                    if rate_left <= 0:
                        log.error(
                            "brain.rate_limit_exhausted",
                            provider=self.name,
                            model=model,
                            attempts=rate_used,
                            cooling_id=cooling_id,
                            error=str(exc)[:500],
                        )
                        break
                    delay = _retry_wait(
                        exc,
                        rate_used,
                        rate_limited=True,
                        base=self._rate_limit_base_wait,
                        cap=self._rate_limit_max_wait,
                    )
                    cooling = True
                    retry_at = (
                        datetime.now(UTC) + timedelta(seconds=delay)
                    ).isoformat()
                    log.warning(
                        "brain.rate_limited",
                        provider=self.name,
                        model=model,
                        attempt=rate_used,
                        max_attempts=self._rate_limit_retries,
                        wait_s=round(delay, 1),
                        retry_at=retry_at,
                        cooling_id=cooling_id,
                        error=str(exc)[:500],
                    )
                else:
                    general_used += 1
                    general_left -= 1
                    if general_left <= 0:
                        if cooling:
                            # An earlier attempt was rate-limited but the call
                            # finally died on a different error. End the cooldown
                            # marker so the UI does not stay on "cooling" while
                            # the step fails.
                            log.info(
                                "brain.rate_limit_cleared",
                                provider=self.name,
                                model=model,
                                attempts=total_attempts,
                                cooling_id=cooling_id,
                                outcome="failed",
                            )
                        break
                    delay = _retry_wait(
                        exc,
                        general_used,
                        rate_limited=False,
                        base=1.0,
                        cap=20.0,
                    )
                    log.warning(
                        "brain.retry",
                        provider=self.name,
                        model=model,
                        attempt=general_used,
                        max_attempts=self._max_retries,
                        wait_s=round(delay, 1),
                        error=str(exc)[:500],
                    )
                await asyncio.sleep(delay)
            else:
                if cooling:
                    log.info(
                        "brain.rate_limit_cleared",
                        provider=self.name,
                        model=model,
                        attempts=total_attempts,
                        cooling_id=cooling_id,
                    )
                return result

        raise BrainError(
            f"{self.name} request failed after retries: {last_exc}"
        ) from last_exc


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status from an SDK exception (or its response)."""
    for obj in (exc, getattr(exc, "response", None)):
        if obj is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return value
    return None


def _is_rate_limit(exc: BaseException) -> bool:
    """True for HTTP 429 / quota-style errors (OpenAI, Gemini, GLM, …)."""
    if _status_code(exc) == 429:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    if re.search(r"\b429\b", text):
        return True
    return any(
        hint in text
        for hint in (
            "rate limit",
            "rate_limit",
            "quota",
            "resource_exhausted",
            "too many requests",
        )
    )


def _parse_retry_after(value: Any) -> float | None:
    """Parse ``Retry-After`` / ``retryDelay`` into seconds.

    Handles plain seconds (``30``), durations (``1m30s``, ``250ms``), and HTTP
    dates. Returns ``None`` when the value is not parseable.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    match = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", raw)
    if match and (match.group(1) or match.group(2)):
        return int(match.group(1) or 0) * 60 + float(match.group(2) or 0)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)ms", raw)
    if match:
        return float(match.group(1)) / 1000.0

    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0.0, (when - datetime.now(UTC)).total_seconds())
    except Exception:  # noqa: BLE001 - an unparseable header is not fatal
        return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Read a provider hint for how long to wait before retrying."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers:
        for name in (
            "retry-after",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ):
            try:
                parsed = _parse_retry_after(headers.get(name))
            except Exception:  # noqa: BLE001 - some header maps are odd
                parsed = None
            if parsed is not None:
                return parsed
    # Gemini's OpenAI-compatible endpoint returns google.rpc.RetryInfo in the
    # error body: {"error": {"details": [{"retryDelay": "31s"}]}}.
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = None
    if isinstance(body, dict):
        details = (body.get("error") or {}).get("details") or []
        for detail in details:
            if isinstance(detail, dict) and detail.get("retryDelay"):
                parsed = _parse_retry_after(detail["retryDelay"])
                if parsed is not None:
                    return parsed
    return None


def _retry_wait(
    exc: BaseException,
    attempt: int,
    *,
    rate_limited: bool,
    base: float,
    cap: float,
) -> float:
    """Wait before the next attempt: the provider hint, else exponential."""
    hinted = _retry_after_seconds(exc) if rate_limited else None
    if hinted is not None and hinted > 0:
        return min(cap, max(1.0, hinted))
    delay = base * (2 ** max(0, attempt - 1))
    return min(cap, max(0.5, delay))


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
        out["tool_calls"] = [_to_openai_tool_call(tc) for tc in m.tool_calls]
    if m.name:
        out["name"] = m.name
    return out


def _to_openai_tool_call(tc: ToolCall) -> dict[str, Any]:
    """Rebuild one tool call, replaying provider extras verbatim.

    Gemini 3 attaches ``extra_content.google.thought_signature`` to every
    function call and rejects the *next* request if it is not echoed back. The
    field is opaque and encrypted — it can only be round-tripped, never built —
    so anything captured in ``ToolCall.extra`` is re-emitted as-is.
    """
    import json

    out: dict[str, Any] = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }
    extra_content = (tc.extra or {}).get("extra_content")
    if extra_content:
        out["extra_content"] = extra_content
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
        # Keep provider extras (Gemini 3 thought signatures, etc.) so they can
        # be replayed with the call on the next turn. The OpenAI SDK surfaces
        # unknown response fields via ``model_extra``; ``getattr`` covers
        # versions that expose them as real attributes.
        extra_content = getattr(tc, "extra_content", None)
        if extra_content is None:
            extra_content = (getattr(tc, "model_extra", None) or {}).get("extra_content")
        extra = {"extra_content": extra_content} if extra_content else {}
        tool_calls.append(
            ToolCall(id=tc.id, name=tc.function.name, arguments=args, extra=extra)
        )

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
