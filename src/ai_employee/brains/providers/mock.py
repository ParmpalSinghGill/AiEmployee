"""Deterministic, offline Brain.

Makes the entire framework runnable and testable without any API keys or network.
It produces plausible, deterministic responses, supports the structured-output and
tool-calling contracts, and generates stable hash-based embeddings.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Sequence

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import ChatResult, Message, Role, ToolCall, ToolSpec, Usage


class MockBrain(Brain):
    name = "mock"
    supports_tools = True
    supports_streaming = True
    supports_structured_output = True
    supports_embeddings = True
    supports_reasoning = True

    model_name = "mock-1 (offline)"

    def __init__(self, *, embedding_dim: int = 1536) -> None:
        self._dim = embedding_dim

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
        last_user = next(
            (m.content for m in reversed(messages) if m.role == Role.user), ""
        )
        # Concatenate ALL system messages: the schema instruction is often appended
        # as a second system message after a role-specific one.
        system = "\n".join(m.content for m in messages if m.role == Role.system)

        # If a JSON schema was requested, synthesize a conforming object so
        # structured_output works offline.
        if "JSON schema" in system or "json object" in system.lower():
            payload = _synth_from_schema(system)
            if payload is not None:
                return ChatResult(
                    content=json.dumps(payload),
                    finish_reason="stop",
                    model="mock",
                    usage=_usage(messages),
                )

        # If tools are offered and the prompt looks like it wants action, emit a call.
        if tools and _looks_actionable(last_user):
            chosen = tools[0]
            return ChatResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_mock_1",
                        name=chosen.name,
                        arguments=_mock_args(chosen),
                    )
                ],
                finish_reason="tool_calls",
                model="mock",
                usage=_usage(messages),
            )

        reply = f"[mock:{model or 'default'}] "
        if reasoning:
            reply += "(reasoning) "
        reply += f"Acknowledged: {last_user[:200]}" if last_user else "Hello from the mock brain."
        return ChatResult(
            content=reply,
            finish_reason="stop",
            model="mock",
            usage=_usage(messages),
            extra={"reasoning_content": "step-by-step mock reasoning"} if reasoning else {},
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        result = await self.chat(messages, model=model)
        for token in re.findall(r"\S+\s*", result.content):
            yield token

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding: stable across runs, unit-ish scale.
        out: list[float] = []
        counter = 0
        while len(out) < self._dim:
            h = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            for b in h:
                out.append((b / 255.0) * 2.0 - 1.0)
                if len(out) >= self._dim:
                    break
            counter += 1
        return out


def _usage(messages: Sequence[Message]) -> Usage:
    prompt = sum(len(m.content) for m in messages) // 4
    return Usage(prompt_tokens=prompt, completion_tokens=8, total_tokens=prompt + 8)


def _looks_actionable(text: str) -> bool:
    verbs = ("run", "execute", "search", "read", "write", "fetch", "list", "create", "query")
    return any(v in text.lower() for v in verbs)


def _mock_args(tool: ToolSpec) -> dict:
    props = tool.parameters.get("properties", {})
    args: dict = {}
    for key, spec in props.items():
        t = spec.get("type")
        if t == "string":
            args[key] = spec.get("default", "mock")
        elif t in ("integer", "number"):
            args[key] = spec.get("default", 1)
        elif t == "boolean":
            args[key] = spec.get("default", True)
        elif t == "array":
            args[key] = spec.get("default", [])
        else:
            args[key] = spec.get("default")
    return args


def _synth_from_schema(system_text: str) -> dict | None:
    """Extract a JSON schema embedded in the system text and synthesize a value.

    The schema is the outermost JSON object, so we match the FIRST ``{`` to its
    balanced closing brace (string-aware) rather than grabbing a nested object.
    """
    start = system_text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(system_text)):
        ch = system_text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    schema = json.loads(system_text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                if isinstance(schema, dict):
                    return _synth_value(schema, schema)
                return None
    return None


def _resolve_ref(schema: dict, root: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    # e.g. "#/$defs/PlanStep"
    parts = ref.lstrip("#/").split("/")
    node: dict = root
    for p in parts:
        node = node.get(p, {})
    return node


def _synth_value(schema: dict, root: dict):
    schema = _resolve_ref(schema, root)
    # Handle anyOf/oneOf (e.g. Optional[...] => [type, null]).
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"]
            return _synth_value(options[0], root) if options else None
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        props = schema.get("properties", {})
        return {k: _synth_value(v, root) for k, v in props.items()}
    if t == "array":
        item = schema.get("items", {"type": "string"})
        return [_synth_value(item, root)]
    if t == "integer":
        return 1
    if t == "number":
        return 0.5
    if t == "boolean":
        return True
    return "mock"
