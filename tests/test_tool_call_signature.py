"""Gemini 3 thought-signature round-trip.

Gemini 3 returns an opaque ``extra_content.google.thought_signature`` with every
function call and rejects the *next* request if it is not echoed back verbatim:

    400 INVALID_ARGUMENT - Function call is missing a thought_signature...

The field cannot be generated or interpreted — only round-tripped — so these
tests assert it survives parsing, message history, serialization, and replay,
and that it is never fabricated for providers that don't send one.
"""

from __future__ import annotations

from types import SimpleNamespace

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from ai_employee.brains.providers.openai_compatible import (
    OpenAICompatibleBrain,
    _parse_completion,
    _to_openai_message,
)
from ai_employee.brains.schemas import ChatResult, Message, Role, ToolCall, ToolSpec

SIGNATURE = "CqQBCp8BARFNMg9vcGFxdWUtc2lnbmF0dXJlLWV4YW1wbGU="
GEMINI_EXTRA = {"google": {"thought_signature": SIGNATURE}}

TOOL = ToolSpec(
    name="list_dir",
    description="List a directory",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)


def _tool_call(**overrides) -> ChatCompletionMessageToolCall:
    payload = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "list_dir", "arguments": '{"path": "."}'},
    }
    payload.update(overrides)
    return ChatCompletionMessageToolCall.model_validate(payload)


def _completion(*tool_calls: ChatCompletionMessageToolCall) -> SimpleNamespace:
    message = SimpleNamespace(content="", tool_calls=list(tool_calls), reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=None,
        model="gemini-3.8-flash",
    )


# ── parsing ─────────────────────────────────────────────────────────────────
def test_parse_captures_thought_signature() -> None:
    result = _parse_completion(_completion(_tool_call(extra_content=GEMINI_EXTRA)))
    assert result.tool_calls[0].extra["extra_content"] == GEMINI_EXTRA


def test_parse_leaves_extra_empty_without_extra_content() -> None:
    result = _parse_completion(_completion(_tool_call()))
    assert result.tool_calls[0].extra == {}


# ── replay ──────────────────────────────────────────────────────────────────
def test_replay_emits_the_same_signature() -> None:
    result = _parse_completion(_completion(_tool_call(extra_content=GEMINI_EXTRA)))
    message = Message(role=Role.assistant, content="", tool_calls=result.tool_calls)

    payload = _to_openai_message(message)

    assert payload["tool_calls"][0]["extra_content"] == GEMINI_EXTRA
    # The signature is metadata, never part of the model-visible arguments.
    assert payload["tool_calls"][0]["function"]["arguments"] == '{"path": "."}'


def test_replay_does_not_invent_extra_content_for_other_providers() -> None:
    message = Message(
        role=Role.assistant,
        tool_calls=[ToolCall(id="call_1", name="list_dir", arguments={"path": "."})],
    )
    payload = _to_openai_message(message)
    assert "extra_content" not in payload["tool_calls"][0]


# ── persistence ─────────────────────────────────────────────────────────────
def test_signature_survives_message_serialization() -> None:
    """History may be dumped/reloaded (checkpoints, resume) between turns."""
    original = ChatResult(
        tool_calls=[
            ToolCall(
                id="call_1",
                name="list_dir",
                arguments={"path": "."},
                extra={"extra_content": GEMINI_EXTRA},
            )
        ]
    )
    message = Message(role=Role.assistant, tool_calls=original.tool_calls)

    restored = Message.model_validate(message.model_dump())
    payload = _to_openai_message(restored)

    assert payload["tool_calls"][0]["extra_content"] == GEMINI_EXTRA


# ── end to end: the second turn is what used to fail ────────────────────────
async def test_second_turn_replays_signature_to_the_api() -> None:
    brain = OpenAICompatibleBrain(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        chat_model="gemini-3.8-flash",
    )
    sent: list[dict] = []

    async def create(**kwargs):
        sent.append(kwargs)
        return _completion(_tool_call(extra_content=GEMINI_EXTRA))

    brain._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    first = await brain.chat([Message.user("list files")], tools=[TOOL])
    assert first.tool_calls[0].extra["extra_content"] == GEMINI_EXTRA

    # Turn 2: assistant tool request + the tool's observation.
    history = [
        Message.user("list files"),
        Message(role=Role.assistant, content="", tool_calls=first.tool_calls),
        Message.tool("a.txt\nb.txt", tool_call_id="call_1"),
    ]
    await brain.chat(history, tools=[TOOL])

    replayed = next(m for m in sent[1]["messages"] if m["role"] == "assistant")
    assert replayed["tool_calls"][0]["extra_content"] == GEMINI_EXTRA
