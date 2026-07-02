"""Brain abstraction & provider tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ai_employee.brains.schemas import Message, ToolSpec


class _Plan(BaseModel):
    title: str
    steps: list[str]


async def test_mock_chat(brain):
    res = await brain.chat([Message.user("hello")])
    assert "hello" in res.content.lower()
    assert res.usage.total_tokens >= 0


async def test_mock_structured_output(brain):
    plan = await brain.structured_output([Message.user("plan a trip")], _Plan)
    assert isinstance(plan, _Plan)
    assert isinstance(plan.steps, list)


async def test_mock_tool_calling(brain):
    tool = ToolSpec(
        name="search",
        description="search",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    res = await brain.chat([Message.user("search for cats")], tools=[tool])
    assert res.has_tool_calls
    assert res.tool_calls[0].name == "search"


async def test_mock_embeddings_dim(brain):
    vecs = await brain.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 64 for v in vecs)


async def test_mock_streaming(brain):
    chunks = [c async for c in brain.stream([Message.user("count to three")])]
    assert "".join(chunks)


def test_factory_selects_mock(settings):
    from ai_employee.brains.factory import get_brain, reset_brain_cache

    reset_brain_cache()
    assert get_brain(settings).name == "mock"


def test_local_provider_constructs(monkeypatch):
    """The local (Ollama/OpenAI-compatible) brain builds from config offline."""
    from ai_employee.brains.providers.concrete import LocalBrain
    from ai_employee.config import get_settings

    s = get_settings()
    brain = LocalBrain(s)
    assert brain.name == "local"
    assert brain._chat_model == s.local_chat_model
    assert brain.supports_reasoning  # qwen3 reasoning model configured
    assert brain.supports_embeddings


def test_separate_think_strips_reasoning():
    from ai_employee.brains.providers.openai_compatible import _separate_think
    from ai_employee.brains.schemas import ChatResult

    r = ChatResult(content="<think>let me reason</think>The answer is 42.")
    out = _separate_think(r)
    assert out.content == "The answer is 42."
    assert "reason" in out.extra["reasoning_content"]


def test_factory_unknown_provider_raises():
    from ai_employee.brains.factory import _build
    from ai_employee.config import get_settings
    from ai_employee.core.exceptions import ConfigurationError

    class Fake:
        value = "does-not-exist"

    with pytest.raises(ConfigurationError):
        _build(Fake(), get_settings())  # type: ignore[arg-type]
