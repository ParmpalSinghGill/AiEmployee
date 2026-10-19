"""Per-agent iteration & loop budgets, configurable from .env.

Covers ``AGENT_MAX_ITERATIONS``, ``AGENT_MAX_ITERATIONS_BY_NAME`` and
``AGENT_LOOP_EXTENSIONS_BY_NAME`` — parsing, precedence, and that the agent loop
actually honors them.
"""

from __future__ import annotations

from ai_employee.config.settings import parse_agent_limits

# ── parsing ──────────────────────────────────────────────────────────────────


def test_parse_agent_limits_basic() -> None:
    assert parse_agent_limits("coding=16, research=12") == {"coding": 16, "research": 12}
    assert parse_agent_limits("coding=16;research=12") == {"coding": 16, "research": 12}
    assert parse_agent_limits("") == {}


def test_parse_agent_limits_normalizes_names() -> None:
    assert parse_agent_limits(" Codeing-Agent = 4 ") == {"codeing_agent": 4}
    assert parse_agent_limits("Coding=8") == {"coding": 8}


def test_parse_agent_limits_tolerates_junk() -> None:
    # Malformed entries are ignored; a typo in .env must never break startup.
    assert parse_agent_limits("coding=abc, =5, x, research=3") == {"research": 3}
    assert parse_agent_limits("coding=-4") == {"coding": 0}
    assert parse_agent_limits("coding=3.0") == {"coding": 3}


# ── resolver precedence ──────────────────────────────────────────────────────


def test_iteration_limit_precedence(settings) -> None:
    # No override → the agent's own built-in default.
    assert settings.iteration_limit_for("coding", 12) == 12

    # Global override applies to everyone.
    s = settings.model_copy(update={"agent_max_iterations": 20})
    assert s.iteration_limit_for("coding", 12) == 20
    assert s.iteration_limit_for("planner", 4) == 20

    # Per-agent override wins over the global and the default.
    s = settings.model_copy(
        update={"agent_max_iterations": 20, "agent_max_iterations_by_name": "coding=30"}
    )
    assert s.iteration_limit_for("coding", 12) == 30
    assert s.iteration_limit_for("planner", 4) == 20


def test_loop_limit_precedence(settings) -> None:
    s = settings.model_copy(update={"agent_loop_extensions_max": 2})
    assert s.loop_limit_for("coding") == 2

    s = settings.model_copy(
        update={
            "agent_loop_extensions_max": 2,
            "agent_loop_extensions_by_name": "coding=5,planner=0",
        }
    )
    assert s.loop_limit_for("coding") == 5
    assert s.loop_limit_for("planner") == 0  # explicit zero is honored
    assert s.loop_limit_for("research") == 2  # falls back to the global


# ── the agent loop honors the settings ────────────────────────────────────────


def _build_agent(name: str, max_iterations: int):
    from ai_employee.agents.base import ToolUsingAgent

    class _Agent(ToolUsingAgent):
        allowed_tools = ["echo"]

    _Agent.name = name
    _Agent.max_iterations = max_iterations
    return _Agent()


async def _run(agent, settings):
    from ai_employee.agents.base import AgentContext
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from tests.test_loop_check import _NeverAnswersBrain, _registry

    begin_task_tool_history()
    try:
        return await agent.run(
            "do work", AgentContext(brain=_NeverAnswersBrain(), tools=_registry(settings))
        )
    finally:
        clear_task_tool_history()


async def test_agent_uses_per_agent_iteration_override(settings, monkeypatch) -> None:
    import ai_employee.agents.base as base_mod

    s = settings.model_copy(
        update={
            "agent_loop_check_enabled": False,  # isolate the base budget
            "agent_max_iterations_by_name": "coding=2, research=9",
        }
    )
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)

    coding = await _run(_build_agent("coding", 12), s)
    assert coding.steps_taken == 2  # per-agent override won over the default (12)

    research = await _run(_build_agent("research", 8), s)
    assert research.steps_taken == 9

    other = await _run(_build_agent("devops", 5), s)
    assert other.steps_taken == 5  # no override → built-in default


async def test_agent_uses_global_iteration_override(settings, monkeypatch) -> None:
    import ai_employee.agents.base as base_mod

    s = settings.model_copy(
        update={"agent_loop_check_enabled": False, "agent_max_iterations": 3}
    )
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)

    result = await _run(_build_agent("anybody", 11), s)
    assert result.steps_taken == 3
    assert "max_iterations" in result.metadata["note"]


async def test_agent_uses_per_agent_loop_override(settings, monkeypatch) -> None:
    """Two base rounds × (1 base + 1 extension) = 4 rounds, one loop check."""
    import ai_employee.agents.base as base_mod

    s = settings.model_copy(
        update={
            "agent_loop_check_enabled": True,
            "agent_loop_extensions_max": 2,
            "agent_loop_extensions_by_name": "coding=1",
        }
    )
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)

    result = await _run(_build_agent("coding", 2), s)
    assert result.steps_taken == 2 * (1 + 1)
    assert len(result.metadata.get("loop_checks") or []) == 1

    # A different agent still gets the global 2 extensions → 3 windows.
    other = await _run(_build_agent("research", 2), s)
    assert other.steps_taken == 2 * (1 + 2)
