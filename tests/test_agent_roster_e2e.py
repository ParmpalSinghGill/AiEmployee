"""End-to-end smoke coverage for every registered agent."""

from __future__ import annotations

from ai_employee.agents import build_agent_registry
from ai_employee.brains.providers.mock import MockBrain


class NoToolMockBrain(MockBrain):
    """Keep this roster test offline and focused on agent dispatch."""

    async def chat(self, messages, **kwargs):
        # The ordinary MockBrain intentionally calls the first advertised tool
        # for imperative prompts. This test is about invoking every role through
        # the facade, not repeating every tool integration test 29 times.
        return await super().chat(messages, tools=None)


async def test_facade_can_call_every_registered_agent(employee, monkeypatch):
    """Exercise all agents through AIEmployee.ask_agent using the offline brain."""
    expected = set(build_agent_registry().names())
    monkeypatch.setattr(employee, "brain", NoToolMockBrain(embedding_dim=64))
    monkeypatch.setattr(employee.settings, "coding_backend", "brain")

    results = {}
    for agent_name in sorted(expected):
        results[agent_name] = await employee.ask_agent(
            agent_name,
            f"Give a brief role summary for {agent_name}.",
            session_id=f"agent-roster-{agent_name}",
        )

    assert set(results) == expected
    for agent_name, result in results.items():
        assert result["agent"] == agent_name
        # The approval specialist is expected to pause unless explicitly
        # approved; every other agent should return a smoke-test response.
        if agent_name == "human_approval":
            assert result["needs_approval"]
        else:
            assert result["success"]
            assert result["output"].strip()

