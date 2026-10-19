"""Agent registry — assembles every agent the orchestrator can route to."""

from __future__ import annotations

from ai_employee.agents.base import AgentContext, AgentResult, BaseAgent, ToolUsingAgent
from ai_employee.agents.specs import AGENT_SPECS, AgentSpec
from ai_employee.core.logging import get_logger

log = get_logger("agent.registry")


class _SpecAgent(ToolUsingAgent):
    """A ToolUsingAgent configured entirely from an :class:`AgentSpec`."""

    def __init__(self, spec: AgentSpec) -> None:
        self.name = spec.name
        self.role = spec.role
        self.description = spec.description
        self.system_prompt = spec.system_prompt
        self.category = spec.category
        self.allowed_tools = list(spec.tools) if spec.tools is not None else None
        self.max_iterations = spec.max_iterations


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent, *, replace: bool = False) -> BaseAgent:
        if agent.name in self._agents and not replace:
            raise ValueError(f"agent {agent.name!r} already registered")
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"unknown agent: {name!r}")
        return self._agents[name]

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def names(self) -> list[str]:
        return sorted(self._agents)

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def catalog(self) -> list[dict[str, str]]:
        """Human/LLM-readable roster used for routing & the /agents endpoint."""
        return [
            {
                "name": a.name,
                "role": getattr(a, "role", ""),
                "description": a.description,
                "category": getattr(a, "category", ""),
            }
            for a in self._agents.values()
        ]


def build_agent_registry() -> AgentRegistry:
    """Instantiate all spec-driven agents plus the custom ones."""
    from ai_employee.agents.custom import (
        CEOAgent,
        CodingAgent,
        HumanApprovalAgent,
        PlannerAgent,
    )

    reg = AgentRegistry()
    for spec in AGENT_SPECS:
        # The coding agent has a switchable backend (brain / claude / codex / cursor).
        reg.register(CodingAgent(spec) if spec.name == "coding" else _SpecAgent(spec))
    reg.register(PlannerAgent())
    reg.register(CEOAgent())
    reg.register(HumanApprovalAgent())
    log.info("agents.ready", count=len(reg.names()))
    return reg


__all__ = ["AgentRegistry", "build_agent_registry", "AgentContext", "AgentResult"]
