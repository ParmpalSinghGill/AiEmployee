"""Agent registry — assembles every agent the orchestrator can route to."""

from __future__ import annotations

from collections.abc import Callable

from ai_employee.agents.base import AgentContext, AgentResult, BaseAgent, ToolUsingAgent
from ai_employee.agents.routing import LEGACY_AGENT_ALIASES, resolve_agent_name
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
        self.requires = spec.requires


def _google_connected() -> bool:
    from ai_employee.config import get_settings
    from ai_employee.integrations.google_oauth import load_accounts

    try:
        return bool(load_accounts(get_settings()))
    except Exception:  # noqa: BLE001 - an unreadable store means "not connected"
        return False


#: Integration name → "is it connected right now?". Checked on every catalog
#: build, so connecting Google mid-session makes comms plannable immediately.
INTEGRATION_CHECKS: dict[str, Callable[[], bool]] = {"google": _google_connected}


def integration_available(name: str | None) -> bool:
    if not name:
        return True
    check = INTEGRATION_CHECKS.get(name)
    return bool(check and check())


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent, *, replace: bool = False) -> BaseAgent:
        if agent.name in self._agents and not replace:
            raise ValueError(f"agent {agent.name!r} already registered")
        self._agents[agent.name] = agent
        return agent

    def resolve(self, name: str) -> str:
        """The registered name for ``name``, following legacy aliases."""
        if name in self._agents:
            return name
        target = resolve_agent_name(name)
        if target != name and target in self._agents:
            log.info("agent.legacy_alias", requested=name, resolved=target)
            return target
        return name

    def get(self, name: str) -> BaseAgent:
        resolved = self.resolve(name)
        if resolved not in self._agents:
            raise KeyError(f"unknown agent: {name!r}")
        return self._agents[resolved]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.resolve(name) in self._agents

    def names(self) -> list[str]:
        return sorted(self._agents)

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def catalog(self, *, include_unavailable: bool = False) -> list[dict[str, str]]:
        """Human/LLM-readable roster used for routing & the /agents endpoint.

        Agents whose required integration is not connected (``comms`` without a
        Google account) are left out by default — this is the roster the planner
        and teammates see, and they must not plan work that can only fail. Pass
        ``include_unavailable=True`` for a full listing (e.g. an admin view).
        """
        out: list[dict[str, str]] = []
        for a in self._agents.values():
            requires = getattr(a, "requires", None)
            available = integration_available(requires)
            if not available and not include_unavailable:
                continue
            entry = {
                "name": a.name,
                "role": getattr(a, "role", ""),
                "description": a.description,
                "category": getattr(a, "category", ""),
            }
            if include_unavailable and requires:
                entry["requires"] = requires
                entry["available"] = "yes" if available else "no"
            out.append(entry)
        return out


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


__all__ = [
    "LEGACY_AGENT_ALIASES",
    "AgentContext",
    "AgentRegistry",
    "AgentResult",
    "build_agent_registry",
    "integration_available",
]
