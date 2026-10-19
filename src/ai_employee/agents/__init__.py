"""Agent layer.

A :class:`BaseAgent` runs a tool-using (ReAct-style) loop over a :class:`Brain`.
Specialized agents (CEO, Planner, Research, Coding, …) are produced from
declarative specs plus a few custom subclasses, and assembled in an
:class:`AgentRegistry` for the LangGraph orchestrator to route between.
"""

from ai_employee.agents.base import AgentContext, AgentResult, BaseAgent, ToolUsingAgent
from ai_employee.agents.registry import AgentRegistry, build_agent_registry
from ai_employee.agents.routing import DEFAULT_WORKER_AGENT, MODE_BASE_AGENT, base_agent_for_mode

__all__ = [
    "BaseAgent",
    "ToolUsingAgent",
    "AgentContext",
    "AgentResult",
    "AgentRegistry",
    "build_agent_registry",
    "DEFAULT_WORKER_AGENT",
    "MODE_BASE_AGENT",
    "base_agent_for_mode",
]
