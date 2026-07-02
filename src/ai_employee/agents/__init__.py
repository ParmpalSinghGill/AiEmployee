"""Agent layer.

A :class:`BaseAgent` runs a tool-using (ReAct-style) loop over a :class:`Brain`.
Specialized agents (CEO, Planner, Research, Coding, …) are produced from
declarative specs plus a few custom subclasses, and assembled in an
:class:`AgentRegistry` for the LangGraph orchestrator to route between.
"""

from ai_employee.agents.base import AgentContext, AgentResult, BaseAgent, ToolUsingAgent
from ai_employee.agents.registry import AgentRegistry, build_agent_registry

__all__ = [
    "BaseAgent",
    "ToolUsingAgent",
    "AgentContext",
    "AgentResult",
    "AgentRegistry",
    "build_agent_registry",
]
