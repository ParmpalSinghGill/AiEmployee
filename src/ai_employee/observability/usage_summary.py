"""Summarise which agents and tools a run actually used.

A run's raw evidence is scattered across ``step_results`` (one record per planned
step, tagged with its assigned agent) and ``trace`` (one entry per execute node,
carrying that step's ``tool_trace``). This module folds both into two small,
stable lists an operator (or the CLI / API / Result tab) can read at a glance:

    {"agents_used": [{"agent", "steps", "tools", "ok", "failed"}],
     "tools_used":  [{"tool", "calls", "ok", "failed", "agents"}]}

Both lists are ordered by first appearance (agents) / call count (tools) so the
output is deterministic and easy to test.
"""

from __future__ import annotations

from typing import Any


def _agent_entry(agents: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    entry = agents.get(name)
    if entry is None:
        entry = {
            "agent": name,
            "steps": 0,
            "tools": set(),
            "ok": 0,
            "failed": 0,
        }
        agents[name] = entry
    return entry


def _tool_entry(tools: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    entry = tools.get(name)
    if entry is None:
        entry = {"tool": name, "calls": 0, "ok": 0, "failed": 0, "agents": set()}
        tools[name] = entry
    return entry


def _accumulate_tool_trace(
    tool_trace: list[dict[str, Any]] | None,
    *,
    agent: str | None,
    tools: dict[str, dict[str, Any]],
    agents: dict[str, dict[str, Any]],
) -> None:
    for call in tool_trace or []:
        name = call.get("tool")
        if not name:
            continue
        tool = _tool_entry(tools, name)
        tool["calls"] += 1
        if call.get("ok"):
            tool["ok"] += 1
        else:
            tool["failed"] += 1
        if agent:
            tool["agents"].add(agent)
            a = _agent_entry(agents, agent)
            a["tools"].add(name)
            if call.get("ok"):
                a["ok"] += 1
            else:
                a["failed"] += 1


def summarize_usage(
    *,
    step_results: list[dict[str, Any]] | None = None,
    trace: list[dict[str, Any]] | None = None,
    agent: str | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build ``{"agents_used": [...], "tools_used": [...]}`` from run evidence.

    Accepts either a full workflow (``step_results`` + ``trace``) or a single
    agent result (``agent`` + ``tool_trace``). Extra evidence is simply merged.
    """
    agents: dict[str, dict[str, Any]] = {}
    tools: dict[str, dict[str, Any]] = {}

    # 1. Planned steps → one count per agent, and capture any per-step tool trace
    #    a caller attached directly to the record.
    for record in step_results or []:
        name = record.get("agent")
        if not name:
            continue
        entry = _agent_entry(agents, name)
        entry["steps"] += 1
        if record.get("skipped"):
            continue
        _accumulate_tool_trace(
            record.get("tool_trace"),
            agent=name,
            tools=tools,
            agents=agents,
        )

    # 2. Execute-node trace → tool calls attributed to the running agent. Steps
    #    that halted before producing a record (e.g. approval gate) still show up
    #    here, which is exactly when an operator wants to see what ran.
    for entry in trace or []:
        name = entry.get("agent")
        if name:
            # Register the agent even when it made no tool calls (planner, ceo).
            _agent_entry(agents, name)
        _accumulate_tool_trace(
            entry.get("tools"),
            agent=name,
            tools=tools,
            agents=agents,
        )

    # 3. A single agent result (e.g. `ask`/`download`): the agent ran even if it
    #    never appears in a plan.
    if agent:
        _agent_entry(agents, agent)
        _accumulate_tool_trace(tool_trace, agent=agent, tools=tools, agents=agents)

    # Finalise: sets → sorted lists, stable ordering.
    agents_used: list[dict[str, Any]] = []
    for name, entry in agents.items():
        agents_used.append(
            {
                "agent": name,
                "steps": entry["steps"],
                "tools": sorted(entry["tools"]),
                "ok": entry["ok"],
                "failed": entry["failed"],
            }
        )

    tools_used: list[dict[str, Any]] = []
    for name, entry in tools.items():
        tools_used.append(
            {
                "tool": name,
                "calls": entry["calls"],
                "ok": entry["ok"],
                "failed": entry["failed"],
                "agents": sorted(entry["agents"]),
            }
        )
    # Most-used first, then alphabetical for a deterministic tie-break.
    tools_used.sort(key=lambda t: (-t["calls"], t["tool"]))

    return {"agents_used": agents_used, "tools_used": tools_used}


__all__ = ["summarize_usage"]
