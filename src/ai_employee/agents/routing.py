"""Which base agent a run uses, decided by its mode.

The general ``task`` pipeline is **not** pinned: the planner decomposes the goal
and assigns the right specialist to each step (``research`` for finding things,
``coding`` for code, ``data_analysis`` for data, ``testing`` for tests, …). That
is the whole point of the multi-agent workforce, so pinning ``task`` to a single
worker would stop it from ever doing anything but that worker's job.

Only the dedicated single-purpose modes pin to one agent, and those modes run
their own pipelines (``ResearchOrchestrator`` / ``PlanOrchestrator`` / the
downloader) rather than the general orchestrator — so their entries here just
document the agent that owns that kind of run for introspection.
"""

from __future__ import annotations

#: Mode → the single agent that owns that kind of run. ``task`` is deliberately
#: absent: the general pipeline is multi-agent and must not be pinned.
MODE_BASE_AGENT: dict[str, str] = {
    "research": "research",
    "plan": "planner",
    "download": "research",
}

#: Orchestration roles, not workers: the planner never assigns a step to them.
INTERNAL_AGENTS: frozenset[str] = frozenset({"planner", "ceo", "human_approval"})

#: Retired agent name → the agent that now owns that work.
#:
#: The roster was consolidated from 26 specialists to 8 (2026-10-22). Old names
#: still turn up in saved checkpoints and run snapshots, in per-agent limits in
#: ``.env``, in callers that pin a base agent, and in LLM output that remembers
#: the old roster. They resolve transparently instead of failing; history keeps
#: showing the name that actually ran at the time.
LEGACY_AGENT_ALIASES: dict[str, str] = {
    # External information, hubs, downloads, hardware sizing → research
    "scout": "research",
    "browser": "research",
    "resource_acquirer": "research",
    "resource_downloader": "research",
    "plan_researcher": "research",
    "hardware_analyst": "research",
    "rag": "research",
    "learning": "research",
    "memory_manager": "research",
    "task_manager": "research",
    # Application, infra, database and ML code → coding
    "codesmith": "coding",
    "devops": "coding",
    "database": "coding",
    # Requirement validation → testing
    "qa": "testing",
    # Read-only audits (security, logs/health) → reviewer
    "security": "reviewer",
    "monitoring": "reviewer",
    # Notes / knowledge curation → documentation
    "knowledge_manager": "documentation",
    # Google Workspace → comms
    "email": "comms",
    "calendar": "comms",
}

#: Agent to use when a caller *explicitly* pins a run without naming one. This is
#: not the default for the ``task`` mode — ``task`` does not pin at all.
DEFAULT_WORKER_AGENT = "research"


def resolve_agent_name(name: str | None) -> str:
    """Map a retired agent name to its replacement; other names pass through."""
    key = (name or "").strip()
    return LEGACY_AGENT_ALIASES.get(key, LEGACY_AGENT_ALIASES.get(key.lower(), key))


def base_agent_for_mode(mode: str | None) -> str | None:
    """Return the single agent that owns ``mode``, or ``None`` to not pin.

    ``None`` (the default for the ``task`` mode) means "let the planner assign
    each step to the best specialist" — the normal multi-agent behavior. Passing a
    name to ``AIEmployee.run(base_agent=...)`` still pins explicitly.
    """
    key = (mode or "task").strip().lower()
    return MODE_BASE_AGENT.get(key)


__all__ = [
    "DEFAULT_WORKER_AGENT",
    "INTERNAL_AGENTS",
    "LEGACY_AGENT_ALIASES",
    "MODE_BASE_AGENT",
    "base_agent_for_mode",
    "resolve_agent_name",
]
