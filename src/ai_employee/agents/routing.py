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
    "download": "resource_downloader",
}

#: Agent to use when a caller *explicitly* pins a run without naming one. This is
#: not the default for the ``task`` mode — ``task`` does not pin at all.
DEFAULT_WORKER_AGENT = "research"


def base_agent_for_mode(mode: str | None) -> str | None:
    """Return the single agent that owns ``mode``, or ``None`` to not pin.

    ``None`` (the default for the ``task`` mode) means "let the planner assign
    each step to the best specialist" — the normal multi-agent behavior. Passing a
    name to ``AIEmployee.run(base_agent=...)`` still pins explicitly.
    """
    key = (mode or "task").strip().lower()
    return MODE_BASE_AGENT.get(key)


__all__ = ["MODE_BASE_AGENT", "DEFAULT_WORKER_AGENT", "base_agent_for_mode"]
