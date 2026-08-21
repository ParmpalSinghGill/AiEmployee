"""Dependency-aware scheduling for parallel step execution.

The sequential executor walks the plan with a cursor, so a step waits for the one
before it even when they share no data. This module answers a different question:
given what has finished, which steps can start *right now*?

Steps declare ``depends_on`` (step ids). A step is READY when every dependency has
completed; it stays WAITING otherwise, and starts the moment its last dependency
lands. Steps with no dependencies are ready immediately, so independent branches
of a plan run at the same time.

Pure functions over plain dicts — no graph state, no I/O — so the scheduling rules
can be tested directly.
"""

from __future__ import annotations

from typing import Any

# A plan whose dependencies form a cycle would deadlock the scheduler. Rather than
# hang, we break the cycle by releasing the earliest-declared step in it.
_MAX_SCHEDULE_ROUNDS = 10_000


def step_dependencies(step: dict[str, Any]) -> list[int]:
    """Declared dependency ids for ``step``, minus self-references.

    Planners occasionally emit a step that depends on itself; treating that
    literally would make the step permanently unrunnable.
    """
    own = step.get("id")
    deps: list[int] = []
    for dep in step.get("depends_on") or []:
        if dep == own:
            continue
        try:
            deps.append(int(dep))
        except (TypeError, ValueError):
            continue
    return deps


def ready_steps(
    steps: list[dict[str, Any]],
    completed_ids: set[int],
    *,
    running_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Steps whose dependencies are all satisfied and that aren't done or in flight.

    Dependencies pointing at ids absent from the plan are ignored — a planner
    referencing a step it never emitted must not strand the rest of the run.
    """
    running = running_ids or set()
    known = {s.get("id") for s in steps}
    out: list[dict[str, Any]] = []
    for step in steps:
        sid = step.get("id")
        if sid in completed_ids or sid in running:
            continue
        deps = [d for d in step_dependencies(step) if d in known]
        if all(d in completed_ids for d in deps):
            out.append(step)
    return out


def has_runnable_work(
    steps: list[dict[str, Any]],
    completed_ids: set[int],
    running_ids: set[int] | None = None,
) -> bool:
    """True while any step is still pending, regardless of readiness."""
    running = running_ids or set()
    return any(s.get("id") not in completed_ids and s.get("id") not in running for s in steps)


def schedule_rounds(steps: list[dict[str, Any]]) -> list[list[int]]:
    """Group step ids into waves that could run concurrently.

    Wave N contains every step whose dependencies all land in waves < N. Used for
    plan preview and for tests; the live executor schedules greedily instead, so a
    step starts the instant its own dependencies finish rather than waiting for its
    whole wave.
    """
    completed: set[int] = set()
    waves: list[list[int]] = []
    remaining = list(steps)
    rounds = 0
    while remaining and rounds < _MAX_SCHEDULE_ROUNDS:
        rounds += 1
        wave = ready_steps(remaining, completed)
        if not wave:
            # Dependency cycle (or all-unmet): release the first remaining step so
            # the plan makes progress instead of deadlocking.
            wave = [remaining[0]]
        ids = [s["id"] for s in wave]
        waves.append(ids)
        completed.update(ids)
        picked = set(ids)
        remaining = [s for s in remaining if s.get("id") not in picked]
    return waves


def max_parallelism(steps: list[dict[str, Any]]) -> int:
    """Widest wave in the plan — how many steps could run at once at peak."""
    waves = schedule_rounds(steps)
    return max((len(w) for w in waves), default=0)


def describe_schedule(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of how a plan would be parallelized, for logs and the UI."""
    waves = schedule_rounds(steps)
    return {
        "waves": waves,
        "wave_count": len(waves),
        "step_count": len(steps),
        "max_parallelism": max((len(w) for w in waves), default=0),
        # A plan is only worth running in parallel if some wave has >1 step.
        "parallelizable": any(len(w) > 1 for w in waves),
    }
