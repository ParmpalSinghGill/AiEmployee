"""Data types for the scenario suite: scenarios, evidence, and check results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Check:
    """One graded assertion about a run."""

    name: str
    ok: bool
    detail: str = ""
    weight: float = 1.0


@dataclass
class Evidence:
    """Everything an evaluator needs to judge one run.

    ``files_before`` / ``files_after`` are ``{relative_path: sha256}`` snapshots of
    the scenario workspace taken either side of the run. ``created`` are files that
    did not exist before; ``changed`` are files whose hash moved.
    """

    scenario_id: str
    goal: str
    mode: str
    result: dict[str, Any] = field(default_factory=dict)
    workspace: str = ""
    files_before: dict[str, str] = field(default_factory=dict)
    files_after: dict[str, str] = field(default_factory=dict)
    created: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0

    # ── Convenience accessors ────────────────────────────────────────────────
    @property
    def answer(self) -> str:
        return str(self.result.get("answer") or "").strip()

    @property
    def plan_steps(self) -> list[dict[str, Any]]:
        plan = self.result.get("plan") or {}
        return list(plan.get("steps") or [])

    def planned_agents(self) -> list[str]:
        return [s.get("agent") for s in self.plan_steps if s.get("agent")]

    def executed_agents(self) -> set[str]:
        return {
            a.get("agent")
            for a in (self.result.get("agents_used") or [])
            if a.get("agent")
        }

    def all_agents(self) -> set[str]:
        names = set(self.planned_agents()) | self.executed_agents()
        names |= {s.get("agent") for s in (self.result.get("steps") or []) if s.get("agent")}
        return {n for n in names if n}

    def tools(self) -> dict[str, dict[str, Any]]:
        return {t.get("tool"): t for t in (self.result.get("tools_used") or [])}

    def tool_ok(self, name: str) -> int:
        return int((self.tools().get(name) or {}).get("ok") or 0)

    def tool_failed(self, name: str) -> int:
        return int((self.tools().get(name) or {}).get("failed") or 0)

    @property
    def needs_approval(self) -> dict[str, Any] | None:
        return self.result.get("needs_approval")

    # ── Workspace helpers ────────────────────────────────────────────────────
    def workspace_path(self) -> Path:
        return Path(self.workspace)

    def read(self, rel: str) -> str:
        p = self.workspace_path() / rel
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def exists(self, rel: str) -> bool:
        return (self.workspace_path() / rel).exists()

    def created_with_suffix(self, suffix: str) -> list[str]:
        return [f for f in self.created if f.lower().endswith(suffix)]


#: A scenario seeds a workspace, runs a goal, and is graded by named checks.
Seed = Callable[[Path], None]
@dataclass
class Scenario:
    id: str
    title: str
    goal: str
    table: str = ""  # the "expected agents / good result" line from the spec
    mode: str = "task"  # task | research | download
    #: step agents that MUST appear in the planner's assignment
    require_agents: tuple[str, ...] = ()
    #: groups where at least one member must appear in the planner's assignment
    require_any: tuple[tuple[str, ...], ...] = ()
    #: named content checks (functions in checks.py)
    checks: tuple[str, ...] = ()
    #: files the run must NOT modify (relative to the workspace)
    protect: tuple[str, ...] = ()
    #: approval keys the harness pre-grants so the run can complete unattended
    grant: frozenset[str] = frozenset()
    seed: Seed | None = None
    notes: str = ""
