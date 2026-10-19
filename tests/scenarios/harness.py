"""Execute scenarios through the real pipeline and grade the results.

The runner is provider-agnostic: point it at the offline ``mock`` brain for a
deterministic smoke run, or at a configured real provider (DeepSeek/OpenAI/local…)
for a true realtime routing test. Each scenario runs in its own isolated
workspace so seeded files and written artefacts never bleed between cases.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .checks import CHECKS
from .models import Check, Evidence, Scenario

_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".hf_home", "node_modules"}
_SKIP_SUFFIXES = (".pyc", ".pyo")

#: Extra approval keys the framework recognises beyond settings.approval_required.
_EXTRA_APPROVAL_KEYS = {
    "waiting_for_user",
    "run_research_code",
    "resource_acquisition",
    "resource_download",
    "shell_write",
    "email_send",
    "git_push",
    "db_write",
    "deploy",
    "mcp_tool",
    "calendar_write",
}


# ── Workspace snapshotting ───────────────────────────────────────────────────
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def snapshot(root: Path) -> dict[str, str]:
    """``{relative_path: sha256}`` for every file under ``root``."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        out[str(path.relative_to(root))] = digest
    return out


def _diff(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str]]:
    created = sorted(set(after) - set(before))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    return created, changed


# ── Environment + employee construction ──────────────────────────────────────
def _env_updates(scenario: Scenario, root: Path, brain: str) -> dict[str, str]:
    """The environment overrides that point the process at this scenario."""
    run_dir = root / scenario.id
    (run_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)

    updates: dict[str, str] = {
        "WORKSPACE_DIR": str(run_dir / "workspace"),
        "SESSIONS_DIR": str(run_dir / "sessions"),
        "MEMORY_BACKEND": "memory",
        "LOG_LEVEL": "ERROR",  # keep scenario output readable
        "USAGE_LEDGER_FILE": str(run_dir / "usage.jsonl"),
    }
    if brain == "mock":
        # Isolate from any real UI-managed defaults/keys so we truly stay offline.
        updates.update(
            {
                "LLM_PROVIDER": "mock",
                "EMBEDDING_PROVIDER": "mock",
                "UI_PREFERENCES_FILE": str(run_dir / "ui_preferences.json"),
                "SECRETS_FILE": str(run_dir / "secrets.json"),
            }
        )
    return updates


def _apply_env(updates: dict[str, str]) -> dict[str, str | None]:
    """Set env vars, returning their previous values (for exact restore)."""
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _refresh_settings() -> None:
    from ai_employee.brains.factory import reset_brain_cache
    from ai_employee.config import reload_settings

    reload_settings()
    reset_brain_cache()


def _all_approval_keys() -> set[str]:
    try:
        from ai_employee.config import get_settings

        keys = set(get_settings().approval_required)
    except Exception:  # noqa: BLE001
        keys = set()
    return keys | _EXTRA_APPROVAL_KEYS


def _default_factory() -> Any:
    from ai_employee.core.employee import AIEmployee

    return AIEmployee()


# ── Run one scenario ─────────────────────────────────────────────────────────
async def run_scenario(
    scenario: Scenario,
    *,
    root: Path,
    brain: str = "mock",
    approve_all: bool = False,
    employee_factory: Callable[[], Any] | None = None,
    timeout: float = 900.0,
) -> Evidence:
    """Seed the workspace, run the goal, and return the raw evidence."""
    previous_env = _apply_env(_env_updates(scenario, root, brain))
    _refresh_settings()
    workspace = root / scenario.id / "workspace"

    factory = employee_factory or _default_factory
    emp = factory()

    # Seed fixtures AFTER the employee is built so tools are already pointed at ws.
    if scenario.seed is not None:
        scenario.seed(workspace)

    before = snapshot(workspace)
    granted = set(scenario.grant)
    if approve_all:
        granted |= _all_approval_keys()

    session_id = scenario.id
    started = time.time()
    error: str | None = None
    result: dict[str, Any] = {}
    try:
        coro = _dispatch(emp, scenario, session_id, granted)
        result = await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        error = f"timed out after {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        aclose = getattr(emp, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()
        _restore_env(previous_env)
        _refresh_settings()

    after = snapshot(workspace)
    created, changed = _diff(before, after)

    return Evidence(
        scenario_id=scenario.id,
        goal=scenario.goal,
        mode=scenario.mode,
        result=result if isinstance(result, dict) else {},
        workspace=str(workspace),
        files_before=before,
        files_after=after,
        created=created,
        changed=changed,
        error=error,
        elapsed_s=round(time.time() - started, 1),
    )


async def _dispatch(emp: Any, scenario: Scenario, session_id: str, granted: set[str]) -> dict:
    if scenario.mode == "research":
        return await emp.research(scenario.goal, session_id=session_id, approved_ops=granted, auto_run=False)
    if scenario.mode == "download":
        return await emp.download(scenario.goal, session_id=session_id, approved_ops=granted)
    # default: full pipeline
    kwargs: dict[str, Any] = {"session_id": session_id, "approved_ops": granted}
    if _accepts_kwarg(emp.run, "on_progress"):
        kwargs["on_progress"] = None
    return await emp.run(scenario.goal, **kwargs)


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


# ── Grading ──────────────────────────────────────────────────────────────────
def evaluate_scenario(scenario: Scenario, ev: Evidence) -> dict[str, Any]:
    """Apply the routing + content checks and summarise pass/fail."""
    checks: list[Check] = []

    if ev.error:
        checks.append(Check("run_completed", False, ev.error))

    if scenario.mode == "task":
        checks.append(
            Check(
                "planner_decomposed",
                bool(ev.plan_steps),
                f"{len(ev.plan_steps)} planned step(s)" if ev.plan_steps else "the planner produced no steps",
            )
        )
        planned = ev.planned_agents()
        missing = [a for a in scenario.require_agents if a not in planned]
        checks.append(
            Check(
                "routes_to_expected_agents",
                not missing,
                f"planned: {planned}" if not missing else f"missing from plan: {missing} (planned: {planned})",
            )
        )
        for group in scenario.require_any:
            hit = [a for a in group if a in planned]
            checks.append(
                Check(
                    f"uses_one_of[{','.join(group)}]",
                    bool(hit),
                    f"matched {hit}" if hit else f"none of {list(group)} planned",
                )
            )

    answered = bool(ev.answer)
    needs = ev.needs_approval
    checks.append(
        Check(
            "produced_answer",
            answered or bool(needs),
            "answer present" if answered else ("paused for approval" if needs else "no answer produced"),
        )
    )

    for name in scenario.checks:
        fn = CHECKS.get(name)
        if fn is None:  # pragma: no cover - guarded by test_scenarios
            checks.append(Check(name, False, "unknown check"))
        else:
            checks.append(fn(scenario, ev))

    total = sum(c.weight for c in checks if c.weight > 0)
    passed = sum(c.weight for c in checks if c.ok and c.weight > 0)
    score = round(passed / total, 3) if total else 0.0
    if ev.error:
        status = "error"
    elif score >= 1.0:
        status = "pass"
    elif score >= 0.5:
        status = "partial"
    else:
        status = "fail"

    return {
        "scenario_id": scenario.id,
        "title": scenario.title,
        "status": status,
        "score": score,
        "passed": int(round(passed)),
        "total": int(round(total)),
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail, "weight": c.weight} for c in checks
        ],
        "planned_agents": ev.planned_agents(),
        "executed_agents": sorted(ev.executed_agents()),
        "tools": sorted(ev.tools()),
        "needs_approval": (needs or {}).get("operation") if needs else None,
        "created": ev.created,
        "changed": ev.changed,
        "elapsed_s": ev.elapsed_s,
        "answer_chars": len(ev.answer),
        "error": ev.error,
    }


async def run_all(
    *,
    root: Path,
    scenarios: list[Scenario],
    brain: str = "mock",
    approve_all: bool = False,
    employee_factory: Callable[[], Any] | None = None,
    timeout: float = 900.0,
) -> list[dict[str, Any]]:
    """Run and grade every scenario, returning one graded report per scenario."""
    _ensure_dir(root)
    reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        ev = await run_scenario(
            scenario,
            root=root,
            brain=brain,
            approve_all=approve_all,
            employee_factory=employee_factory,
            timeout=timeout,
        )
        reports.append(evaluate_scenario(scenario, ev))
    return reports


# ── Reporting ────────────────────────────────────────────────────────────────
def _icon(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render_report(
    reports: list[dict[str, Any]],
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    meta = meta or {}
    lines: list[str] = []
    lines.append("# AI Employee — realtime test-scenario report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- Brain: `{meta.get('brain', 'mock')}`  Provider: `{meta.get('provider', '?')}`")
    lines.append(f"- Scenarios: {len(reports)}")
    if meta.get("note"):
        lines.append(f"- Note: {meta['note']}")
    lines.append("")

    counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append(
        f"**Summary:** {counts['pass']} pass · {counts['partial']} partial · "
        f"{counts['fail']} fail · {counts['error']} error"
    )
    lines.append("")

    for r in reports:
        lines.append(f"## {r['scenario_id']} — {r['title']}")
        lines.append("")
        lines.append(
            f"- **Status:** `{r['status']}`  ({r['passed']}/{r['total']} checks, "
            f"score {r['score']}, {r['elapsed_s']}s)"
        )
        lines.append(f"- **Planned agents:** {', '.join(r['planned_agents']) or '—'}")
        lines.append(f"- **Executed agents:** {', '.join(r['executed_agents']) or '—'}")
        lines.append(f"- **Tools used:** {', '.join(r['tools']) or '—'}")
        if r.get("needs_approval"):
            lines.append(f"- **Paused for approval:** `{r['needs_approval']}`")
        if r.get("created"):
            lines.append(f"- **Files created:** {', '.join(r['created'])}")
        if r.get("changed"):
            lines.append(f"- **Files changed:** {', '.join(r['changed'])}")
        if r.get("error"):
            lines.append(f"- **Error:** {r['error']}")
        lines.append("")
        lines.append("| Check | Result | Detail |")
        lines.append("|---|---|---|")
        for c in r["checks"]:
            detail = c["detail"].replace("|", "\\|")
            lines.append(f"| `{c['name']}` | {_icon(c['ok'])} | {detail} |")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "snapshot",
    "run_scenario",
    "run_all",
    "evaluate_scenario",
    "render_report",
]
