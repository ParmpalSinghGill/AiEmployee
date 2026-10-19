"""Reconstruct a run from the structured server log, to recover a lost task.

Runs are normally saved to ``data/sessions`` — but a guest's run historically was
kept in memory only, so a server restart (or a crash) lost it. The full structured
event stream is still in ``data/server.log``; this module parses it back into a
task snapshot the console can open, so a run interrupted by a restart can be
inspected instead of re-run.

Best-effort by design: values are parsed from structlog's console rendering
(Python-literal field values), so a field that can't be parsed is kept as its raw
string rather than dropped.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("recover")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RECORD = re.compile(r"^(\d{4}-\d\d-\d\dT\S+)\s+\[(\w+)\s*\]\s+(\S+)\s*(.*)$")
_KV = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=")
_FAIL = object()


def _literal(raw: str) -> Any:
    """Parse a structlog-rendered value; return the raw string if it isn't a literal."""
    raw = raw.strip()
    if raw == "":
        return ""
    if raw[0] in "\"'[{(":
        try:
            return ast.literal_eval(raw)
        except Exception:  # noqa: BLE001 - truncated/merged literal
            return _FAIL
    try:
        return ast.literal_eval(raw)
    except Exception:  # noqa: BLE001 - bare token (e.g. an agent name)
        return raw


def _parse_fields(rest: str) -> dict[str, Any]:
    rest = rest.rstrip()
    marks = list(_KV.finditer(rest))
    fields: dict[str, Any] = {}
    i = 0
    while i < len(marks):
        key = marks[i].group(1)
        start = marks[i].end()
        j = i + 1
        while True:
            end = marks[j].start() if j < len(marks) else len(rest)
            value = _literal(rest[start:end])
            if value is not _FAIL:
                break
            if j >= len(marks):
                value = rest[start:end].strip()
                break
            j += 1
        fields[key] = value
        i = j
    return fields


def parse_events(path: str | Path) -> list[dict[str, Any]]:
    """Every structlog record in ``path`` as ``{ts, level, event, fields}``."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _RECORD.match(_ANSI.sub("", line))
        if not m:
            continue
        ts, level, event, rest = m.groups()
        out.append({"ts": ts, "level": level.lower(), "event": event, "fields": _parse_fields(rest)})
    return out


@dataclass
class RecoveredRun:
    task_id: str
    session_id: str
    goal: str
    events: list[dict[str, Any]]
    result: dict[str, Any]
    created_at: str
    finished_at: str
    status: str = "completed"

    def snapshot(self) -> dict[str, Any]:
        steps = self.result.get("steps") or []
        return {
            "id": self.task_id,
            "goal": self.goal,
            "session_id": self.session_id,
            "parent_id": None,
            "mode": "task",
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "events": [
                {
                    "seq": i + 1,
                    "ts": e["ts"],
                    "level": e["level"],
                    "event": e["event"],
                    "fields": e["fields"],
                }
                for i, e in enumerate(self.events)
            ],
            "last_seq": len(self.events),
            "result": self.result,
            "error": None,
            "approved_ops": [],
            "resource_decision": None,
            "resumable": False,
            "can_resume": False,
            "progress_saved_at": None,
            "completed_steps": len(steps),
            "planned_steps": len((self.result.get("plan") or {}).get("steps") or []),
            "auto_approve": False,
            "token_budget": 0,
            "tokens_used": 0,
        }


def _build_result(events: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    plan: dict[str, Any] = {}
    answer = ""
    confidence: float | None = None
    step_output: dict[Any, dict[str, Any]] = {}
    step_verified: dict[Any, bool] = {}
    step_retries: dict[Any, int] = {}

    for e in events:
        f = e["fields"]
        if e["event"] == "planner.plan_ready" and isinstance(f.get("plan"), list):
            plan = {"steps": f["plan"], "reasoning": f.get("reasoning", "")}
        elif e["event"] == "brain.response" and f.get("purpose") == "synthesize":
            if isinstance(f.get("content"), str) and f["content"].strip():
                answer = f["content"]
        elif e["event"] == "task.completed":
            confidence = f.get("confidence")
        elif e["event"] == "agent.answer" and f.get("step") is not None:
            step_output[f["step"]] = {
                "output": f.get("output", ""),
                "agent": f.get("agent", ""),
                "note": f.get("note"),
            }
            step_retries[f["step"]] = max(step_retries.get(f["step"], 0), int(f.get("retry") or 0))
        elif e["event"] == "verifier.verdict" and f.get("step") is not None:
            step_verified[f["step"]] = bool(f.get("passed"))

    plan_steps = plan.get("steps") or []
    steps = []
    for i, s in enumerate(plan_steps):
        sid = s.get("id", i + 1)
        ans = step_output.get(sid, {})
        steps.append(
            {
                "id": sid,
                "description": s.get("description", ""),
                "agent": s.get("agent") or ans.get("agent", ""),
                "output": ans.get("output", ""),
                "confidence": None,
                "verified": step_verified.get(sid, False),
                "retries": step_retries.get(sid, 0),
            }
        )
    return {
        "goal": events[0]["fields"].get("goal", "") if events else "",
        "answer": answer,
        "confidence": confidence,
        "plan": plan or None,
        "steps": steps,
        "agents_used": [],
        "tools_used": [],
        "recovered_from_log": True,
    }


def recover_runs(path: str | Path) -> list[RecoveredRun]:
    """Group the log's events into runs, newest first."""
    events = parse_events(path)
    by_task: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for e in events:
        tid = e["fields"].get("task_id")
        if not tid:
            continue
        if tid not in by_task:
            by_task[tid] = []
            order.append(tid)
        by_task[tid].append(e)

    runs: list[RecoveredRun] = []
    for tid in order:
        evs = by_task[tid]
        goal = next(
            (e["fields"].get("goal") for e in evs if e["event"] == "task.started"), ""
        )
        session_id = next((e["fields"].get("session") for e in evs if e["fields"].get("session")), "")
        result = _build_result(evs, tid)
        if not goal:
            goal = result.get("goal") or f"(recovered run {tid})"
        status = "completed"
        if any(e["event"] == "task.cancelled" for e in evs):
            status = "cancelled"
        elif any(e["event"] == "task.failed" for e in evs):
            status = "failed"
        runs.append(
            RecoveredRun(
                task_id=tid,
                session_id=session_id or f"recovered-{tid}",
                goal=goal,
                events=evs,
                result=result,
                created_at=evs[0]["ts"],
                finished_at=evs[-1]["ts"],
                status=status,
            )
        )
    return list(reversed(runs))


def import_run(store: Any, run: RecoveredRun, *, owner_id: str | None, owner_name: str | None) -> dict:
    """Write a recovered run to the session store so the console can list/open it."""
    store.save_session(
        {
            "id": run.session_id,
            "title": (run.goal[:42] + "…") if len(run.goal) > 42 else (run.goal or "Recovered run"),
            "created_at": run.created_at,
            "run_ids": [run.task_id],
            "user_id": owner_id,
            "user_name": owner_name,
            "recovered": True,
        }
    )
    store.save_task(run.session_id, run.snapshot())
    log.info("recover.imported", task_id=run.task_id, session_id=run.session_id)
    return {"session_id": run.session_id, "task_id": run.task_id, "events": len(run.events)}


def existing_task_ids(store: Any) -> set[str]:
    try:
        _metas, snaps = store.load_all()
    except Exception:  # noqa: BLE001
        return set()
    return set(snaps)


def import_all_new(
    store: Any,
    log_path: str | Path,
    *,
    owner_id: str | None,
    owner_name: str | None,
) -> list[tuple[dict, dict]]:
    """Import runs present in the log but not yet on disk (oldest first).

    Returns ``[(session_meta, snapshot), …]`` so the caller can also register them
    with an in-memory TaskManager.
    """
    have = existing_task_ids(store)
    imported: list[tuple[dict, dict]] = []
    for run in reversed(recover_runs(log_path)):  # oldest → newest
        if run.task_id in have:
            continue
        title = (run.goal[:42] + "…") if len(run.goal) > 42 else (run.goal or "Recovered run")
        meta = {
            "id": run.session_id,
            "title": title,
            "created_at": run.created_at,
            "run_ids": [run.task_id],
            "user_id": owner_id,
            "user_name": owner_name,
            "recovered": True,
        }
        snap = run.snapshot()
        store.save_session(meta)
        store.save_task(run.session_id, snap)
        imported.append((meta, snap))
        log.info("recover.imported", task_id=run.task_id, session_id=run.session_id)
    return imported


__all__ = [
    "RecoveredRun",
    "parse_events",
    "recover_runs",
    "import_run",
    "import_all_new",
    "existing_task_ids",
]
