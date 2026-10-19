"""Scheduled tasks and event watchers — the employee works while you're away.

Three kinds of entries, stored in ``data/schedules.json`` and managed from the
console's ⏰ tab:

- **recurring** — run a goal daily at ``HH:MM`` (optionally only on some
  weekdays), or every N minutes.
- **once** — run a goal at one future moment, then disable itself.
- **gmail_watch** — poll a Gmail query every N minutes; when *new* messages
  appear, fire the goal with the new message ids appended (the first poll only
  baselines what already exists — it never storms through the backlog).

A single asyncio loop (started in the FastAPI lifespan) ticks every
``settings.scheduler_tick_seconds``; due entries start ordinary tasks through
the TaskManager, so scheduled runs show up in the console, pause for approval,
notify the phone, and persist exactly like hand-assigned ones. A schedule
whose previous task is still running (or paused for approval) is skipped until
that task finishes — no pileups.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("schedules")

_ACTIVE_STATUSES = {
    "pending",
    "running",
    "cooling",
    "awaiting_approval",
    "waiting_for_user",
}
_SEEN_CAP = 500


# ── Store ────────────────────────────────────────────────────────────────────
def _path(settings: Any) -> Path:
    return Path(settings.schedules_file)


def load_schedules(settings: Any) -> dict[str, dict[str, Any]]:
    p = _path(settings)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        sched = data.get("schedules") if isinstance(data, dict) else None
        return sched if isinstance(sched, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a corrupt store must not crash startup
        log.warning("schedules.load_failed", error=str(exc))
        return {}


def _save(settings: Any, schedules: dict[str, dict[str, Any]]) -> None:
    p = _path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schedules": schedules}, indent=2))


def create_schedule(settings: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate + persist a new schedule; returns the stored entry."""
    kind = spec.get("type")
    if kind not in ("recurring", "once", "gmail_watch"):
        raise ValueError(f"unknown schedule type: {kind!r}")
    goal = str(spec.get("goal") or "").strip()
    if not goal:
        raise ValueError("goal is required")
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "name": str(spec.get("name") or goal[:60]).strip(),
        "type": kind,
        "goal": goal,
        "auto_approve": bool(spec.get("auto_approve")),
        # Who created it — fired tasks land in this user's saved-task list.
        "user_id": spec.get("user_id"),
        "user_name": spec.get("user_name"),
        "enabled": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "last_run": None,
        "last_task_id": None,
        "last_error": None,
        "runs": 0,
    }
    if kind == "recurring":
        every = spec.get("every_minutes")
        at = str(spec.get("at") or "").strip()
        if every:
            entry["every_minutes"] = max(1, int(every))
        elif _valid_hhmm(at):
            entry["at"] = at
            days = spec.get("days")
            if isinstance(days, list) and days:
                entry["days"] = sorted({int(d) % 7 for d in days})
        else:
            raise ValueError("recurring needs 'at' as HH:MM or 'every_minutes'")
    elif kind == "once":
        run_at = str(spec.get("run_at") or "").strip()
        try:
            datetime.fromisoformat(run_at)
        except ValueError as exc:
            raise ValueError("once needs 'run_at' as an ISO datetime") from exc
        entry["run_at"] = run_at
    else:  # gmail_watch
        query = str(spec.get("query") or "").strip()
        if not query:
            raise ValueError("gmail_watch needs a Gmail 'query'")
        entry["query"] = query
        entry["interval_minutes"] = max(1, int(spec.get("interval_minutes") or 5))
        entry["seen_ids"] = None  # None = not yet baselined
    entry["next_run"] = _next_run(entry, datetime.now())
    schedules = load_schedules(settings)
    schedules[entry["id"]] = entry
    _save(settings, schedules)
    log.info("schedules.created", id=entry["id"], type=kind, name=entry["name"])
    return entry


def update_schedule(settings: Any, schedule_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    schedules = load_schedules(settings)
    if schedule_id not in schedules:
        raise KeyError(schedule_id)
    entry = schedules[schedule_id]
    for field in ("enabled", "name", "goal", "auto_approve", "at", "every_minutes",
                  "days", "run_at", "query", "interval_minutes"):
        if field in patch and patch[field] is not None:
            entry[field] = patch[field]
    entry["next_run"] = _next_run(entry, datetime.now())
    _save(settings, schedules)
    return entry


def delete_schedule(settings: Any, schedule_id: str) -> bool:
    schedules = load_schedules(settings)
    if schedules.pop(schedule_id, None) is None:
        return False
    _save(settings, schedules)
    log.info("schedules.deleted", id=schedule_id)
    return True


def _valid_hhmm(v: str) -> bool:
    return bool(re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", v))


def _next_run(entry: dict[str, Any], now: datetime) -> str | None:
    """When this entry should next fire (ISO), or None when it never will."""
    if entry["type"] == "once":
        return entry.get("run_at")
    if entry["type"] == "gmail_watch":
        base = entry.get("last_run")
        start = datetime.fromisoformat(base) if base else now
        return (start + timedelta(minutes=int(entry.get("interval_minutes") or 5))).isoformat(
            timespec="seconds"
        )
    if entry.get("every_minutes"):
        base = entry.get("last_run")
        start = datetime.fromisoformat(base) if base else now
        return (start + timedelta(minutes=int(entry["every_minutes"]))).isoformat(
            timespec="seconds"
        )
    at = entry.get("at")
    if not at:
        return None
    hh, mm = (int(x) for x in at.split(":"))
    days = entry.get("days") or list(range(7))
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    for _ in range(8):  # today + a full week covers every day mask
        if candidate > now and candidate.weekday() in days:
            return candidate.isoformat(timespec="seconds")
        candidate += timedelta(days=1)
    return None


# ── Runner ───────────────────────────────────────────────────────────────────
class SchedulerRunner:
    """Background loop that fires due schedules through the TaskManager."""

    def __init__(self, settings: Any, get_employee: Any, task_manager: Any) -> None:
        self._settings = settings
        self._get_employee = get_employee
        self._tm = task_manager
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())
        log.info("schedules.runner_started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(max(5, int(self._settings.scheduler_tick_seconds)))
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                log.warning("schedules.tick_failed", error=str(exc))

    async def tick(self, now: datetime | None = None) -> list[str]:
        """Run everything due; returns the ids of schedules that fired."""
        now = now or datetime.now()
        fired: list[str] = []
        schedules = load_schedules(self._settings)
        for sid, entry in schedules.items():
            if not entry.get("enabled"):
                continue
            try:
                if self._previous_still_active(entry):
                    continue
                if entry["type"] == "gmail_watch":
                    if self._due(entry, now) and await self._check_gmail(entry, now):
                        fired.append(sid)
                elif self._due(entry, now):
                    self._fire(entry, entry["goal"], now)
                    if entry["type"] == "once":
                        entry["enabled"] = False
                    fired.append(sid)
            except Exception as exc:  # noqa: BLE001 - isolate per schedule
                entry["last_error"] = f"{type(exc).__name__}: {exc}"
                log.warning("schedules.entry_failed", id=sid, error=str(exc))
        if fired or schedules:
            _save(self._settings, schedules)
        return fired

    def _due(self, entry: dict[str, Any], now: datetime) -> bool:
        nxt = entry.get("next_run")
        if not nxt:
            return False
        try:
            return datetime.fromisoformat(nxt) <= now
        except ValueError:
            return False

    def _previous_still_active(self, entry: dict[str, Any]) -> bool:
        last = entry.get("last_task_id")
        if not last:
            return False
        task = self._tm.get_task(last)
        return bool(task) and task.status.value in _ACTIVE_STATUSES

    def _fire(self, entry: dict[str, Any], goal: str, now: datetime) -> None:
        emp = self._get_employee()
        owner = None
        if entry.get("user_id"):
            owner = {
                "id": entry["user_id"],
                "name": entry.get("user_name"),
                # Guest-created schedules keep firing, but their runs stay
                # memory-only like every other guest session.
                "guest": str(entry["user_id"]).startswith("guest:"),
            }
        task = self._tm.start(
            emp,
            goal,
            auto_approve=bool(entry.get("auto_approve")),
            owner=owner,
        )
        entry["last_run"] = now.isoformat(timespec="seconds")
        entry["last_task_id"] = task.id
        entry["last_error"] = None
        entry["runs"] = int(entry.get("runs") or 0) + 1
        entry["next_run"] = _next_run(entry, now)
        log.info(
            "schedules.fired", id=entry["id"], name=entry["name"], task_id=task.id
        )

    async def _check_gmail(self, entry: dict[str, Any], now: datetime) -> bool:
        """Poll the watch query; fire only on messages not seen before."""
        emp = self._get_employee()
        result = await emp.tools.execute(
            "gmail_search",
            {"query": entry["query"], "max_results": 20},
        )
        if not result.ok:
            raise RuntimeError(result.error or "gmail_search failed")
        ids = re.findall(r"^\[([0-9a-f]+)\]", str(result.output), flags=re.MULTILINE)
        seen = entry.get("seen_ids")
        if seen is None:
            # First poll: baseline the existing backlog, don't fire on it.
            entry["seen_ids"] = ids[:_SEEN_CAP]
            entry["last_run"] = now.isoformat(timespec="seconds")
            entry["next_run"] = _next_run(entry, now)
            log.info("schedules.watch_baselined", id=entry["id"], existing=len(ids))
            return False
        new = [i for i in ids if i not in set(seen)]
        entry["seen_ids"] = (new + list(seen))[:_SEEN_CAP]
        if not new:
            entry["last_run"] = now.isoformat(timespec="seconds")
            entry["next_run"] = _next_run(entry, now)
            return False
        goal = (
            f"{entry['goal']}\n\nTriggered by {len(new)} new email(s) matching "
            f"Gmail query '{entry['query']}'. New message id(s): {', '.join(new)}. "
            "Use gmail_read with these ids to read them before acting."
        )
        self._fire(entry, goal, now)
        return True


async def run_schedule(
    settings: Any, schedule_id: str, employee: Any, task_manager: Any
) -> tuple[bool, str | None]:
    """Fire one schedule immediately (a fresh gmail_watch baselines instead).

    Returns ``(fired, task_id)``. Raises :class:`KeyError` for an unknown id.
    """
    schedules = load_schedules(settings)
    entry = schedules[schedule_id]
    runner = SchedulerRunner(settings, lambda: employee, task_manager)
    now = datetime.now()
    if entry["type"] == "gmail_watch":
        fired = await runner._check_gmail(entry, now)  # noqa: SLF001 - same module
    else:
        runner._fire(entry, entry["goal"], now)  # noqa: SLF001 - same module
        fired = True
    schedules[schedule_id] = entry
    _save(settings, schedules)
    return fired, entry.get("last_task_id")


__all__ = [
    "SchedulerRunner",
    "create_schedule",
    "delete_schedule",
    "load_schedules",
    "run_schedule",
    "update_schedule",
]
