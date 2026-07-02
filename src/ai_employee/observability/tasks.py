"""Concurrent task runner + live event capture for the dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from ai_employee.core.logging import add_log_observer, get_logger

log = get_logger("tasks")

# Log "event" names that represent meaningful flow milestones (vs. chatter).
_FLOW_EVENTS = {
    "workflow.planned",
    "workflow.resume",
    "workflow.resume_skip_plan",
    "planner.plan_ready",
    "agent.start",
    "agent.needs_approval",
    "approval.granted",
    "tool.execute",
    "tool.result",
    "tool.duplicate_blocked",
    "tool.approval_required",
    "verifier.verdict",
    "workflow.node",
    "workflow.retry",
    "rag.ingested",
    "memory.degraded_to_in_memory",
}


def _now() -> datetime:
    return datetime.now(UTC)


def _uid() -> str:
    return uuid.uuid4().hex[:12]


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    awaiting_approval = "awaiting_approval"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskEvent(BaseModel):
    seq: int
    ts: str
    level: str = "info"
    event: str
    # Remaining structured fields (agent, tool, step, node, ok, latency_ms, …).
    fields: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_flow(self) -> bool:
        return self.event in _FLOW_EVENTS


_RESERVED = {"event", "level", "timestamp", "task_id", "logger", "logger_name"}


class TaskRecord:
    """One execution of a goal (initial task or a follow-up within a session)."""

    def __init__(
        self,
        *,
        goal: str,
        session_id: str,
        parent_id: str | None = None,
        mode: str = "task",
    ) -> None:
        self.id = _uid()
        self.goal = goal
        self.session_id = session_id
        self.parent_id = parent_id
        self.mode = mode
        self.status = TaskStatus.pending
        self.created_at = _now()
        self.finished_at: datetime | None = None
        self.events: list[TaskEvent] = []
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self._seq = 0
        # Operations the user has approved for this run (persisted so an approval
        # survives a restart). ``resumable`` marks a run interrupted mid-flight.
        self.approved_ops: set[str] = set()
        self.resource_decision: dict[str, Any] | None = None
        self.resumable = False

    def add_log(self, event_dict: dict[str, Any]) -> None:
        self._seq += 1
        fields = {k: v for k, v in event_dict.items() if k not in _RESERVED}
        self.events.append(
            TaskEvent(
                seq=self._seq,
                ts=str(event_dict.get("timestamp", _now().isoformat())),
                level=str(event_dict.get("level", "info")),
                event=str(event_dict.get("event", "")),
                fields=_jsonable(fields),
            )
        )

    def snapshot(self, *, since: int = 0) -> dict[str, Any]:
        events = [e.model_dump() for e in self.events if e.seq > since]
        return {
            "id": self.id,
            "goal": self.goal,
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "mode": self.mode,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "events": events,
            "last_seq": self._seq,
            "result": self.result,
            "error": self.error,
            "approved_ops": sorted(self.approved_ops),
            "resource_decision": self.resource_decision,
        }

    @classmethod
    def from_snapshot(cls, d: dict[str, Any]) -> TaskRecord:
        """Rebuild a record loaded from disk. Non-terminal statuses are clamped
        (a 'running' task from a previous process is no longer live)."""
        rec = cls(
            goal=d["goal"],
            session_id=d["session_id"],
            parent_id=d.get("parent_id"),
            mode=d.get("mode", "task"),
        )
        rec.id = d["id"]
        rec.events = [TaskEvent(**e) for e in d.get("events", [])]
        rec._seq = d.get("last_seq", len(rec.events))
        rec.result = d.get("result")
        rec.error = d.get("error")
        rec.approved_ops = set(d.get("approved_ops", []))
        rd = d.get("resource_decision")
        rec.resource_decision = rd if isinstance(rd, dict) else None
        status = d.get("status", "completed")
        if status in {"running", "pending"}:
            if rec.result:
                status = "completed"
            elif not rec.events:
                # Queued but never actually ran (process died before it started).
                # Keep it 'pending' + resumable so startup can relaunch it safely.
                status = "pending"
                rec.resumable = True
            else:
                # Interrupted mid-flight — don't blindly re-run a partial execution.
                status = "failed"
                if not rec.error:
                    rec.error = "interrupted (process restarted)"
        rec.status = TaskStatus(status)
        return rec


class _Session:
    def __init__(self, title: str) -> None:
        self.id = _uid()
        self.title = title
        self.created_at = _now()
        self.task_ids: list[str] = []


class TaskManager:
    """Owns sessions + tasks and routes live log events to the right task."""

    def __init__(self, store: Any = None) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._sessions: dict[str, _Session] = {}
        self._handles: dict[str, asyncio.Task] = {}
        self._bg: set[asyncio.Task] = set()  # keep refs to fire-and-forget tasks
        self._store = store
        # A single observer routes every log event by its bound task_id.
        add_log_observer(self._route_log)
        if store is not None:
            self._load_from_store()

    def _load_from_store(self) -> None:
        try:
            metas, snaps = self._store.load_all()
        except Exception as exc:  # noqa: BLE001
            log.warning("tasks.load_failed", error=str(exc))
            return
        for d in snaps.values():
            self._tasks[d["id"]] = TaskRecord.from_snapshot(d)
        for meta in metas:
            s = _Session(meta.get("title", "Saved task"))
            s.id = meta["id"]
            s.task_ids = [t for t in meta.get("run_ids", []) if t in self._tasks]
            created = meta.get("created_at")
            if created:
                with contextlib.suppress(Exception):
                    s.created_at = datetime.fromisoformat(created)
            self._sessions[s.id] = s

    # ── Log routing ──────────────────────────────────────────────────────────
    def _route_log(self, event_dict: dict[str, Any]) -> None:
        tid = event_dict.get("task_id")
        if tid and tid in self._tasks:
            event = str(event_dict.get("event", ""))
            # Keep flow milestones + warnings/errors; drop low-signal debug chatter.
            level = str(event_dict.get("level", "info"))
            if event in _FLOW_EVENTS or level in {"warning", "error", "critical"} or event.startswith(
                ("workflow", "agent", "tool", "planner", "verifier", "brain.request", "brain.response")
            ):
                self._tasks[tid].add_log(event_dict)

    # ── Creation / execution ─────────────────────────────────────────────────
    def start(
        self,
        employee: Any,
        goal: str,
        *,
        session_id: str | None = None,
        approved_ops: set[str] | None = None,
        mode: str = "task",
    ) -> TaskRecord:
        """Create a task (new session unless ``session_id`` given) and run it in
        the background. Returns immediately. ``mode`` selects the pipeline:
        ``task`` (general) or ``research`` (model/dataset discovery)."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            parent_id = session.task_ids[-1] if session.task_ids else None
        else:
            session = _Session(title=_title(goal))
            self._sessions[session.id] = session
            parent_id = None

        task = TaskRecord(goal=goal, session_id=session.id, parent_id=parent_id, mode=mode)
        task.approved_ops = set(approved_ops or set())
        self._tasks[task.id] = task
        session.task_ids.append(task.id)
        self._persist_session(session)
        self._persist_task(task)

        self._launch(employee, task)
        return task

    def _launch(self, employee: Any, task: TaskRecord, *, approval: dict | None = None) -> None:
        """(Re)start a task's execution in the background, tracking its handle."""
        handle = asyncio.create_task(self._execute(employee, task, approval=approval))
        self._handles[task.id] = handle
        handle.add_done_callback(lambda _h, tid=task.id: self._handles.pop(tid, None))

    def approve(
        self,
        employee: Any,
        task_id: str,
        *,
        resource_decision: dict[str, Any] | None = None,
    ) -> TaskRecord | None:
        """Approve a paused task and resume it *in place* (same run/tab), rather
        than spawning a new run. Returns None if the task isn't awaiting approval."""
        task = self._tasks.get(task_id)
        if task is None or task.status != TaskStatus.awaiting_approval:
            return None
        approval = (task.result or {}).get("needs_approval") or {}
        op = approval.get("operation")
        if op:
            task.approved_ops = set(task.approved_ops) | {op}
        if resource_decision:
            task.resource_decision = resource_decision
        self._persist_task(task)
        # Flip to running synchronously so a poll racing the background task
        # doesn't see the (terminal) awaiting_approval status and stop polling.
        task.status = TaskStatus.running
        self._persist_task(task)
        self._launch(employee, task, approval=approval)
        return task

    def resume_interrupted(self, employee: Any) -> list[str]:
        """Re-launch tasks left ``pending`` by a previous process (e.g. a queued
        approval that never ran). Call once at startup. Returns the ids resumed."""
        resumed: list[str] = []
        for task in list(self._tasks.values()):
            if task.resumable and task.status == TaskStatus.pending:
                task.resumable = False
                task.status = TaskStatus.running
                self._launch(employee, task)
                resumed.append(task.id)
        if resumed:
            log.info("tasks.resumed_interrupted", count=len(resumed), ids=resumed)
        return resumed

    def cancel(self, task_id: str) -> bool:
        """Stop a running/pending task. Returns True if a live run was cancelled.

        The underlying asyncio task is cancelled, which raises ``CancelledError``
        inside ``_execute`` — that handler marks the record ``cancelled`` and
        persists it. No-op (returns False) if the task is unknown or already
        terminal."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        handle = self._handles.get(task_id)
        if handle is None or handle.done():
            return False
        handle.cancel()
        return True

    async def _execute(
        self, employee: Any, task: TaskRecord, *, approval: dict | None = None
    ) -> None:
        # Bind task_id so every nested log event is attributed to this task. This
        # runs in its own asyncio Task context, so concurrent runs stay isolated.
        structlog.contextvars.bind_contextvars(task_id=task.id)
        approved_ops = set(task.approved_ops)
        task.status = TaskStatus.running
        log.info("task.started", goal=task.goal[:120])
        # Record what the user just approved, inline in the flow/logs/explain, so
        # it's clear which sensitive op was granted and when.
        if approval:
            log.info(
                "approval.granted",
                op=approval.get("operation"),
                tool=approval.get("tool"),
                args=approval.get("args"),
            )
        # Generate a nicer session title in the background (initial task only).
        if task.parent_id is None:
            self._spawn_name(employee, task)
        try:
            checkpoint = (task.result or {}).get("workflow_checkpoint")
            resume = bool(checkpoint and (approval or task.approved_ops))
            if task.mode == "research":
                result = await employee.research(
                    task.goal,
                    session_id=task.session_id,
                    approved_ops=approved_ops,
                    auto_run="run_research_code" in approved_ops,
                )
            else:
                result = await employee.run(
                    task.goal,
                    session_id=task.session_id,
                    approved_ops=approved_ops,
                    resource_decision=task.resource_decision,
                    workflow_checkpoint=checkpoint if resume else None,
                )
            task.result = result
            if result.get("needs_approval"):
                task.status = TaskStatus.awaiting_approval
            else:
                task.status = TaskStatus.completed
            log.info("task.completed", confidence=result.get("confidence"))
        except asyncio.CancelledError:
            # User-requested stop (handle.cancel()). Record it as a terminal
            # state and let the cancellation propagate so the asyncio task ends.
            task.status = TaskStatus.cancelled
            task.error = "cancelled by user"
            log.info("task.cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - surface failure to the UI
            task.status = TaskStatus.failed
            task.error = f"{type(exc).__name__}: {exc}"
            log.warning("task.failed", error=task.error)
        finally:
            task.finished_at = _now()
            structlog.contextvars.unbind_contextvars("task_id")
            self._persist_task(task)
            # Persist memory so a resumed session keeps its accumulated context.
            with contextlib.suppress(Exception):
                employee.persist_memory()

    # ── Auto-naming ──────────────────────────────────────────────────────────
    def _spawn_name(self, employee: Any, task: TaskRecord) -> None:
        handle = asyncio.create_task(self._name_session(employee, task))
        self._bg.add(handle)
        handle.add_done_callback(self._bg.discard)

    async def _name_session(self, employee: Any, task: TaskRecord) -> None:
        try:
            title = await employee.name_task(task.goal)
        except Exception:  # noqa: BLE001
            return
        session = self._sessions.get(task.session_id)
        if session is not None and title:
            session.title = title
            self._persist_session(session)
            log.info("task.named", title=title, task_id=task.id)

    # ── Persistence ──────────────────────────────────────────────────────────
    def _persist_session(self, session: _Session) -> None:
        if self._store is None:
            return
        with contextlib.suppress(Exception):
            self._store.save_session(
                {
                    "id": session.id,
                    "title": session.title,
                    "created_at": session.created_at.isoformat(),
                    "run_ids": list(session.task_ids),
                }
            )

    def _persist_task(self, task: TaskRecord) -> None:
        if self._store is None:
            return
        with contextlib.suppress(Exception):
            self._store.save_task(task.session_id, task.snapshot())

    def delete_session(self, session_id: str) -> None:
        s = self._sessions.pop(session_id, None)
        if s:
            for tid in s.task_ids:
                self._tasks.pop(tid, None)
        if self._store is not None:
            with contextlib.suppress(Exception):
                self._store.delete_session(session_id)

    def delete_all_sessions(self) -> list[str]:
        """Remove every saved session/task from memory and disk."""
        ids = list(self._sessions.keys())
        for sid in ids:
            self.delete_session(sid)
        return ids

    # ── Queries ──────────────────────────────────────────────────────────────
    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        out = []
        for s in sorted(self._sessions.values(), key=lambda x: x.created_at):
            runs = [self._tasks[t] for t in s.task_ids if t in self._tasks]
            out.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "runs": [
                        {
                            "id": r.id,
                            "goal": r.goal,
                            "status": r.status.value,
                            "created_at": r.created_at.isoformat(),
                        }
                        for r in runs
                    ],
                }
            )
        return out

    def session_title(self, session_id: str) -> str | None:
        s = self._sessions.get(session_id)
        return s.title if s else None

    def session_runs(self, session_id: str) -> list[TaskRecord]:
        s = self._sessions.get(session_id)
        if not s:
            return []
        return [self._tasks[t] for t in s.task_ids if t in self._tasks]

    def session_detail(self, session_id: str) -> dict[str, Any] | None:
        """Full session payload (meta + every run's snapshot) for opening/resuming."""
        s = self._sessions.get(session_id)
        if not s:
            return None
        runs = [self._tasks[t] for t in s.task_ids if t in self._tasks]
        return {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "runs": [r.snapshot() for r in runs],
        }


def _title(goal: str) -> str:
    g = goal.strip().replace("\n", " ")
    return (g[:42] + "…") if len(g) > 42 else g or "Untitled task"


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None or isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


_MANAGER: TaskManager | None = None


def get_task_manager() -> TaskManager:
    global _MANAGER
    if _MANAGER is None:
        from ai_employee.config import get_settings

        s = get_settings()
        store = None
        if s.persist_sessions:
            from ai_employee.observability.store import SessionStore

            store = SessionStore(s.sessions_dir)
        _MANAGER = TaskManager(store)
    return _MANAGER
