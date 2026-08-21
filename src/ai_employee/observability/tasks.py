"""Concurrent task runner + live event capture for the dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
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
    waiting_for_user = "waiting_for_user"
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
        # Last time a mid-run checkpoint was stored (used to show resume freshness).
        self.progress_saved_at: datetime | None = None
        # When True, never pause for approval — grant every gated op automatically.
        self.auto_approve: bool = False

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
            "resumable": self.resumable,
            "can_resume": self.can_resume(),
            "progress_saved_at": (
                self.progress_saved_at.isoformat() if self.progress_saved_at else None
            ),
            "completed_steps": len(
                ((self.result or {}).get("workflow_checkpoint") or {}).get("step_results") or []
            ),
            "planned_steps": len(
                (((self.result or {}).get("workflow_checkpoint") or {}).get("plan") or {}).get(
                    "steps"
                )
                or []
            ),
            "auto_approve": self.auto_approve,
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
        rec.resumable = bool(d.get("resumable", False))
        rec.auto_approve = bool(d.get("auto_approve", False))
        status = d.get("status", "completed")
        pending_approval = _pending_approval(rec.result)
        resume_checkpoint = _resume_checkpoint(rec.result)
        if status in {"running", "pending"}:
            if pending_approval:
                op = pending_approval.get("operation")
                status = "waiting_for_user" if op == "waiting_for_user" else "awaiting_approval"
            elif resume_checkpoint:
                # Interrupted after a checkpoint was saved (for example, the server
                # was closed right after approval). Surface it as resumable rather
                # than pretending it completed.
                status = "failed"
                rec.resumable = True
                if not rec.error:
                    rec.error = "interrupted (resume available)"
            elif rec.result:
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

    def can_resume(self) -> bool:
        """True when a saved checkpoint can continue this run in place."""
        return self.resumable and _resume_checkpoint(self.result) is not None


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
            if (
                event in _FLOW_EVENTS
                or level in {"warning", "error", "critical"}
                or event.startswith(
                    (
                        "workflow",
                        "agent",
                        "tool",
                        "planner",
                        "verifier",
                        "brain.request",
                        "brain.response",
                    )
                )
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
        auto_approve: bool = False,
    ) -> TaskRecord:
        """Create a task (new session unless ``session_id`` given) and run it in
        the background. Returns immediately. ``mode`` selects the pipeline:
        ``task`` (general), ``research`` (model/dataset discovery), ``plan``
        (how-to guide with tools), or ``download`` (direct URL downloader)."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            parent_id = session.task_ids[-1] if session.task_ids else None
        else:
            session = _Session(title=_title(goal))
            self._sessions[session.id] = session
            parent_id = None

        task = TaskRecord(goal=goal, session_id=session.id, parent_id=parent_id, mode=mode)
        task.auto_approve = bool(auto_approve)
        ops = set(approved_ops or set())
        if task.auto_approve:
            ops |= _auto_approve_ops()
        task.approved_ops = ops
        self._tasks[task.id] = task
        session.task_ids.append(task.id)
        self._persist_session(session)
        self._persist_task(task)

        self._launch(employee, task)
        return task

    def _launch(
        self,
        employee: Any,
        task: TaskRecord,
        *,
        approval: dict | None = None,
        resume: bool = False,
    ) -> None:
        """(Re)start a task's execution in the background, tracking its handle."""
        handle = asyncio.create_task(
            self._execute(employee, task, approval=approval, resume=resume)
        )
        self._handles[task.id] = handle
        handle.add_done_callback(lambda _h, tid=task.id: self._handles.pop(tid, None))

    def approve(
        self,
        employee: Any,
        task_id: str,
        *,
        resource_decision: dict[str, Any] | None = None,
        approve_all_similar: bool = True,
    ) -> TaskRecord | None:
        """Approve or respond to a paused task and resume it *in place*."""
        task = self._tasks.get(task_id)
        if task is None or task.status not in {
            TaskStatus.awaiting_approval,
            TaskStatus.waiting_for_user,
        }:
            return None
        approval = (task.result or {}).get("needs_approval") or {}
        op = approval.get("operation")
        task.approved_ops = set(task.approved_ops) | _expand_approved_ops(
            op,
            approval,
            approve_all_similar=approve_all_similar,
        )
        if resource_decision:
            task.resource_decision = resource_decision
        if task.result:
            cleared = dict(task.result)
            cleared["needs_approval"] = None
            task.result = cleared
        self._persist_task(task)
        task.status = TaskStatus.running
        task.resumable = False
        task.error = None
        task.finished_at = None
        self._persist_task(task)
        self._launch(employee, task, approval=approval, resume=True)
        return task

    def resume(self, employee: Any, task_id: str) -> TaskRecord | None:
        """Resume an interrupted run from its saved workflow checkpoint."""
        task = self._tasks.get(task_id)
        if task is None or not task.can_resume():
            return None
        handle = self._handles.get(task_id)
        if handle is not None and not handle.done():
            return None
        task.status = TaskStatus.running
        task.resumable = False
        task.error = None
        task.finished_at = None
        self._persist_task(task)
        self._launch(employee, task, resume=True)
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

    def _record_progress(self, task: TaskRecord, checkpoint: dict[str, Any]) -> None:
        """Store a mid-run checkpoint so an interrupted task stays resumable.

        Only the checkpoint is written — never an ``answer`` — so a partially
        finished run is never mistaken for a completed one.
        """
        if not checkpoint or not checkpoint.get("plan"):
            return
        result = dict(task.result or {})
        result["workflow_checkpoint"] = checkpoint
        result.setdefault("goal", task.goal)
        result.setdefault("steps", checkpoint.get("step_results") or [])
        result.setdefault("plan", checkpoint.get("plan"))
        task.result = result
        task.progress_saved_at = _now()

    async def _execute(
        self,
        employee: Any,
        task: TaskRecord,
        *,
        approval: dict | None = None,
        resume: bool = False,
    ) -> None:
        # Bind task_id so every nested log event is attributed to this task. This
        # runs in its own asyncio Task context, so concurrent runs stay isolated.
        structlog.contextvars.bind_contextvars(task_id=task.id)
        task.status = TaskStatus.running
        task.resumable = False
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
            # Auto-approve can grant mid-run and need another pipeline pass.
            for auto_round in range(24):
                checkpoint = (task.result or {}).get("workflow_checkpoint")
                should_resume = bool(checkpoint and (resume or auto_round > 0))
                session_context = "" if should_resume else self._build_session_context(task)
                if session_context and auto_round == 0:
                    prior = self._sessions.get(task.session_id)
                    log.info(
                        "task.session_context",
                        prior_runs=len(prior.task_ids) - 1 if prior else 0,
                        chars=len(session_context),
                    )
                approved_ops = set(task.approved_ops)
                if task.mode == "research":
                    result = await employee.research(
                        task.goal,
                        session_id=task.session_id,
                        approved_ops=approved_ops,
                        auto_run="run_research_code" in approved_ops,
                    )
                elif task.mode == "plan":
                    result = await employee.plan(
                        task.goal,
                        session_id=task.session_id,
                    )
                elif task.mode == "download":
                    result = await employee.download(
                        task.goal,
                        session_id=task.session_id,
                        approved_ops=approved_ops,
                        resource_decision=task.resource_decision,
                    )
                else:
                    kwargs: dict[str, Any] = {
                        "session_id": task.session_id,
                        "approved_ops": approved_ops,
                        "resource_decision": task.resource_decision,
                        "workflow_checkpoint": checkpoint if should_resume else None,
                        "session_context": session_context,
                    }
                    # Progress checkpoints keep an interrupted run resumable, but
                    # only pass the hook to a run() that accepts it — tests and
                    # custom employees may implement a narrower signature.
                    if _accepts_kwarg(employee.run, "on_progress"):
                        kwargs["on_progress"] = lambda cp, t=task: self._record_progress(t, cp)
                    result = await employee.run(task.goal, **kwargs)
                task.result = result
                pending = result.get("needs_approval")
                if not pending:
                    task.status = TaskStatus.completed
                    break
                if not task.auto_approve:
                    op = (pending or {}).get("operation")
                    if op == "waiting_for_user":
                        task.status = TaskStatus.waiting_for_user
                    else:
                        task.status = TaskStatus.awaiting_approval
                    break
                # Grant every related op and continue without pausing the UI.
                op = (pending or {}).get("operation")
                task.approved_ops = (
                    set(task.approved_ops)
                    | _expand_approved_ops(op, pending, approve_all_similar=True)
                    | _auto_approve_ops()
                )
                if op == "waiting_for_user" and not task.resource_decision:
                    task.resource_decision = {"decision": "proceed_without"}
                cleared = dict(result)
                cleared["needs_approval"] = None
                task.result = cleared
                log.info("task.auto_approved", op=op, round=auto_round + 1)
                resume = True
            else:
                # Safety: too many auto-approve rounds — surface the last pause.
                pending = (task.result or {}).get("needs_approval") or {}
                op = pending.get("operation")
                task.status = (
                    TaskStatus.waiting_for_user
                    if op == "waiting_for_user"
                    else TaskStatus.awaiting_approval
                )
            log.info("task.completed", confidence=(task.result or {}).get("confidence"))
        except asyncio.CancelledError:
            # User-requested stop (handle.cancel()). Record it as a terminal
            # state and let the cancellation propagate so the asyncio task ends.
            task.status = TaskStatus.cancelled
            task.error = "cancelled by user"
            # Progress checkpoints are saved as steps finish, so a stopped run can
            # continue from the last completed step instead of starting over.
            task.resumable = _resume_checkpoint(task.result) is not None
            log.info("task.cancelled", resumable=task.resumable)
            raise
        except Exception as exc:  # noqa: BLE001 - surface failure to the UI
            task.status = TaskStatus.failed
            task.error = f"{type(exc).__name__}: {exc}"
            task.resumable = _resume_checkpoint(task.result) is not None
            log.warning("task.failed", error=task.error, resumable=task.resumable)
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

    def delete_run(self, session_id: str, task_id: str) -> bool:
        """Remove one follow-up run from a session. The initial task cannot be deleted."""
        session = self._sessions.get(session_id)
        if session is None or task_id not in session.task_ids:
            return False
        if session.task_ids[0] == task_id:
            return False
        self.cancel(task_id)
        session.task_ids = [tid for tid in session.task_ids if tid != task_id]
        self._tasks.pop(task_id, None)
        self._handles.pop(task_id, None)
        if self._store is not None:
            with contextlib.suppress(Exception):
                self._store.delete_task(session_id, task_id)
        self._persist_session(session)
        return True

    def delete_all_sessions(self) -> list[str]:
        """Remove every saved session/task from memory and disk."""
        ids = list(self._sessions.keys())
        for sid in ids:
            self.delete_session(sid)
        return ids

    # ── Queries ──────────────────────────────────────────────────────────────
    def get_task(self, task_id: str) -> TaskRecord | None:
        task = self._tasks.get(task_id)
        if task is not None:
            self._reconcile_liveness(task)
        return task

    def _reconcile_liveness(self, task: TaskRecord) -> None:
        """Flip a 'running' task to stopped when its worker is no longer alive.

        A run whose asyncio task died without reaching the except handlers (the
        event loop went away, the worker was killed) would otherwise sit at
        'running' forever and the UI would poll it indefinitely. The UI polls this
        every second, so a dead task is surfaced within a second or two.
        """
        if task.status not in {TaskStatus.running, TaskStatus.pending}:
            return
        handle = self._handles.get(task.id)
        if handle is not None and not handle.done():
            return  # genuinely still running
        if handle is None and task.status == TaskStatus.pending:
            return  # queued by startup, not yet launched
        task.status = TaskStatus.failed
        task.error = task.error or "interrupted (worker stopped)"
        task.resumable = _resume_checkpoint(task.result) is not None
        task.finished_at = task.finished_at or _now()
        log.warning("task.stopped_unexpectedly", task_id=task.id, resumable=task.resumable)
        self._persist_task(task)

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
                            "resumable": r.resumable,
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

    def _build_session_context(self, task: TaskRecord) -> str:
        """Context for a run: local asset inventory + earlier runs in this session.

        The inventory is workspace-wide and deliberately *not* session-scoped — a
        brand-new task must still know what has already been downloaded so it
        reuses those files instead of spending bandwidth re-fetching them.
        """
        from ai_employee.observability.asset_registry import (
            AssetRegistry,
            build_local_inventory_context,
        )
        from ai_employee.observability.session_context import build_prior_session_context

        parts: list[str] = []
        try:
            from ai_employee.config import get_settings

            inventory = build_local_inventory_context(AssetRegistry(get_settings().workspace_dir))
            if inventory:
                parts.append(inventory)
        except Exception as exc:  # noqa: BLE001 - inventory must never break a run
            log.warning("asset_inventory_failed", error=str(exc))

        session = self._sessions.get(task.session_id)
        if session:
            prior = [
                self._tasks[tid]
                for tid in session.task_ids
                if tid != task.id and tid in self._tasks
            ]
            prior_ctx = build_prior_session_context(prior)
            if prior_ctx:
                parts.append(prior_ctx)
        return "\n\n".join(parts)

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


def _pending_approval(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    pending = result.get("needs_approval")
    return pending if isinstance(pending, dict) else None


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True if ``fn`` accepts keyword ``name`` (directly or via **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _resume_checkpoint(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the saved checkpoint when it can safely resume in execute()."""
    if not isinstance(result, dict):
        return None
    checkpoint = result.get("workflow_checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    plan = checkpoint.get("plan") or {}
    steps = plan.get("steps") or []
    if not steps:
        return None
    try:
        cursor = int(checkpoint.get("cursor", 0))
    except (TypeError, ValueError):
        return None
    # Resume currently restarts from execute(); only offer it while there are
    # still planned steps left to run.
    if 0 <= cursor < len(steps):
        return checkpoint
    return None


# Tool name → approval key (for batch "allow all similar" in one click).
_TOOL_APPROVAL_KEYS: dict[str, str] = {
    "download_resource": "resource_download",
    "make_dir": "shell_write",
    "write_file": "shell_write",
    "shell": "shell_write",
    "request_resource_input": "resource_acquisition",
}

# Approvals commonly needed together during resource download workflows.
_RESOURCE_WORKFLOW_OPS = frozenset({"resource_download", "shell_write", "resource_acquisition"})

# Extra keys not always listed in settings but still used as approval gates.
_AUTO_APPROVE_EXTRA = frozenset({"waiting_for_user", "run_research_code", "resource_acquisition"})


def _auto_approve_ops() -> set[str]:
    """Every known approval key — used when Auto Approve is enabled."""
    from ai_employee.config import get_settings

    return (
        set(get_settings().approval_required)
        | set(_AUTO_APPROVE_EXTRA)
        | set(_TOOL_APPROVAL_KEYS.values())
        | set(_RESOURCE_WORKFLOW_OPS)
    )


def _expand_approved_ops(
    op: str | None,
    approval: dict[str, Any],
    *,
    approve_all_similar: bool,
) -> set[str]:
    """Ops to grant for this approval click."""
    ops: set[str] = set()
    if op == "waiting_for_user":
        ops.add("resource_acquisition")
        ops.add("waiting_for_user")
    elif op:
        ops.add(op)

    if not approve_all_similar:
        # mkdir / writes are almost always paired with downloads in one step.
        if op == "resource_download":
            ops.add("shell_write")
        return ops

    batched = approval.get("batched_calls") or []
    for call in batched:
        key = _TOOL_APPROVAL_KEYS.get(str(call.get("tool") or ""))
        if key:
            ops.add(key)

    tool_key = _TOOL_APPROVAL_KEYS.get(str(approval.get("tool") or ""))
    if tool_key:
        ops.add(tool_key)

    if ops & _RESOURCE_WORKFLOW_OPS or any(
        call.get("tool") == "download_resource" for call in batched
    ):
        ops |= set(_RESOURCE_WORKFLOW_OPS)

    return ops


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
