"""The AIEmployee facade — the single entry point most users touch.

Wires the brain, memory, tools, agents and the LangGraph orchestrator together
from configuration, and exposes a tiny high-level API:

    employee = AIEmployee()
    result = await employee.run("Research X and write a brief")
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ai_employee.agents import build_agent_registry
from ai_employee.brains import get_brain, get_embedder
from ai_employee.config import Settings, get_settings
from ai_employee.core.logging import configure_logging, get_logger
from ai_employee.memory import build_memory_manager
from ai_employee.observability.usage_summary import summarize_usage
from ai_employee.tools import default_registry
from ai_employee.workflows import Orchestrator, PlanOrchestrator, ResearchOrchestrator

log = get_logger("employee")


def _short_goal(goal: str) -> str:
    g = goal.strip().replace("\n", " ")
    return (g[:42] + "…") if len(g) > 42 else (g or "Untitled task")


def _normalize_download_goal(goal: str) -> str:
    g = goal.strip()
    if not g:
        return "Download the requested resource into the workspace."
    if ("http://" in g or "https://" in g) and not re.search(
        r"\b(download|save|clone|pull|fetch|acquire)\b", g, re.I
    ):
        return (
            "Download this resource URL into the workspace and report the saved path. "
            "If it is YouTube, GitHub, Hugging Face, or Kaggle, use the automated "
            "downloader.\n\n"
            f"URL:\n{g}"
        )
    return g


class AIEmployee:
    """High-level orchestration facade."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level, json=self.settings.log_json)

        # Overlay UI-managed API keys onto settings before anything reads them.
        from ai_employee.config.secrets_store import apply_secrets
        from ai_employee.config.ui_preferences import apply_ui_preferences

        apply_secrets(self.settings)
        apply_ui_preferences(self.settings)

        # Persist per-brain token usage to the ledger (idempotent registration).
        from ai_employee.core.usage import register_usage_sink
        from ai_employee.observability.usage_ledger import make_sink

        register_usage_sink(make_sink(self.settings.usage_ledger_file))

        self.brain = get_brain(self.settings)
        self.embedder = get_embedder(self.settings)
        self.tools = default_registry(self.settings)
        self.agents = build_agent_registry()
        self.memory = build_memory_manager(self.embedder, self.settings)
        self._restore_memory()
        from ai_employee.knowledge import KnowledgeBase

        self.knowledge = KnowledgeBase(self.memory, self.brain)
        self.orchestrator = Orchestrator(
            brain=self.brain,
            agents=self.agents,
            tools=self.tools,
            memory=self.memory,
        )
        self.research_orchestrator = ResearchOrchestrator(
            brain=self.brain,
            agents=self.agents,
            tools=self.tools,
            memory=self.memory,
        )
        self.plan_orchestrator = PlanOrchestrator(
            brain=self.brain,
            agents=self.agents,
            tools=self.tools,
            memory=self.memory,
        )
        log.info(
            "employee.ready",
            provider=self.brain.name,
            agents=len(self.agents.names()),
            tools=len(self.tools.names()),
        )
        # Lifecycle: an employee is replaced when the operator switches the brain
        # (provider/model) in the console. A run that started earlier must keep the
        # model it was launched with, so a retired employee is only closed once the
        # last task still using it has finished. See acquire/release/mark_retired.
        self._task_refs = 0
        self._retired = False
        self._closed = False
        self._closing = False

    # ── Run lifecycle (one model per task) ──────────────────────────────────
    def acquire(self) -> None:
        """A background task is now using this employee — keep its brain alive."""
        self._task_refs += 1

    def release(self) -> None:
        """A task finished; close this employee if it was retired and is now idle."""
        self._task_refs = max(0, self._task_refs - 1)
        self._maybe_close()

    def mark_retired(self) -> None:
        """This employee was replaced by a newer one. Close once no task uses it."""
        self._retired = True
        self._maybe_close()

    def _maybe_close(self) -> None:
        if not (self._retired and self._task_refs <= 0) or self._closing:
            return
        self._closing = True
        import asyncio
        import contextlib

        # No running event loop in some unit tests — the client is GC'd instead.
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(self.aclose())

    # ── Primary API ──────────────────────────────────────────────────────────
    async def run(
        self,
        goal: str,
        *,
        session_id: str = "default",
        approved_ops: set[str] | None = None,
        resource_decision: dict[str, Any] | None = None,
        workflow_checkpoint: dict[str, Any] | None = None,
        session_context: str = "",
        base_agent: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Plan and execute a high-level goal end-to-end.

        ``base_agent`` is **opt-in**: pass it to pin every step of the run to one
        agent. By default (``None``) the planner assigns each step to the best
        specialist — research for finding things, coding for code, data_analysis
        for data, testing for tests, and so on.

        ``on_progress`` receives a resume checkpoint after each completed node, so a
        run stopped mid-flight can be continued instead of restarting from scratch.
        """
        from ai_employee.core.usage import track_usage

        with track_usage() as meter:
            state = await self.orchestrator.run(
                goal,
                session_id=session_id,
                approved_ops=approved_ops,
                resource_decision=resource_decision,
                workflow_checkpoint=workflow_checkpoint,
                session_context=session_context,
                base_agent=base_agent,
                on_progress=on_progress,
            )
        checkpoint = self.orchestrator.checkpoint_from_state(
            state, goal=goal, session_id=session_id
        )
        usage = summarize_usage(
            step_results=state.get("step_results", []),
            trace=state.get("trace", []),
        )
        return {
            "goal": goal,
            "base_agent": state.get("base_agent") or base_agent or "",
            "brain_provider": self.brain.name,
            "brain_model": self.brain.model_name,
            "answer": state.get("final", ""),
            "confidence": state.get("confidence"),
            "plan": state.get("plan"),
            "steps": state.get("step_results", []),
            "tokens": meter.snapshot(),
            "needs_approval": state.get("needs_approval"),
            "trace": state.get("trace", []),
            "workflow_checkpoint": checkpoint if state.get("plan") else None,
            "agents_used": usage["agents_used"],
            "tools_used": usage["tools_used"],
        }

    async def research(
        self,
        goal: str,
        *,
        session_id: str = "default",
        approved_ops: set[str] | None = None,
        auto_run: bool = False,
    ) -> dict[str, Any]:
        """Run the research-assistant decision tree for an ML problem.

        Finds models/datasets/papers, generates the right code (eval / inference /
        training / reproduce-from-paper) plus download & run instructions, and
        estimates hardware. Executing the code locally is approval-gated: pass
        ``auto_run=True`` together with ``approved_ops={'run_research_code'}``.
        """
        state = await self.research_orchestrator.run(
            goal,
            session_id=session_id,
            approved_ops=approved_ops,
            auto_run=auto_run,
        )
        return {
            "goal": goal,
            "result": state.get("result", {}),
            "tokens": state.get("tokens"),
            "needs_approval": state.get("needs_approval"),
            "trace": state.get("trace", []),
        }

    async def plan(
        self,
        goal: str,
        *,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """Research the goal and produce a step-by-step how-to guide with tools.

        Does not execute code or download resources. The result lists ordered
        steps and marks each recommended tool as open-source or proprietary.
        """
        state = await self.plan_orchestrator.run(goal, session_id=session_id)
        return {
            "goal": goal,
            "result": state.get("result", {}),
            "tokens": state.get("tokens"),
            "trace": state.get("trace", []),
        }

    async def download(
        self,
        goal: str,
        *,
        session_id: str = "default",
        approved_ops: set[str] | None = None,
        resource_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the direct downloader agent for one or more resource URLs."""
        from ai_employee.agents import AgentContext
        from ai_employee.core.usage import track_usage
        from ai_employee.observability.resource_ctx import (
            bind_approved_ops,
            bind_resource_decision,
            bind_resource_session,
            clear_resource_decision,
        )

        task = _normalize_download_goal(goal)
        ctx = AgentContext(
            brain=self.brain,
            tools=self.tools,
            memory=self.memory,
            session_id=session_id,
            approved_ops=set(approved_ops or ()),
        )
        ctx.scratch["agent_catalog"] = self.agents.catalog()

        with track_usage() as meter:
            bind_approved_ops(approved_ops or set())
            bind_resource_decision(resource_decision)
            bind_resource_session(session_id)
            try:
                result = await self.agents.get("resource_downloader").run(task, ctx)
            finally:
                clear_resource_decision()
        usage = summarize_usage(agent=result.agent, tool_trace=result.tool_trace)
        return {
            "goal": goal,
            "answer": result.output,
            "confidence": result.confidence,
            "steps": [],
            "tokens": meter.snapshot(),
            "needs_approval": result.needs_approval,
            "trace": [],
            "agent": result.agent,
            "agents_used": usage["agents_used"],
            "tools_used": usage["tools_used"],
        }

    async def chat(self, message: str, *, session_id: str = "default") -> str:
        """Lightweight single-turn chat with conversation memory (no full pipeline)."""
        from ai_employee.brains.schemas import Message

        use_memory = self.settings.use_stored_memory
        if use_memory:
            await self.memory.record_message("user", message, session_id=session_id)
            window = await self.memory.conversation_window(session_id, limit=10)
        else:
            window = []
        messages = [Message.system("You are a helpful, capable AI employee.")]
        messages += [
            Message(
                role="assistant" if r.metadata.get("role") == "assistant" else "user",
                content=r.content,
            )
            for r in window
        ]
        messages.append(Message.user(message))
        reply = await self.brain.chat(messages)
        if use_memory:
            await self.memory.record_message("assistant", reply.content, session_id=session_id)
        return reply.content

    async def name_task(self, goal: str) -> str:
        """Generate a short, human-friendly title for a goal via a tiny LLM call.

        Cheap (≈20 tokens) and best-effort: any failure, or the offline mock, falls
        back to a truncated goal so naming never blocks or breaks a task.
        """
        fallback = _short_goal(goal)
        if not self.settings.auto_name_tasks or self.brain.name == "mock":
            return fallback
        prompt = (
            "Summarize the following task as a short, specific title of 3 to 6 words. "
            "Use Title Case. No quotes, no trailing period. Reply with ONLY the title.\n\n"
            f"Task: {goal}"
        )
        try:
            # 256 output tokens: on reasoning-enabled models the chain-of-thought is
            # billed against this same budget, so a small ceiling leaves nothing for
            # the visible title and returns finish_reason="length" with empty text.
            text = await self.brain.complete(prompt, max_tokens=256, temperature=0.3)
        except Exception as exc:  # noqa: BLE001
            log.debug("name_task.failed", error=str(exc))
            return fallback
        import re

        title = (text or "").strip().splitlines()[0] if text.strip() else ""
        title = re.sub(r"^(title|task)\s*[:\-]\s*", "", title, flags=re.I).strip(" \"'.`*")
        return title[:60] or fallback

    async def ask_agent(self, agent_name: str, task: str, *, session_id: str = "default") -> dict:
        """Run a single named agent directly (bypassing planning)."""
        from ai_employee.agents import AgentContext

        ctx = AgentContext(
            brain=self.brain,
            tools=self.tools,
            memory=self.memory,
            session_id=session_id,
        )
        ctx.scratch["agent_catalog"] = self.agents.catalog()
        result = await self.agents.get(agent_name).run(task, ctx)
        usage = summarize_usage(agent=result.agent, tool_trace=result.tool_trace)
        out = result.model_dump()
        out["agents_used"] = usage["agents_used"]
        out["tools_used"] = usage["tools_used"]
        return out

    # ── Persistence (memory carries context across restarts for "resume") ────
    def _store(self):
        from ai_employee.observability.store import SessionStore

        return SessionStore(self.settings.sessions_dir)

    def _restore_memory(self) -> None:
        if not (self.settings.persist_sessions and self.memory.is_persistable):
            return
        try:
            records = self._store().load_memory()
            if records:
                self.memory.import_records(records)
                log.info("memory.restored", records=len(records))
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.restore_failed", error=str(exc))

    def persist_memory(self) -> None:
        """Snapshot in-memory memory to disk so resumed sessions keep context."""
        if not (self.settings.persist_sessions and self.memory.is_persistable):
            return
        try:
            self._store().save_memory(self.memory.export_records())
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.persist_failed", error=str(exc))

    async def clear_memories(self) -> dict[str, Any]:
        """Wipe every memory namespace (long-term, skills, conversation, knowledge, …)."""
        store = self._store()
        records = self.memory.export_records() if self.memory.is_persistable else []
        removed = len(records) if self.memory.is_persistable else None
        if records:
            store.archive_memory(records)
        await self.memory.clear(None)
        self.persist_memory()
        log.info("memory.cleared_all", records=removed, archived=bool(records))
        return {"cleared": True, "records_removed": removed, "backup_saved": bool(records)}

    def memory_status(self) -> dict[str, Any]:
        """Current memory toggle state and whether an archived backup exists."""
        store = self._store()
        backup = store.load_memory_backup() if self.memory.is_persistable else []
        current = self.memory.export_records() if self.memory.is_persistable else []
        return {
            "use_stored_memory": self.settings.use_stored_memory,
            "backup_available": bool(backup),
            "backup_records": len(backup),
            "current_records": len(current),
        }

    async def restore_memories(self) -> dict[str, Any]:
        """Reload memory from the archive written before the last clear."""
        if not self.memory.is_persistable:
            return {"restored": False, "records_restored": 0}
        store = self._store()
        records = store.load_memory_backup()
        if not records:
            return {"restored": False, "records_restored": 0}
        await self.memory.clear(None)
        self.memory.import_records(records)
        self.persist_memory()
        log.info("memory.restored_from_backup", records=len(records))
        return {"restored": True, "records_restored": len(records)}

    async def clear_session_memory(self, session_id: str) -> dict[str, Any]:
        """Drop recall context for one saved task/session (history on disk is kept)."""
        removed = await self.memory.clear_session(session_id)
        self.persist_memory()
        log.info("memory.cleared_session", session_id=session_id, records=removed)
        return {"cleared": True, "session_id": session_id, "records_removed": removed}

    # ── Resources: task <-> source links, isolation, reference-counted delete ──
    def resource_manager(self):
        """The workspace resource catalog (task <-> source links, bytes)."""
        from ai_employee.resources import manager_for

        return manager_for(self.settings)

    def delete_task_resources(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        purge: bool = True,
    ) -> dict[str, Any]:
        """Reference-counted resource cleanup for a deleted task/run.

        A source is kept while any *other* task still links it, and deleted (with
        its bytes) only when the removed task was its **last** link.
        """
        mgr = self.resource_manager()
        report: dict[str, Any] = {"unlinked": [], "deleted": [], "kept": [], "bytes_freed": 0}
        if session_id:
            report = mgr.delete_for_session(session_id, purge=purge)
        if task_id:
            extra = mgr.delete_for_task(task_id, purge=purge)
            for key in ("unlinked", "deleted", "kept"):
                report[key] = sorted(set(report.get(key, [])) | set(extra.get(key, [])))
            report["bytes_freed"] = report.get("bytes_freed", 0) + extra.get("bytes_freed", 0)
        return report

    async def isolate_resources(
        self,
        *,
        all: bool = False,  # noqa: A002 - mirrors the API request field
        session_id: str | None = None,
        task_id: str | None = None,
        allow_redownload: bool = True,
    ) -> dict[str, Any]:
        """Isolate tasks so no source stays shared (re-downloading if required)."""
        from ai_employee.resources.isolation import isolate_all, isolate_task

        mgr = self.resource_manager()
        if all:
            return await isolate_all(mgr, settings=self.settings, allow_redownload=allow_redownload)
        ref = (task_id or session_id or "").strip()
        return await isolate_task(
            mgr, ref, settings=self.settings, allow_redownload=allow_redownload
        )

    def purge_resources(
        self, *, purge_bytes: bool = True, wipe_workspace: bool = False
    ) -> dict[str, Any]:
        """Delete every resource (catalog + bytes) for a fresh start.

        With ``wipe_workspace`` also delete every other file in the workspace
        (downloads and outputs that were never catalogued).
        """
        mgr = self.resource_manager()
        report = mgr.purge_all(purge_bytes=purge_bytes)
        if wipe_workspace:
            report["workspace"] = mgr.wipe_workspace()
        return report

    def resources_overview(self) -> dict[str, Any]:
        """Every resource with its task links — for the console's Resources panel."""
        mgr = self.resource_manager()
        items: list[dict[str, Any]] = []
        for rec in sorted(mgr.records().values(), key=lambda r: r.name):
            tasks = mgr.tasks_for(rec.id)
            items.append(
                {
                    "id": rec.id,
                    "name": rec.name,
                    "kind": rec.kind,
                    "visibility": rec.visibility,
                    "owner_task": rec.owner_task,
                    "size": rec.size,
                    "path": rec.path,
                    "source": rec.source,
                    "url": rec.url,
                    "exists": mgr.exists(rec),
                    "tasks": tasks,
                    "task_count": len(tasks),
                }
            )
        return {
            "resources": items,
            "count": len(items),
            "shared": sum(1 for r in items if r["visibility"] == "shared"),
            "private": sum(1 for r in items if r["visibility"] == "private"),
        }

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.persist_memory()
        await self.brain.aclose()
        if self.embedder is not self.brain:
            await self.embedder.aclose()
