"""The AI Employee's execution graph, built with LangGraph.

Pipeline (mirrors the framework spec):

    plan → execute(step) ↺ → synthesize → reflect → END
                 │
                 └─ verify each step; retry on failure (bounded)

State flows through typed dicts; nodes are async. The graph is provider-agnostic —
all model calls go through the injected :class:`Brain`.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ai_employee.agents import AgentContext, AgentRegistry
from ai_employee.brains.base import Brain
from ai_employee.config import get_settings
from ai_employee.core.logging import bind_context, get_logger
from ai_employee.memory import MemoryManager, MemoryNamespace
from ai_employee.tools.invocation_history import (
    begin_task_tool_history,
    clear_task_tool_history,
)
from ai_employee.tools.registry import ToolRegistry
from ai_employee.verification import Verifier, audit_resources, build_waiting_for_user_payload

log = get_logger("workflow")

# Phrases that signal an agent declined on policy/legal grounds rather than failing
# mechanically. Retrying such a step just reproduces the same refusal, so we don't.
_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "cannot provide",
    "can't provide",
    "cannot share",
    "cannot fulfill",
    "cannot comply",
    "unable to provide",
    "not able to provide",
    "legally restricted",
    "against policy",
    "not permitted",
    "i'm unable",
)


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _looks_unusable(record: dict[str, Any]) -> bool:
    """True if a completed step produced nothing a dependent step could build on."""
    if record.get("skipped"):
        return True
    audit = record.get("resource_audit") or {}
    outcome = audit.get("outcome", "")
    if outcome in ("completed", "waiting_for_user", "partial"):
        return False
    if audit.get("issues") and outcome not in ("waiting_for_user",):
        return True
    if (
        audit.get("applies")
        and audit.get("claims_found")
        and not audit.get("has_urls")
        and not audit.get("has_verified_download")
    ):
        return True
    out = (record.get("output") or "").strip()
    if len(out) < 20:
        return True
    return _looks_like_refusal(out)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:500]


def _keep_last(_old: Any, new: Any) -> Any:
    return new


class EmployeeState(TypedDict, total=False):
    goal: str
    session_id: str
    session_context: str
    plan: dict[str, Any]
    step_results: Annotated[list[dict[str, Any]], lambda old, new: (old or []) + new]
    cursor: Annotated[int, _keep_last]
    final: Annotated[str, _keep_last]
    confidence: Annotated[float, _keep_last]
    needs_approval: Annotated[dict[str, Any] | None, _keep_last]
    resource_decision: Annotated[dict[str, Any] | None, _keep_last]
    approved_ops: Annotated[list[str], _keep_last]
    resume_execute: Annotated[bool, _keep_last]
    trace: Annotated[list[dict[str, Any]], lambda old, new: (old or []) + new]


class Orchestrator:
    """Builds and runs the LangGraph pipeline with injected dependencies."""

    def __init__(
        self,
        *,
        brain: Brain,
        agents: AgentRegistry,
        tools: ToolRegistry,
        memory: MemoryManager,
        max_retries_per_step: int = 1,
    ) -> None:
        self._brain = brain
        self._agents = agents
        self._tools = tools
        self._memory = memory
        self._verifier = Verifier(brain)
        self._max_retries = max_retries_per_step
        self._graph = self._build()

    # ── Graph construction ───────────────────────────────────────────────────
    def _build(self):
        g = StateGraph(EmployeeState)
        g.add_node("plan", self._plan_node)
        g.add_node("execute", self._execute_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("reflect", self._reflect_node)

        g.add_conditional_edges(
            START,
            self._route_start,
            {"plan": "plan", "execute": "execute"},
        )
        g.add_edge("plan", "execute")
        g.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {"continue": "execute", "synthesize": "synthesize", "halt": END},
        )
        g.add_edge("synthesize", "reflect")
        g.add_edge("reflect", END)
        return g.compile()

    def _context(self, state: EmployeeState, extra_scratch: dict | None = None) -> AgentContext:
        from ai_employee.observability.resource_ctx import (
            bind_approved_ops,
            bind_resource_decision,
            bind_resource_session,
        )

        decision = state.get("resource_decision")
        approved = set(state.get("approved_ops") or [])
        bind_resource_decision(decision)
        bind_approved_ops(approved)
        bind_resource_session(state.get("session_id"))
        ctx = AgentContext(
            brain=self._brain,
            tools=self._tools,
            memory=self._memory,
            session_id=state.get("session_id", "default"),
            approved_ops=approved,
            session_context=state.get("session_context") or "",
        )
        ctx.scratch["agent_catalog"] = self._agents.catalog()
        if extra_scratch:
            ctx.scratch.update(extra_scratch)
        return ctx

    # ── Nodes ────────────────────────────────────────────────────────────────
    @staticmethod
    def _goal_with_session_context(state: EmployeeState) -> str:
        goal = state.get("goal") or ""
        ctx = (state.get("session_context") or "").strip()
        if not ctx:
            return goal
        return f"{ctx}\n\nCURRENT GOAL:\n{goal}"

    async def _plan_node(self, state: EmployeeState) -> EmployeeState:
        bind_context(node="plan", session=state.get("session_id"))
        if state.get("resume_execute") and state.get("plan"):
            log.info(
                "workflow.resume_skip_plan",
                cursor=state.get("cursor", 0),
                steps=len(state["plan"].get("steps", [])),
            )
            return {"trace": [{"node": "plan", "resumed": True}]}
        ctx = self._context(state)
        result = await self._agents.get("planner").run(self._goal_with_session_context(state), ctx)
        plan = result.metadata.get("plan", {})
        log.info("workflow.planned", steps=len(plan.get("steps", [])))
        return {
            "plan": plan,
            "cursor": 0,
            "step_results": [],
            "trace": [{"node": "plan", "steps": len(plan.get("steps", []))}],
        }

    async def _execute_node(self, state: EmployeeState) -> EmployeeState:
        steps = state["plan"].get("steps", [])
        cursor = state.get("cursor", 0)
        step = steps[cursor]
        bind_context(node="execute", step=step["id"], agent=step["agent"])
        settings = get_settings()
        results = state.get("step_results", [])

        # Respect data dependencies: if a prerequisite step produced nothing usable
        # (e.g. the search found no video), skip this step instead of hallucinating.
        if settings.skip_on_failed_dependency:
            unmet = self._unmet_dependencies(step, results)
            if unmet:
                log.info(
                    "workflow.skip_step", step=step["id"], reason="unmet_dependency", deps=unmet
                )
                skip_record = {
                    "id": step["id"],
                    "description": step["description"],
                    "agent": step["agent"],
                    "output": (
                        f"(skipped — prerequisite step(s) {unmet} produced no usable result, "
                        "so there is nothing for this step to act on.)"
                    ),
                    "confidence": 0.0,
                    "verified": False,
                    "retries": 0,
                    "skipped": True,
                }
                return {
                    "step_results": [skip_record],
                    "cursor": cursor + 1,
                    "trace": [
                        {"node": "execute", "step": step["id"], "skipped": True, "deps": unmet}
                    ],
                }

        agent = self._resolve_agent(step["agent"])
        prior = self._format_prior(results)
        task = step["description"] + (f"\n\nContext from prior steps:\n{prior}" if prior else "")

        ctx = self._context(state, {"step_results": results})
        result = await agent.run(task, ctx)

        # Approval gate: stop the whole run and surface the request.
        if result.needs_approval:
            op = (result.needs_approval or {}).get("operation")
            approved = set(state.get("approved_ops") or [])
            n_ok = sum(
                1
                for t in result.tool_trace
                if t.get("tool") == "download_resource" and t.get("ok")
            )
            # Partial downloads or acquisition already granted — don't re-prompt.
            if op == "waiting_for_user" and (
                n_ok > 0
                or approved & {"resource_acquisition", "waiting_for_user"}
            ):
                log.info(
                    "workflow.skip_reapproval",
                    succeeded=n_ok,
                    op=op,
                    approved=sorted(approved),
                )
                result = result.model_copy(update={"needs_approval": None, "success": True})
            else:
                return {
                    "needs_approval": result.needs_approval,
                    "cursor": cursor,
                    "approved_ops": list(state.get("approved_ops") or []),
                    "trace": [{"node": "execute", "step": step["id"], "halted": "approval"}],
                }

        resource_audit = audit_resources(
            task=step["description"],
            success_criteria=step.get("success_criteria", ""),
            output=result.output,
            tool_trace=result.tool_trace,
            workspace=settings.workspace_dir,
        )

        if resource_audit.get("outcome") == "waiting_for_user":
            approved = set(state.get("approved_ops") or [])
            if approved & {"resource_acquisition", "waiting_for_user"}:
                log.info("workflow.skip_wait_audit", approved=sorted(approved))
            elif not resource_audit.get("download_attempted"):
                log.info(
                    "workflow.skip_wait_no_download_attempt",
                    outcome=resource_audit.get("outcome"),
                )
            else:
                payload = build_waiting_for_user_payload(
                    task=step["description"],
                    resource_audit=resource_audit,
                    tool_trace=result.tool_trace,
                )
                return {
                    "needs_approval": payload,
                    "cursor": cursor,
                    "approved_ops": list(state.get("approved_ops") or []),
                    "trace": [
                        {"node": "execute", "step": step["id"], "halted": "waiting_for_user"}
                    ],
                }

        # Verify and (bounded) retry — but don't loop pointlessly.
        verdict = await self._verifier.verify(
            task=step["description"],
            output=result.output,
            success_criteria=step.get("success_criteria", ""),
            resource_audit=resource_audit,
        )
        retries = 0
        while not verdict.passed and retries < self._max_retries:
            # A policy refusal won't change on retry — accept it with the verdict noted.
            if settings.no_retry_on_refusal and _looks_like_refusal(result.output):
                log.info("workflow.retry_skipped", step=step["id"], reason="policy_refusal")
                break
            retries += 1
            log.info("workflow.retry", step=step["id"], attempt=retries)
            prev_output = result.output
            fix_task = (
                f"{task}\n\nA reviewer found issues: {verdict.issues}. "
                f"Fix them. Suggestion: {verdict.suggestion}"
            )
            result = await agent.run(fix_task, ctx)
            resource_audit = audit_resources(
                task=step["description"],
                success_criteria=step.get("success_criteria", ""),
                output=result.output,
                tool_trace=result.tool_trace,
                workspace=settings.workspace_dir,
            )
            verdict = await self._verifier.verify(
                task=step["description"],
                output=result.output,
                success_criteria=step.get("success_criteria", ""),
                resource_audit=resource_audit,
            )
            # If the retry produced essentially the same answer, stop — it's stalled.
            if _normalize(result.output) == _normalize(prev_output):
                log.info("workflow.retry_stalled", step=step["id"], attempt=retries)
                break

        step_record = {
            "id": step["id"],
            "description": step["description"],
            "agent": step["agent"],
            "output": result.output,
            "confidence": min(result.confidence, verdict.confidence),
            "verified": verdict.passed,
            "retries": retries,
            "resource_audit": resource_audit if resource_audit.get("applies") else None,
        }
        return {
            "step_results": [step_record],
            "cursor": cursor + 1,
            "trace": [
                {
                    "node": "execute",
                    "step": step["id"],
                    "agent": step["agent"],
                    "verified": verdict.passed,
                    "tools": result.tool_trace,
                }
            ],
        }

    async def _synthesize_node(self, state: EmployeeState) -> EmployeeState:
        bind_context(node="synthesize")
        log.info("workflow.node", node="synthesize", steps=len(state.get("step_results", [])))
        ctx = self._context(
            state,
            {
                "step_results": state.get("step_results", []),
                "brain_model": self._brain.model_name,
                "brain_provider": self._brain.name,
            },
        )
        result = await self._agents.get("ceo").run(state["goal"], ctx)
        results = state.get("step_results", [])
        confidence = (
            sum(r.get("confidence", 0.5) for r in results) / len(results) if results else 0.5
        )
        return {
            "final": result.output,
            "confidence": round(confidence, 3),
            "trace": [{"node": "synthesize"}],
        }

    async def _reflect_node(self, state: EmployeeState) -> EmployeeState:
        bind_context(node="reflect")
        log.info("workflow.node", node="reflect")
        results = state.get("step_results", [])
        confidence = state.get("confidence", 0.0) or 0.0
        resource_issues = any((r.get("resource_audit") or {}).get("issues") for r in results)
        unverified = any(not r.get("verified", True) for r in results)
        # Persist only trustworthy outcomes — skip hallucinated resource claims.
        from ai_employee.config import get_settings

        if (
            get_settings().use_stored_memory
            and confidence >= 0.5
            and not resource_issues
            and not unverified
        ):
            try:
                await self._memory.record(
                    f"GOAL: {state['goal']}\nRESULT: {state.get('final', '')}",
                    namespace=MemoryNamespace.long_term,
                    session_id=state.get("session_id"),
                    metadata={"confidence": confidence},
                )
                plan_steps = state.get("plan", {}).get("steps", [])
                if len(plan_steps) >= 2:
                    await self._memory.remember_skill(
                        name=state["goal"][:60],
                        description=f"Reusable approach for: {state['goal']}",
                        steps=[s["description"] for s in plan_steps],
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("reflect.memory_failed", error=str(exc))
        else:
            log.info(
                "reflect.memory_skipped",
                confidence=confidence,
                resource_issues=resource_issues,
                unverified=unverified,
                use_stored_memory=get_settings().use_stored_memory,
            )
        return {"trace": [{"node": "reflect"}]}

    # ── Routing ──────────────────────────────────────────────────────────────
    @staticmethod
    def _route_start(state: EmployeeState) -> str:
        if state.get("resume_execute") and state.get("plan"):
            return "execute"
        return "plan"

    def _route_after_execute(self, state: EmployeeState) -> str:
        if state.get("needs_approval"):
            return "halt"
        steps = state["plan"].get("steps", [])
        if state.get("cursor", 0) >= len(steps):
            return "synthesize"
        return "continue"

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _resolve_agent(self, name: str):
        if name in self._agents:
            return self._agents.get(name)
        log.debug("workflow.unknown_agent_fallback", requested=name)
        return self._agents.get("research")

    @staticmethod
    def _format_prior(results: list[dict[str, Any]]) -> str:
        return "\n".join(f"[step {r['id']}] {r['output'][:500]}" for r in results[-3:])

    @staticmethod
    def _unmet_dependencies(step: dict[str, Any], results: list[dict[str, Any]]) -> list[int]:
        """Dependency step ids that already ran but yielded nothing usable."""
        by_id = {r.get("id"): r for r in results}
        unmet: list[int] = []
        for dep in step.get("depends_on", []) or []:
            if dep == step.get("id"):
                continue  # ignore self-references (planners sometimes emit them)
            rec = by_id.get(dep)
            if rec is not None and _looks_unusable(rec):
                unmet.append(dep)
        return unmet

    @staticmethod
    def checkpoint_from_state(
        state: EmployeeState, *, goal: str, session_id: str
    ) -> dict[str, Any]:
        """Serializable snapshot for approve / server-restart resume."""
        return {
            "goal": goal,
            "session_id": session_id,
            "plan": state.get("plan"),
            "cursor": state.get("cursor", 0),
            "step_results": list(state.get("step_results") or []),
            "approved_ops": list(state.get("approved_ops") or []),
        }

    # ── Public API ───────────────────────────────────────────────────────────
    async def run(
        self,
        goal: str,
        *,
        session_id: str = "default",
        approved_ops: set[str] | None = None,
        resource_decision: dict[str, Any] | None = None,
        workflow_checkpoint: dict[str, Any] | None = None,
        session_context: str = "",
    ) -> EmployeeState:
        begin_task_tool_history()
        try:
            merged = set(approved_ops or [])
            if workflow_checkpoint:
                merged |= set(workflow_checkpoint.get("approved_ops") or [])
            ops = sorted(merged)
            if workflow_checkpoint and workflow_checkpoint.get("plan"):
                initial: EmployeeState = {
                    "goal": workflow_checkpoint.get("goal") or goal,
                    "session_id": workflow_checkpoint.get("session_id") or session_id,
                    "plan": workflow_checkpoint["plan"],
                    "cursor": int(workflow_checkpoint.get("cursor", 0)),
                    "step_results": list(workflow_checkpoint.get("step_results") or []),
                    "trace": [],
                    "resume_execute": True,
                    "approved_ops": ops,
                }
                log.info(
                    "workflow.resume",
                    cursor=initial["cursor"],
                    steps=len(initial["plan"].get("steps", [])),
                    approved_ops=ops,
                )
            else:
                initial = {
                    "goal": goal,
                    "session_id": session_id,
                    "session_context": session_context or "",
                    "step_results": [],
                    "cursor": 0,
                    "trace": [],
                }
                if ops:
                    initial["approved_ops"] = ops
            if resource_decision:
                initial["resource_decision"] = resource_decision
            # recursion_limit guards against pathological plans.
            final_state = await self._graph.ainvoke(initial, {"recursion_limit": 50})
            return final_state
        finally:
            clear_task_tool_history()
            from ai_employee.observability.resource_ctx import clear_resource_decision

            clear_resource_decision()
