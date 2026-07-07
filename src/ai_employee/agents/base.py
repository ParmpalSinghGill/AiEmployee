"""Base agent abstractions and the shared tool-using loop."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message, Role
from ai_employee.config import get_settings
from ai_employee.core.exceptions import ApprovalRequired
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip, clip_obj
from ai_employee.memory import MemoryManager, MemoryNamespace
from ai_employee.tools import ToolRegistry

log = get_logger("agent")

_PROBE_TOOLS = frozenset({"fetch_page", "http_request"})
_RESOURCE_DOWNLOAD_AGENTS = frozenset({"research", "resource_acquirer", "resource_downloader"})


@dataclass
class AgentContext:
    """Everything an agent needs to act, injected by the orchestrator.

    Dependency injection keeps agents testable: pass a mock brain / registry.
    """

    brain: Brain
    tools: ToolRegistry
    memory: MemoryManager | None = None
    session_id: str = "default"
    task_id: str | None = None
    # Prior runs in this saved session (follow-up context from persisted snapshots).
    session_context: str = ""
    # Operations the human has pre-approved for this run (e.g. {"shell_write"}).
    approved_ops: set[str] = field(default_factory=set)
    scratch: dict[str, Any] = field(default_factory=dict)


class AgentResult(BaseModel):
    agent: str
    success: bool = True
    output: str = ""
    confidence: float = 0.7
    steps_taken: int = 0
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    needs_approval: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent(abc.ABC):
    """Common interface for every agent."""

    name: str = "agent"
    role: str = "generic worker"
    description: str = ""
    system_prompt: str = "You are a capable AI employee. Complete the task precisely."
    #: None => all tools; otherwise a whitelist of tool names.
    allowed_tools: list[str] | None = None
    #: Hard cap on tool-use iterations.
    max_iterations: int = 8

    @abc.abstractmethod
    async def run(self, task: str, ctx: AgentContext) -> AgentResult: ...


class ToolUsingAgent(BaseAgent):
    """A ReAct-style agent: reason → call tools → observe → repeat → answer.

    This single implementation powers most specialized agents; behavior is shaped
    by ``system_prompt`` and ``allowed_tools``.
    """

    async def run(self, task: str, ctx: AgentContext) -> AgentResult:
        log.info("agent.start", agent=self.name, session=ctx.session_id)
        messages = await self._build_initial_messages(task, ctx)
        tool_specs = ctx.tools.specs(self.allowed_tools)
        trace: list[dict[str, Any]] = []
        settings = get_settings()
        # Count tool failures by (tool, error-class) so we can bail out of a
        # doom-loop instead of re-trying the same broken call dozens of times.
        fail_counts: dict[str, int] = {}
        dup_total = 0
        stop_reason: str | None = None
        steps_done = 0

        for step in range(self.max_iterations):
            steps_done = step + 1
            # Keep context (and per-call latency) from compounding as the loop runs.
            messages = self._compact_history(messages, settings.agent_history_window)
            result = await ctx.brain.chat(
                messages,
                tools=tool_specs or None,
                temperature=0.2,
            )
            if not result.has_tool_calls:
                # Final answer.
                await self._remember(task, result.content, ctx)
                log.info(
                    "agent.answer",
                    agent=self.name,
                    output=clip(result.content, 4000),
                    steps=steps_done,
                )
                return AgentResult(
                    agent=self.name,
                    success=True,
                    output=result.content,
                    confidence=self._estimate_confidence(result.content, trace),
                    steps_taken=steps_done,
                    tool_trace=trace,
                    metadata={"usage": result.usage.model_dump()},
                )

            # Record the assistant's tool request, then execute each call.
            messages.append(
                Message(role=Role.assistant, content=result.content, tool_calls=result.tool_calls)
            )
            abort = False
            pending_exc: ApprovalRequired | None = None
            batched_pending: list[dict[str, Any]] = []
            for call in result.tool_calls:
                try:
                    tool_result = await ctx.tools.execute(
                        call.name,
                        call.arguments,
                        approved=self._is_approved(call.name, ctx, call.arguments),
                    )
                    observation = tool_result.as_text()
                    if tool_result.metadata.get("duplicate"):
                        dup_total += 1
                        log.info(
                            "agent.duplicate_call",
                            agent=self.name,
                            tool=call.name,
                            count=dup_total,
                        )
                        if dup_total >= 3:
                            if (
                                self.name in _RESOURCE_DOWNLOAD_AGENTS
                                and not self._downloads_attempted(trace)
                            ):
                                observation += (
                                    "\n\n[SYSTEM] Too many repeated searches. Stop searching "
                                    "and call download_resource on your best YouTube and "
                                    "Hugging Face URLs now (yt-dlp / huggingface_hub)."
                                )
                            else:
                                observation += (
                                    "\n\n[SYSTEM] Too many repeated identical tool calls. "
                                    "Stop searching and synthesize your best final answer from "
                                    "what you already have."
                                )
                                stop_reason = "aborted after repeated duplicate tool calls"
                                abort = True
                    trace.append(
                        {
                            "tool": call.name,
                            "args": call.arguments,
                            "ok": tool_result.ok,
                            "output": clip(tool_result.output, 2000),
                            "error": tool_result.error,
                            "observation": clip(observation, 1000),
                        }
                    )
                    if not tool_result.ok:
                        ecls = self._error_class(tool_result.error)
                        sig = f"{call.name}:{ecls}"
                        fail_counts[sig] = fail_counts.get(sig, 0) + 1
                        if fail_counts[sig] >= settings.agent_repeat_failure_limit:
                            log.info(
                                "agent.repeated_failure",
                                agent=self.name,
                                tool=call.name,
                                error_class=ecls,
                                count=fail_counts[sig],
                            )
                            if (
                                self.name in _RESOURCE_DOWNLOAD_AGENTS
                                and call.name in _PROBE_TOOLS
                                and not self._downloads_attempted(trace)
                            ):
                                observation += (
                                    f"\n\n[SYSTEM] '{call.name}' failed {fail_counts[sig]}× "
                                    "(often 403 on stock sites) but you have NOT called "
                                    "download_resource yet. fetch_page failures do NOT mean "
                                    "YouTube/HF downloads will fail — call download_resource "
                                    "on every YouTube and Hugging Face URL before finishing."
                                )
                            elif (
                                self.name in _RESOURCE_DOWNLOAD_AGENTS
                                and call.name == "download_resource"
                                and not self._downloads_succeeded(trace)
                            ):
                                observation += (
                                    f"\n\n[SYSTEM] This candidate URL failed {fail_counts[sig]}× "
                                    f"with error type ({ecls}). Do NOT stop after the first few "
                                    "failed links. Mark this candidate as failed, skip it, and "
                                    "continue trying the next distinct resource URL before asking "
                                    "the user for manual help."
                                )
                            else:
                                observation += (
                                    f"\n\n[SYSTEM] '{call.name}' has now failed {fail_counts[sig]}× with the same "
                                    f"error type ({ecls}). Stop retrying this approach. If a dependency or resource "
                                    "is missing, say so plainly; otherwise give your best final answer now."
                                )
                                stop_reason = f"aborted after repeated tool failure ({sig})"
                                abort = True
                except ApprovalRequired as exc:
                    if exc.operation in ctx.approved_ops:
                        tool_result = await ctx.tools.execute(
                            call.name,
                            call.arguments,
                            approved=True,
                        )
                        observation = tool_result.as_text()
                        trace.append(
                            {
                                "tool": call.name,
                                "args": call.arguments,
                                "ok": tool_result.ok,
                                "output": clip(tool_result.output, 2000),
                                "error": tool_result.error,
                                "observation": clip(observation, 1000),
                                "preapproved": True,
                            }
                        )
                        messages.append(
                            Message.tool(
                                observation[:8000],
                                tool_call_id=call.id,
                                name=call.name,
                            )
                        )
                        continue
                    if pending_exc is not None and pending_exc.operation != exc.operation:
                        payload = {**pending_exc.payload, "batched_calls": batched_pending}
                        if len(batched_pending) > 1:
                            payload["batch_count"] = len(batched_pending)
                        return AgentResult(
                            agent=self.name,
                            success=False,
                            output=f"Paused: needs approval for {pending_exc.operation}.",
                            confidence=0.0,
                            steps_taken=steps_done,
                            tool_trace=trace,
                            needs_approval={"operation": pending_exc.operation, **payload},
                        )
                    if pending_exc is None:
                        pending_exc = exc
                    for item in self._batched_approval_calls(result.tool_calls, call, ctx):
                        if item not in batched_pending:
                            batched_pending.append(item)
                    log.info(
                        "agent.needs_approval",
                        agent=self.name,
                        op=exc.operation,
                        tool=exc.payload.get("tool"),
                        args=clip_obj(exc.payload.get("args") or {}),
                        batched=len(batched_pending),
                    )
                    messages.append(
                        Message.tool(
                            f"Paused: needs approval for {exc.operation}"
                            + (
                                f" ({len(batched_pending)} operations in this batch)."
                                if len(batched_pending) > 1
                                else "."
                            )
                            + " Approve in the UI to continue.",
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                    continue
                messages.append(
                    Message.tool(observation[:8000], tool_call_id=call.id, name=call.name)
                )
            if pending_exc:
                payload = {**pending_exc.payload, "batched_calls": batched_pending}
                if len(batched_pending) > 1:
                    payload["batch_count"] = len(batched_pending)
                return AgentResult(
                    agent=self.name,
                    success=False,
                    output=f"Paused: needs approval for {pending_exc.operation}.",
                    confidence=0.0,
                    steps_taken=steps_done,
                    tool_trace=trace,
                    needs_approval={"operation": pending_exc.operation, **payload},
                )
            if abort:
                break

        # Loop ended (iterations exhausted, or aborted on a repeated failure).
        stop_reason = stop_reason or "max_iterations reached"
        if self.name in _RESOURCE_DOWNLOAD_AGENTS and not self._downloads_attempted(trace):
            final_prompt = (
                "Stop using search/fetch tools. You MUST call download_resource on every "
                "YouTube and Hugging Face URL you found (yt-dlp and huggingface_hub are "
                "automated). Try YouTube URLs first. Only after download_resource succeeds "
                "or fails for each URL, give your final answer with real local paths or "
                "call request_resource_input with the actual download errors."
            )
        else:
            final_prompt = (
                "Stop using tools now. Give your best final answer, and clearly "
                "state any blocker you could not get past."
            )
        final = await ctx.brain.chat(
            [
                *self._compact_history(messages, settings.agent_history_window),
                Message.user(final_prompt),
            ]
        )
        log.info(
            "agent.answer",
            agent=self.name,
            output=clip(final.content, 4000),
            steps=steps_done,
            note=stop_reason,
        )
        return AgentResult(
            agent=self.name,
            success=True,
            output=final.content,
            confidence=0.35 if "failure" in stop_reason else 0.4,
            steps_taken=steps_done,
            tool_trace=trace,
            metadata={"note": stop_reason},
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _downloads_attempted(trace: list[dict[str, Any]]) -> bool:
        return any(t.get("tool") == "download_resource" for t in trace)

    @staticmethod
    def _downloads_succeeded(trace: list[dict[str, Any]]) -> bool:
        return any(t.get("tool") == "download_resource" and t.get("ok") for t in trace)

    @staticmethod
    def _batched_approval_calls(
        tool_calls: list[Any],
        start_call: Any,
        ctx: AgentContext,
    ) -> list[dict[str, Any]]:
        """All tool calls from ``start_call`` onward that share its approval key."""
        start_idx = next(i for i, c in enumerate(tool_calls) if c.id == start_call.id)
        first_tool = ctx.tools.get(start_call.name)
        if first_tool is None:
            return [{"tool": start_call.name, "args": start_call.arguments}]
        target_key = first_tool.requires_approval(start_call.arguments)
        if not target_key:
            return [{"tool": start_call.name, "args": start_call.arguments}]
        batched: list[dict[str, Any]] = []
        for call in tool_calls[start_idx:]:
            tool = ctx.tools.get(call.name)
            key = tool.requires_approval(call.arguments) if tool else None
            if key == target_key:
                batched.append({"tool": call.name, "args": call.arguments})
        return batched

    @staticmethod
    def _error_class(error: str | None) -> str:
        """Normalize a tool error into a stable 'class' so identical failures group.

        Prefers a Python exception name (``IndentationError``, ``ModuleNotFoundError``);
        otherwise strips volatile bits (numbers, quoted paths) from the last line.
        """
        if not error:
            return "unknown"
        line = str(error).strip().splitlines()[-1].strip()
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning))\b", line)
        if m:
            return m.group(1)
        return re.sub(r"\d+|'[^']*'|\"[^\"]*\"", "", line)[:60].strip() or "error"

    def _compact_history(self, messages: list[Message], window: int) -> list[Message]:
        """Shrink the *content* of older messages once history exceeds ``window``.

        Message structure (roles, tool-call pairing) is preserved verbatim — only
        stale tool outputs / assistant prose are truncated — so the provider
        contract holds while context (and latency) stops compounding.
        """
        if window <= 0 or len(messages) <= window:
            return messages
        keep_tail = max(4, window // 2)
        n = len(messages)
        out: list[Message] = []
        for i, m in enumerate(messages):
            if m.role == Role.system or i >= n - keep_tail:
                out.append(m)
            elif m.role == Role.tool and len(m.content) > 240:
                out.append(
                    m.model_copy(
                        update={"content": m.content[:240] + " …(older tool output trimmed)"}
                    )
                )
            elif m.role == Role.assistant and len(m.content) > 400:
                out.append(m.model_copy(update={"content": m.content[:400] + " …(trimmed)"}))
            else:
                out.append(m)
        return out

    async def _build_initial_messages(self, task: str, ctx: AgentContext) -> list[Message]:
        messages = [Message.system(self.system_prompt)]
        context_blobs: list[str] = []
        if ctx.session_context:
            context_blobs.append(ctx.session_context)
        if ctx.memory is not None and get_settings().use_stored_memory:
            relevant = await ctx.memory.recall(
                task,
                namespace=MemoryNamespace.long_term,
                k=4,
                session_id=ctx.session_id,
            )
            skills = await ctx.memory.find_skill(task, k=2)
            context_blobs.extend(r.content for r in (*relevant, *skills))
        if context_blobs:
            messages.append(
                Message.system(
                    "Relevant prior knowledge & learned skills:\n"
                    + "\n---\n".join(context_blobs)
                )
            )
        messages.append(Message.user(task))
        # The agent's *input*: system prompt + recalled context + the task it was given.
        log.info(
            "agent.context",
            agent=self.name,
            system=clip(self.system_prompt, 800),
            memories=[clip(b, 300) for b in context_blobs],
            task=clip(task, 1200),
        )
        return messages

    def _is_approved(
        self, tool_name: str, ctx: AgentContext, arguments: dict[str, Any] | None = None
    ) -> bool:
        tool = ctx.tools.get(tool_name)
        key = tool.requires_approval(arguments or {})
        if key is None:
            if tool_name == "request_resource_input":
                return bool(ctx.approved_ops & {"resource_acquisition", "waiting_for_user"})
            return True
        return key in ctx.approved_ops

    async def _remember(self, task: str, answer: str, ctx: AgentContext) -> None:
        if ctx.memory is not None and answer and get_settings().use_stored_memory:
            await ctx.memory.record(
                f"Task: {task}\nOutcome: {answer}",
                namespace=MemoryNamespace.task,
                session_id=ctx.session_id,
                metadata={"agent": self.name},
            )

    def _estimate_confidence(self, answer: str, trace: list[dict]) -> float:
        """Cheap heuristic confidence; the verification layer refines it."""
        score = 0.6
        if any(t["ok"] for t in trace):
            score += 0.2
        if any(not t["ok"] for t in trace):
            score -= 0.2
        if len(answer.strip()) < 10:
            score -= 0.2
        hedges = ("i'm not sure", "cannot", "unable", "i don't know")
        if any(h in answer.lower() for h in hedges):
            score -= 0.2
        return max(0.0, min(1.0, score))


class SimpleLLMAgent(ToolUsingAgent):
    """An agent that never uses tools — pure reasoning/generation."""

    allowed_tools: list[str] = []
