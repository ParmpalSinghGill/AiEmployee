"""Base agent abstractions and the shared tool-using loop."""

from __future__ import annotations

import abc
import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.agents.loop_check import LoopCheckService, LoopVerdict
from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message, Role
from ai_employee.config import get_settings
from ai_employee.core.exceptions import ApprovalRequired
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip, clip_obj, tool_clip
from ai_employee.memory import MemoryManager, MemoryNamespace
from ai_employee.tools import ToolRegistry
from ai_employee.tools.invocation_history import bind_dedup_scope, reset_dedup_scope

_SCOPE_SEQ = itertools.count()


def _scope_counter() -> int:
    """Unique per agent turn, so a retry of the same step starts with a clean slate."""
    return next(_SCOPE_SEQ)


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
        # Scope duplicate detection to this agent's turn. Task-wide dedup blocked an
        # agent from reading something a *different* agent had already read, which
        # tripped the repeated-duplicate abort and killed the step before it worked.
        token = bind_dedup_scope(f"{self.name}:{id(ctx)}:{_scope_counter()}")
        try:
            return await self._run_inner(task, ctx)
        finally:
            reset_dedup_scope(token)

    async def _run_inner(self, task: str, ctx: AgentContext) -> AgentResult:
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

        # The agent's total rounds = a base window plus any extra windows the
        # loop check grants when it decides we are progressing, not stuck. This
        # keeps a small fixed cap for the common case while letting a long but
        # healthy task finish instead of being cut off mid-work. Both numbers are
        # configurable per agent from .env (AGENT_MAX_ITERATIONS* /
        # AGENT_LOOP_EXTENSIONS*), falling back to the agent's built-in default.
        max_iterations = settings.iteration_limit_for(self.name, self.max_iterations)
        max_extensions = (
            settings.loop_limit_for(self.name) if settings.agent_loop_check_enabled else 0
        )
        extensions_used = 0
        loop_checks: list[dict[str, Any]] = []

        while True:
            for _ in range(max_iterations):
                steps_done += 1
                # Keep context (and per-call latency) from compounding as the loop runs.
                messages = self._compact_history(messages, settings.agent_history_window)
                result = await ctx.brain.chat(
                    messages,
                    tools=tool_specs or None,
                    temperature=0.2,
                )
                if not result.has_tool_calls:
                    output = (result.content or "").strip()
                    # finish_reason="length" means the model was cut off at the token
                    # ceiling — the answer ends mid-word. Length alone can't detect it
                    # (a 124-char stub looks "long enough"), so trust the provider's
                    # signal and ask for a complete answer instead of shipping a stub.
                    truncated = str(result.finish_reason or "").lower() == "length"
                    if len(output) < 20 or truncated:
                        if truncated:
                            log.warning(
                                "agent.output_truncated",
                                agent=self.name,
                                chars=len(output),
                                note="hit llm_max_tokens; retrying for a complete answer",
                            )
                        output = await self._final_answer(
                            messages,
                            trace,
                            ctx,
                            stop_reason=(
                                "truncated_at_token_limit" if truncated else "empty_or_truncated_answer"
                            ),
                            previous=output if truncated else "",
                        )
                    await self._remember(task, output, ctx)
                    log.info(
                        "agent.answer",
                        agent=self.name,
                        output=clip(output),
                        steps=steps_done,
                    )
                    return AgentResult(
                        agent=self.name,
                        success=True,
                        output=output,
                        confidence=self._estimate_confidence(output, trace),
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
                                "output": tool_clip(tool_result.output),
                                "error": tool_result.error,
                                "observation": tool_clip(observation),
                            }
                        )
                        if not tool_result.ok:
                            ecls = self._error_class(tool_result.error)
                            sig = f"{call.name}:{ecls}"
                            fail_counts[sig] = fail_counts.get(sig, 0) + 1
                            err_text = tool_result.error or ""
                            if ecls == "UnicodeDecodeError":
                                observation += (
                                    "\n\n[SYSTEM] UnicodeDecodeError: this file is not UTF-8 "
                                    "(byte 0xA0 is often a cp1252/latin-1 non-breaking space). "
                                    "Retry with pd.read_csv(path, encoding='latin-1') or "
                                    "encoding='cp1252'. csv_preview reports the encoding to use. "
                                    "Do not import chardet."
                                )
                            elif ecls == "InvalidToolArgs":
                                observation += (
                                    "\n\n[SYSTEM] Tool call JSON was truncated or invalid. "
                                    "Retry python_exec with a SHORTER script (under 40 lines). "
                                    "Do not wrap arguments in _raw."
                                )
                            elif ecls == "ModuleNotFoundError" and "chardet" in err_text:
                                observation += (
                                    "\n\n[SYSTEM] chardet is not installed. Skip detection — "
                                    "use encoding='latin-1' instead of importing chardet."
                                )
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
                                elif ecls in {
                                    "UnicodeDecodeError",
                                    "InvalidToolArgs",
                                    "ValidationError",
                                }:
                                    # Recoverable: encoding mismatch or truncated tool JSON.
                                    pass
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
                                    "output": tool_clip(tool_result.output),
                                    "error": tool_result.error,
                                    "observation": tool_clip(observation),
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

            if abort:
                break

            # Iteration cap reached with no natural answer. Ask the loop check
            # whether we are stuck (stop) or merely long (grant another window).
            if extensions_used >= max_extensions:
                stop_reason = stop_reason or "max_iterations reached"
                break
            verdict = await self._check_loop(ctx, messages, trace, base_iterations=max_iterations)
            loop_checks.append(
                {
                    "is_loop": verdict.is_loop,
                    "confidence": verdict.confidence,
                    "method": verdict.method,
                    "reason": verdict.reason,
                    "signals": verdict.signals,
                }
            )
            log.info(
                "agent.loop_check",
                agent=self.name,
                is_loop=verdict.is_loop,
                confidence=verdict.confidence,
                method=verdict.method,
                reason=verdict.reason,
            )
            if verdict.is_loop:
                stop_reason = f"stopped by loop check ({verdict.method}: {verdict.reason})"
                log.info("agent.loop_stopped", agent=self.name, reason=verdict.reason)
                break
            extensions_used += 1
            log.info(
                "agent.loop_extended",
                agent=self.name,
                extension=extensions_used,
                extra_iterations=max_iterations,
                reason=verdict.reason,
            )

        # Loop ended (iterations exhausted, or aborted on a repeated failure).
        stop_reason = stop_reason or "max_iterations reached"
        output = await self._final_answer(messages, trace, ctx, stop_reason)
        log.info(
            "agent.answer",
            agent=self.name,
            output=clip(output),
            steps=steps_done,
            note=stop_reason,
        )
        return AgentResult(
            agent=self.name,
            success=True,
            output=output,
            confidence=0.35 if "failure" in stop_reason else 0.4,
            steps_taken=steps_done,
            tool_trace=trace,
            metadata={"note": stop_reason, "loop_checks": loop_checks},
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    async def _final_answer(
        self,
        messages: list[Message],
        trace: list[dict[str, Any]],
        ctx: AgentContext,
        stop_reason: str | None,
        previous: str = "",
    ) -> str:
        """Force a non-empty step output so dependents are not skipped.

        ``previous`` carries a partial answer that was cut off at the token limit;
        the retry is told to be brief so the whole thing fits this time.
        """
        settings = get_settings()
        history = self._compact_history(messages, settings.agent_history_window)
        if self.name in _RESOURCE_DOWNLOAD_AGENTS and not self._downloads_attempted(trace):
            prompt = (
                "Stop using search/fetch tools. You MUST call download_resource on every "
                "YouTube and Hugging Face URL you found (yt-dlp and huggingface_hub are "
                "automated). Try YouTube URLs first. Only after download_resource succeeds "
                "or fails for each URL, give your final answer with real local paths or "
                "call request_resource_input with the actual download errors."
            )
        elif previous:
            prompt = (
                "Your previous answer was cut off at the output limit mid-sentence. "
                "Write the COMPLETE answer again, but shorter so it fits — under 250 "
                "words. Lead with results, file paths and concrete numbers; drop "
                "preamble and reasoning. It must end with a finished sentence.\n\n"
                f"The truncated attempt started:\n{previous[:500]}"
            )
        else:
            prompt = (
                "Stop using tools now. Write a concise final answer (under 400 words). "
                "Lead with results and file paths. Clearly state any blocker. "
                "Do not narrate your reasoning."
            )
        first = await ctx.brain.chat([*history, Message.user(prompt)], temperature=0.2)
        text = (first.content or "").strip()
        if len(text) >= 20:
            return text
        retry = await ctx.brain.chat(
            [
                *history,
                Message.user(prompt),
                Message.assistant(first.content or ""),
                Message.user(
                    "That reply was empty or truncated. Output the final answer only, "
                    "under 300 words. No preamble."
                ),
            ],
            temperature=0.2,
        )
        text = (retry.content or "").strip()
        if len(text) >= 20:
            return text
        if trace:
            return self._fallback_from_trace(trace, stop_reason or "empty_model_answer")
        return text or "(No answer was produced.)"

    @staticmethod
    def _fallback_from_trace(trace: list[dict[str, Any]], stop_reason: str) -> str:
        lines = [
            f"(Agent stopped without a written answer: {stop_reason}.)",
            "Tool evidence for downstream steps:",
        ]
        for entry in trace[-8:]:
            status = "ok" if entry.get("ok") else "failed"
            payload = entry.get("output") if entry.get("ok") else entry.get("error")
            snippet = re.sub(r"\s+", " ", str(payload or ""))[:700]
            lines.append(f"- {entry.get('tool')} [{status}]: {snippet}")
        return "\n".join(lines)

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
        text = str(error).strip()
        if (
            text.startswith("InvalidToolArgs")
            or "_raw" in text
            or "Field required" in text
            or "validation error for Args" in text
        ):
            return "InvalidToolArgs"
        for line in text.splitlines():
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning))\b", line.strip())
            if m:
                return m.group(1)
        line = text.splitlines()[-1].strip()
        return re.sub(r"\d+|'[^']*'|\"[^\"]*\"", "", line)[:60].strip() or "error"

    async def _check_loop(
        self,
        ctx: AgentContext,
        messages: list[Message],
        trace: list[dict[str, Any]],
        *,
        base_iterations: int | None = None,
    ) -> LoopVerdict:
        """Decide whether hitting the iteration cap means "stuck" or just "long".

        Uses the agent's own budget as the inspection window unless a global
        override is set. Never raises: a failure to judge defaults to "keep
        going", which is safe because the extension count is bounded.
        """
        settings = get_settings()
        window = settings.agent_loop_check_window or base_iterations or self.max_iterations
        service = LoopCheckService(ctx.brain, settings)
        try:
            return await service.evaluate(
                agent=self.name,
                trace=trace,
                messages=messages,
                window=window,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001 - the check must not break the run
            log.warning("agent.loop_check_error", agent=self.name, error=str(exc))
            return LoopVerdict(False, 0.0, "error", f"loop check failed: {exc}", {})

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
        roster_msg = self._roster_message(ctx)
        if roster_msg:
            messages.append(roster_msg)
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
                    "Relevant prior knowledge & learned skills:\n" + "\n---\n".join(context_blobs)
                )
            )
        messages.append(Message.user(task))
        # The agent's *input*: system prompt + recalled context + the task it was given.
        log.info(
            "agent.context",
            agent=self.name,
            system=clip(self.system_prompt),
            memories=[clip(b) for b in context_blobs],
            task=clip(task),
        )
        return messages

    def _roster_message(self, ctx: AgentContext) -> Message | None:
        """Tell the agent who its teammates are so it can stay in its lane.

        The planner already assigns steps to specific agents, but a step's
        description can drift into a neighboring specialty (e.g. a "coding" step
        that also asks for a chart). Without knowing other agents exist, an agent
        just does whatever the task text asks with whatever tools it has, instead
        of doing its own part and flagging the rest for the right specialist.
        """
        catalog = ctx.scratch.get("agent_catalog") or []
        peers = [a for a in catalog if a.get("name") != self.name]
        if not peers:
            return None
        roster = "\n".join(f"- {a['name']}: {a['role']} — {a['description']}" for a in peers)
        return Message.system(
            f"You are the '{self.name}' ({self.role}) on a multi-agent team. Do ONLY the "
            "part of the task that matches your role above — do not take over work that "
            "belongs to a teammate just because you happen to have a tool for it.\n"
            "If the task (or something you discover mid-task) needs a teammate's "
            "specialty, do your own part and clearly say in your final answer what "
            "still needs to be done and by which agent (name it) — do not silently do "
            "it yourself or silently skip it.\n\n"
            f"TEAMMATES:\n{roster}"
        )

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
