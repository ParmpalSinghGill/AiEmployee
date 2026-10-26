"""Loop detection for the agent ReAct loop — "is it stuck, or just slow?".

When an agent exhausts its iteration budget without producing a final answer we
do **not** stop blindly. We inspect the recent transcript and ask one question:

    Is the agent stuck in a loop, or is it still making progress?

A stuck agent is stopped; a progressing one is granted another iteration window
(up to ``agent_loop_extensions_max`` extra windows), so a long-but-healthy task
can finish instead of being guillotined by a fixed cap.

Design — hybrid (see ``LoopCheckService``):
  * **Heuristics** decide the clear cases. They are free, deterministic and work
    offline (the suite runs on the mock provider), so an obvious loop never
    costs a model call.
  * Only an **ambiguous** verdict escalates to a cheap **LLM judge**. When no
    judge is available (mock/offline, or the call fails) an ambiguous step is
    treated as *progressing* — the safe default, because the extension count is
    itself bounded.

This module is deliberately free of any dependency on ``agents.base`` (which
imports *it*) so it stays unit-testable and could later be swapped for an
out-of-process service behind the same :meth:`LoopCheckService.evaluate` shape.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message, Role
from ai_employee.config.settings import Settings, get_settings
from ai_employee.core.logging import get_logger
from ai_employee.tools.invocation_history import normalize_tool_arguments

log = get_logger("loop_check")

#: Tools that only observe — they bring information but produce no artifact.
_READ_TOOLS = frozenset({"read_file", "list_dir", "csv_preview", "json_query"})
#: Tools that create/modify something — a successful call is real progress.
_WRITE_TOOLS = frozenset({"write_file", "make_dir"})


class LoopJudge(BaseModel):
    """Structured verdict returned by the LLM fallback."""

    is_loop: bool = Field(
        description=(
            "True if the agent is stuck repeating itself or making no progress "
            "toward the task; False if it is still advancing."
        )
    )
    reason: str = Field(default="", description="One short sentence justifying the verdict.")


@dataclass(frozen=True)
class LoopVerdict:
    """The outcome of a loop check."""

    is_loop: bool
    confidence: float
    method: str  # "heuristic" | "llm" | "disabled" | "error"
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def ambiguous(self) -> bool:
        return bool(self.signals.get("ambiguous"))


# ── Heuristics ───────────────────────────────────────────────────────────────


def _signature(entry: dict[str, Any]) -> str:
    """Stable identity of a tool call: tool name + normalized arguments."""
    return f"{entry.get('tool')}:{normalize_tool_arguments(entry.get('args'))}"


def _read_target(entry: dict[str, Any]) -> str:
    args = entry.get("args") or {}
    for key in ("path", "file", "url", "pattern", "query"):
        if isinstance(args, dict) and args.get(key):
            return f"{entry.get('tool')}:{args[key]}"
    return _signature(entry)


def _has_cycle(signatures: list[str], min_repeats: int = 4) -> bool:
    """True if a short A,B(,C) pattern repeats ``min_repeats`` times at the tail.

    ``min_repeats`` defaults to 4 (i.e. the pattern must repeat *more than 3*
    times) so that a normal edit→test→edit→test rhythm — which repeats twice —
    is not mistaken for a loop.
    """
    n = len(signatures)
    for period in (2, 3):
        span = period * min_repeats
        if n < span:
            continue
        tail = signatures[-span:]
        if len(set(tail)) > 1 and all(tail[i] == tail[i % period] for i in range(span)):
            return True
    return False


def _max_duplicate_prose(messages: Sequence[Message], window: int) -> int:
    """Highest repeat count of an identical non-trivial assistant message."""
    texts: list[str] = []
    for m in list(messages)[-(max(4, window) * 2) :]:
        if m.role == Role.assistant and m.content and len(m.content.strip()) >= 20:
            texts.append(re.sub(r"\s+", " ", m.content.strip())[:200])
    return max(Counter(texts).values(), default=0)


def heuristic_check(
    *,
    trace: Sequence[dict[str, Any]],
    messages: Sequence[Message],
    window: int,
    repeat_threshold: int = 4,
    cycle_min_repeats: int = 4,
    repeat_failure_limit: int = 2,
) -> LoopVerdict:
    """Cheap, deterministic first pass.

    Returns a definite LOOP, a definite PROGRESS (an artifact was produced), or
    an ambiguous verdict the caller may escalate to an LLM judge.
    """
    tail = list(trace)[-window:] if window > 0 else list(trace)
    if not tail:
        return LoopVerdict(False, 0.0, "heuristic", "no tool activity to judge", {})

    signatures = [_signature(e) for e in tail]
    max_repeat = max(Counter(signatures).values(), default=0)
    reads = Counter(_read_target(e) for e in tail if e.get("tool") in _READ_TOOLS)
    max_same_read = max(reads.values(), default=0)
    cycle = _has_cycle(signatures, cycle_min_repeats)
    dup_prose = _max_duplicate_prose(messages, window)
    wrote = any(e.get("ok") and e.get("tool") in _WRITE_TOOLS for e in tail)

    fail_classes = Counter(
        (e.get("tool"), str(e.get("error") or "")[:60]) for e in tail if not e.get("ok")
    )
    max_fail = max(fail_classes.values(), default=0)

    signals: dict[str, Any] = {
        "window": len(tail),
        "max_repeat": max_repeat,
        "max_same_read": max_same_read,
        "cycle": cycle,
        "dup_prose": dup_prose,
        "wrote": wrote,
        "max_fail": max_fail,
    }

    hard_loop = (
        max_repeat >= repeat_threshold
        or max_same_read >= repeat_threshold
        or cycle
        or max_fail >= repeat_failure_limit
        or dup_prose >= 2
    )
    if hard_loop:
        reasons = []
        if max_repeat >= repeat_threshold:
            reasons.append(f"a call repeated {max_repeat}×")
        if max_same_read >= repeat_threshold:
            reasons.append(f"the same target was read {max_same_read}×")
        if cycle:
            reasons.append("a short cycle repeated")
        if max_fail >= repeat_failure_limit:
            reasons.append(f"the same failure recurred {max_fail}×")
        if dup_prose >= 2:
            reasons.append(f"the assistant repeated itself {dup_prose}×")
        return LoopVerdict(True, 0.8, "heuristic", "; ".join(reasons), signals)

    if wrote:
        return LoopVerdict(False, 0.75, "heuristic", "produced an artifact", signals)

    # No clear repetition and no artifact: could be a slow-but-legitimate window
    # (e.g. gathering information) or an unproductive read loop. Escalate.
    return LoopVerdict(
        False,
        0.3,
        "heuristic",
        "no clear repetition; needs judgement",
        {**signals, "ambiguous": True},
    )


# ── The service (heuristic → optional LLM judge) ─────────────────────────────


class LoopCheckService:
    """Decide whether an agent that hit its cap is stuck.

    ``brain`` is only used for the ambiguous case; pass ``None`` to force the
    pure-heuristic path (offline).
    """

    def __init__(self, brain: Brain | None = None, settings: Settings | None = None) -> None:
        self._brain = brain
        self._settings = settings

    async def evaluate(
        self,
        *,
        agent: str,
        trace: Sequence[dict[str, Any]],
        messages: Sequence[Message],
        window: int,
        settings: Settings | None = None,
    ) -> LoopVerdict:
        settings = settings or self._settings or get_settings()
        if not settings.agent_loop_check_enabled:
            return LoopVerdict(False, 0.0, "disabled", "loop check disabled", {})

        verdict = heuristic_check(
            trace=trace,
            messages=messages,
            window=window,
            repeat_threshold=settings.agent_loop_repeat_threshold,
            cycle_min_repeats=settings.agent_loop_cycle_min_repeats,
            repeat_failure_limit=settings.agent_repeat_failure_limit,
        )
        if not verdict.ambiguous:
            return verdict

        if not self._judge_available(settings):
            # Offline / no judge: keep the safe default (allow the extension).
            return verdict

        try:
            return await self._llm_judge(
                agent=agent, trace=list(trace)[-window:], settings=settings
            )
        except Exception as exc:  # noqa: BLE001 - never let the judge break the run
            log.warning("loop_check.judge_failed", agent=agent, error=str(exc))
            return verdict

    def _judge_available(self, settings: Settings) -> bool:
        provider = getattr(settings.llm_provider, "value", settings.llm_provider)
        return self._brain is not None and settings.agent_loop_check_llm and str(provider) != "mock"

    async def _llm_judge(
        self, *, agent: str, trace: list[dict[str, Any]], settings: Settings
    ) -> LoopVerdict:
        assert self._brain is not None
        lines = []
        for i, e in enumerate(trace, 1):
            status = "ok" if e.get("ok") else f"FAILED: {str(e.get('error') or '')[:80]}"
            args = e.get("args") or {}
            detail = args.get("command") or args.get("path") or args.get("query") or args
            lines.append(f"{i}. {e.get('tool')}({str(detail)[:140]}) -> {status}")
        transcript = "\n".join(lines) or "(no tool calls)"
        messages = [
            Message.system(
                "You audit an AI agent's tool transcript to decide if it is STUCK in a loop "
                "or still making PROGRESS toward its task. Signs of a loop: repeating the "
                "same or near-identical calls, re-reading the same files, retrying the same "
                "failing action, or many actions with no new information or artifact. Signs "
                "of progress: new information, new artifacts, varied investigation, or "
                "recovering from an error. Answer only with the JSON verdict."
            ),
            Message.user(
                f"Agent role: {agent}\nRecent tool calls (oldest first):\n{transcript}\n\n"
                "Is this agent stuck in a loop?"
            ),
        ]
        judge = await self._brain.structured_output(messages, LoopJudge, temperature=0.0)
        return LoopVerdict(
            is_loop=bool(judge.is_loop),
            confidence=0.6,
            method="llm",
            reason=(judge.reason or "").strip() or "llm verdict",
            signals={"ambiguous_resolved_by": "llm"},
        )


__all__ = ["LoopCheckService", "LoopVerdict", "LoopJudge", "heuristic_check"]
