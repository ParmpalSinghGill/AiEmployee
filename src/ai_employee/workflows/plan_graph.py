"""Plan-mode workflow — research a goal and produce an actionable how-to guide.

Given a high-level goal, this graph:

    research → synthesize → assemble → END

* **research** — the ``research`` agent (download/write tools withheld) searches
  the web and public hubs for approaches, libraries, and products that can
  complete the goal.
* **synthesize** — the Brain produces a typed :class:`PlanGuide`: ordered steps
  ("first do this, then that"), tools to use, and whether each tool is
  open-source or proprietary.
* **assemble** — package a structured result for the dashboard.

No code execution and no downloads — Plan mode is guide-only. Runs offline with
the mock brain (empty tool results still yield a sensible guide).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from ai_employee.agents import AgentContext, AgentRegistry
from ai_employee.agents.specs import NO_ACQUISITION_TOOLS
from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message
from ai_employee.core.logging import bind_context, get_logger
from ai_employee.core.usage import track_usage
from ai_employee.memory import MemoryManager
from ai_employee.tools import ToolRegistry
from ai_employee.tools.invocation_history import (
    begin_task_tool_history,
    clear_task_tool_history,
)

log = get_logger("workflow.plan")

LicenseKind = Literal["open_source", "proprietary", "freemium", "saas", "unknown"]


class ToolOption(BaseModel):
    """A concrete tool/library/product useful for completing the goal."""

    name: str = Field(description="Tool or product name, e.g. 'yt-dlp' or 'Adobe Premiere'.")
    kind: LicenseKind = Field(
        default="unknown",
        description="open_source | proprietary | freemium | saas | unknown",
    )
    url: str = ""
    purpose: str = Field(default="", description="What this tool is used for in the plan.")
    license: str = Field(default="", description="License name if known (MIT, Apache-2.0, …).")
    notes: str = ""


class GuideStep(BaseModel):
    """One ordered step in the how-to guide."""

    id: int = Field(description="1-based step number.")
    title: str = Field(description="Short step title, e.g. 'Install dependencies'.")
    instruction: str = Field(
        description="Clear imperative guidance: what to do, in what order, with what inputs."
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Names of tools from the tools list used in this step.",
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Earlier step ids that must finish first."
    )
    success_criteria: str = Field(default="", description="How to know this step is done.")
    tips: str = Field(default="", description="Optional pitfalls or alternatives.")


class PlanGuide(BaseModel):
    """Full research-backed plan: tools + ordered steps."""

    summary: str = Field(description="1–3 sentence overview of the recommended approach.")
    approach: str = Field(
        default="",
        description="Why this approach (trade-offs vs alternatives).",
    )
    tools: list[ToolOption] = Field(
        default_factory=list,
        description="Tools/libraries/products — mark each open_source or proprietary.",
    )
    steps: list[GuideStep] = Field(
        default_factory=list,
        description="Ordered how-to steps: first do this, then that.",
    )
    alternatives: str = Field(
        default="",
        description="Other viable paths the user could take instead.",
    )
    risks: str = Field(default="", description="Key risks, costs, or blockers.")


class PlanState(TypedDict, total=False):
    goal: str
    session_id: str
    research_notes: str
    guide: PlanGuide
    result: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], lambda old, new: (old or []) + new]


class PlanOrchestrator:
    """Builds and runs the plan-mode research → guide graph."""

    def __init__(
        self,
        *,
        brain: Brain,
        agents: AgentRegistry,
        tools: ToolRegistry,
        memory: MemoryManager,
    ) -> None:
        self._brain = brain
        self._agents = agents
        self._tools = tools
        self._memory = memory
        self._graph = self._build()

    def _build(self):
        g = StateGraph(PlanState)
        g.add_node("research", self._research_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("assemble", self._assemble_node)
        g.set_entry_point("research")
        g.add_edge("research", "synthesize")
        g.add_edge("synthesize", "assemble")
        g.add_edge("assemble", END)
        return g.compile()

    def _context(self, state: PlanState) -> AgentContext:
        ctx = AgentContext(
            brain=self._brain,
            tools=self._tools,
            memory=self._memory,
            session_id=state.get("session_id", "default"),
            approved_ops=set(),
            # Guide-only: research may search and read, never download or write.
            denied_tools=NO_ACQUISITION_TOOLS,
        )
        ctx.scratch["agent_catalog"] = self._agents.catalog()
        return ctx

    async def _research_node(self, state: PlanState) -> PlanState:
        bind_context(node="research", agent="research", session=state.get("session_id"))
        ctx = self._context(state)
        task = (
            f"Research how to accomplish this goal end-to-end. Do NOT download files "
            f"and do NOT write code — only investigate approaches and tools.\n\n"
            f"GOAL:\n{state['goal']}\n\n"
            "Search the web and relevant hubs (GitHub, Hugging Face, docs sites, "
            "product pages). For each promising tool or library, note:\n"
            "- Name and official URL\n"
            "- Whether it is open source, proprietary, freemium, or SaaS\n"
            "- License if known\n"
            "- What part of the goal it covers\n\n"
            "Also outline a practical sequence of work (first … then …). "
            "Prefer open-source options when they are good enough; call out "
            "proprietary tools when they are clearly better or required."
        )
        notes = ""
        try:
            res = await self._agents.get("research").run(task, ctx)
            notes = res.output or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("plan.research_failed", error=str(exc))
            notes = f"(research unavailable: {exc})"
        log.info("plan.researched", chars=len(notes))
        return {
            "research_notes": notes,
            "trace": [{"node": "research", "chars": len(notes)}],
        }

    async def _synthesize_node(self, state: PlanState) -> PlanState:
        bind_context(node="synthesize", agent="brain")
        notes = (state.get("research_notes") or "").strip()
        messages = [
            Message.system(
                "You are a planning specialist. From the research notes (and the goal), "
                "produce a concrete, ordered how-to guide.\n\n"
                "Rules:\n"
                "- Steps must be imperative and sequential (first do X, then Y).\n"
                "- Each step lists which tools it uses (by name).\n"
                "- Every tool must declare kind: open_source, proprietary, freemium, "
                "saas, or unknown. Prefer open_source when viable; include proprietary "
                "alternatives when they matter.\n"
                "- Include success_criteria so the reader knows when a step is done.\n"
                "- Do not invent fake product URLs; leave url empty if unknown.\n"
                "- If research notes are thin, still produce a sensible best-effort plan "
                "from general knowledge, and mark uncertain tools as kind=unknown."
            ),
            Message.user(
                f"GOAL:\n{state['goal']}\n\n"
                f"RESEARCH NOTES:\n{notes or '(none — invent a careful best-effort plan)'}"
            ),
        ]
        try:
            guide = await self._brain.structured_output(messages, PlanGuide)
        except Exception as exc:  # noqa: BLE001
            log.warning("plan.synthesize_failed", error=str(exc))
            guide = self._fallback_guide(state["goal"], notes, error=str(exc))
        guide = self._normalize_guide(guide, state["goal"])
        log.info(
            "plan.synthesized",
            steps=len(guide.steps),
            tools=len(guide.tools),
            oss=sum(1 for t in guide.tools if t.kind == "open_source"),
            proprietary=sum(1 for t in guide.tools if t.kind == "proprietary"),
        )
        return {
            "guide": guide,
            "trace": [
                {
                    "node": "synthesize",
                    "steps": len(guide.steps),
                    "tools": len(guide.tools),
                }
            ],
        }

    @staticmethod
    def _fallback_guide(goal: str, notes: str, *, error: str = "") -> PlanGuide:
        """Best-effort offline plan when the Brain cannot synthesize."""
        snippet = (notes or "").strip()
        tools = [
            ToolOption(
                name="web_search / documentation",
                kind="open_source",
                purpose="Find official docs and tutorials for the stack you choose.",
            ),
            ToolOption(
                name="GitHub",
                kind="open_source",
                url="https://github.com",
                purpose="Discover and evaluate open-source implementations.",
                license="varies",
            ),
            ToolOption(
                name="Commercial SaaS alternative",
                kind="proprietary",
                purpose="Use when you need managed hosting, SLAs, or enterprise support.",
            ),
        ]
        steps = [
            GuideStep(
                id=1,
                title="Clarify scope and success criteria",
                instruction=(
                    f"Write down the exact outcome for: {goal}. "
                    "List inputs, outputs, constraints (budget, license, privacy), "
                    "and how you will know you are done."
                ),
                tools=[],
                success_criteria="Scope and acceptance criteria are written down.",
            ),
            GuideStep(
                id=2,
                title="Choose open-source vs proprietary tools",
                instruction=(
                    "Compare candidate tools. Prefer open-source when quality is enough; "
                    "pick proprietary/SaaS when you need support, compliance, or speed. "
                    "Record license and cost for each choice."
                ),
                tools=[t.name for t in tools],
                depends_on=[1],
                success_criteria="Tool list with kind (open_source/proprietary) is decided.",
            ),
            GuideStep(
                id=3,
                title="Set up environment",
                instruction=(
                    "Install the chosen tools, create a project folder, and verify each "
                    "binary/API key works with a tiny smoke test."
                ),
                tools=[t.name for t in tools[:2]],
                depends_on=[2],
                success_criteria="Smoke test succeeds for every selected tool.",
            ),
            GuideStep(
                id=4,
                title="Execute the core workflow",
                instruction=(
                    "Follow the recommended sequence for the goal: prepare data/inputs, "
                    "run the main processing step, then inspect intermediate outputs."
                ),
                tools=[t.name for t in tools],
                depends_on=[3],
                success_criteria="Core pipeline produces a first usable result.",
            ),
            GuideStep(
                id=5,
                title="Validate, document, and iterate",
                instruction=(
                    "Check results against the success criteria from step 1. Document "
                    "commands, configs, and open-source vs proprietary choices. Fix gaps."
                ),
                tools=[],
                depends_on=[4],
                success_criteria="Result meets acceptance criteria and is reproducible.",
            ),
        ]
        summary = f"Fallback how-to plan for: {goal}." + (
            f" (LLM unavailable: {error})" if error else ""
        )
        approach = (
            "Start with clear success criteria, prefer open-source tooling when viable, "
            "and keep a proprietary alternative ready if quality or support demands it."
        )
        if snippet and not snippet.startswith("(research unavailable"):
            approach += f"\n\nResearch notes excerpt:\n{snippet[:800]}"
        return PlanGuide(
            summary=summary,
            approach=approach,
            tools=tools,
            steps=steps,
            alternatives="Swap any open-source tool for a managed proprietary product if ops cost dominates.",
            risks="Fallback plan is generic — re-run Plan mode with a working LLM for tailored tool picks.",
        )

    @staticmethod
    def _normalize_guide(guide: PlanGuide, goal: str) -> PlanGuide:
        if not guide.steps:
            guide.steps = [
                GuideStep(
                    id=1,
                    title="Start",
                    instruction=goal,
                    tools=[],
                    success_criteria="Initial progress made.",
                )
            ]
        # Stable 1-based ids
        for i, step in enumerate(sorted(guide.steps, key=lambda s: s.id), start=1):
            step.id = i
        guide.steps = sorted(guide.steps, key=lambda s: s.id)
        # Ensure tool kinds are valid
        valid = {"open_source", "proprietary", "freemium", "saas", "unknown"}
        for t in guide.tools:
            if t.kind not in valid:
                t.kind = "unknown"
        return guide

    def _assemble_node(self, state: PlanState) -> PlanState:
        guide = state.get("guide") or PlanGuide(summary="", steps=[], tools=[])
        oss = [t for t in guide.tools if t.kind == "open_source"]
        prop = [t for t in guide.tools if t.kind in {"proprietary", "saas"}]
        other = [t for t in guide.tools if t not in oss and t not in prop]
        result = {
            "goal": state["goal"],
            "summary": guide.summary,
            "approach": guide.approach,
            "tools": [t.model_dump() for t in guide.tools],
            "tools_open_source": [t.model_dump() for t in oss],
            "tools_proprietary": [t.model_dump() for t in prop],
            "tools_other": [t.model_dump() for t in other],
            "steps": [s.model_dump() for s in guide.steps],
            "alternatives": guide.alternatives,
            "risks": guide.risks,
            "research_notes": (state.get("research_notes") or "")[:4000],
        }
        return {"result": result, "trace": [{"node": "assemble", "steps": len(guide.steps)}]}

    async def run(
        self,
        goal: str,
        *,
        session_id: str = "default",
    ) -> dict[str, Any]:
        state: PlanState = {"goal": goal, "session_id": session_id}
        begin_task_tool_history()
        try:
            with track_usage() as meter:
                final = await self._graph.ainvoke(state)
        finally:
            clear_task_tool_history()
        tokens = meter.snapshot()
        log.info("plan.tokens", total=tokens["total_tokens"], calls=tokens["calls"])
        if isinstance(final.get("result"), dict):
            final["result"]["tokens"] = tokens
        final["tokens"] = tokens
        return final


__all__ = [
    "PlanOrchestrator",
    "PlanGuide",
    "GuideStep",
    "ToolOption",
]
