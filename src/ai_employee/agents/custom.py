"""Agents with bespoke logic beyond the generic tool-using loop."""

from __future__ import annotations

from ai_employee.agents.base import AgentContext, AgentResult, BaseAgent, ToolUsingAgent
from ai_employee.agents.specs import AgentSpec
from ai_employee.brains.schemas import Message
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip
from ai_employee.verification.resources import format_resource_block, is_resource_task

log = get_logger("agent.custom")


class CodingAgent(ToolUsingAgent):
    """Software-engineer agent with a switchable backend.

    By default (``coding_backend == "brain"``) it behaves like any tool-using agent,
    writing and running code via the configured LLM. When the backend is ``claude``,
    ``codex``, or ``cursor`` it delegates the entire task to that external CLI,
    which writes and runs the code itself — letting a dedicated coding agent do the
    heavy lifting.
    """

    def __init__(self, spec: AgentSpec) -> None:
        self.name = spec.name
        self.role = spec.role
        self.description = spec.description
        self.system_prompt = spec.system_prompt
        self.category = spec.category
        self.allowed_tools = list(spec.tools) if spec.tools is not None else None
        self.max_iterations = spec.max_iterations

    async def run(self, task: str, ctx: AgentContext) -> AgentResult:
        from ai_employee.config import get_settings

        backend = get_settings().coding_backend
        if backend in ("claude", "codex", "cursor"):
            return await self._run_external(task, ctx, backend)
        return await super().run(task, ctx)

    async def _run_external(self, task: str, ctx: AgentContext, backend: str) -> AgentResult:
        from ai_employee.integrations.code_agents import run_external_code_agent

        log.info("agent.start", agent=self.name, session=ctx.session_id)
        log.info(
            "agent.context",
            agent=self.name,
            system=f"Delegated to external coding agent: {backend} CLI (writes & runs code).",
            memories=[],
            task=clip(task, 1200),
        )
        res = await run_external_code_agent(backend, task)
        log.info("agent.answer", agent=self.name, output=clip(res["output"], 4000), note=f"backend={backend}")
        return AgentResult(
            agent=self.name,
            success=res["ok"],
            output=res["output"],
            confidence=0.8 if res["ok"] else 0.3,
            steps_taken=1,
            tool_trace=[
                {
                    "tool": f"{backend}_cli",
                    "args": {"task": clip(task, 300)},
                    "ok": res["ok"],
                    "output": clip(res["output"], 2000),
                    "error": res.get("error"),
                }
            ],
            metadata={"backend": backend, "returncode": res.get("returncode")},
        )


class PlannerAgent(BaseAgent):
    """Decomposes a goal into a plan. Output is the JSON of a :class:`Plan`."""

    name = "planner"
    role = "Planner"
    description = "Breaks a high-level goal into ordered, agent-assigned steps."

    async def run(self, task: str, ctx: AgentContext) -> AgentResult:
        from ai_employee.planning.planner import Planner

        catalog = ctx.scratch.get("agent_catalog", [])
        planner = Planner(ctx.brain)
        plan = await planner.make_plan(task, agent_catalog=catalog)
        return AgentResult(
            agent=self.name,
            success=True,
            output=plan.model_dump_json(indent=2),
            confidence=0.8,
            metadata={"plan": plan.model_dump()},
        )


class CEOAgent(BaseAgent):
    """The 'manager': frames the goal, and at the end synthesizes a final answer
    from the per-step results produced by the workforce."""

    name = "ceo"
    role = "Chief Executive"
    description = "Frames goals and synthesizes the team's work into a final deliverable."

    async def run(self, task: str, ctx: AgentContext) -> AgentResult:
        step_results = ctx.scratch.get("step_results", [])
        if not step_results:
            # No work to synthesize yet — just restate/clarify the objective.
            answer = await ctx.brain.complete(
                task,
                system=(
                    "You are the CEO of an AI employee team. Restate the objective "
                    "crisply and note the key success criteria."
                ),
            )
            return AgentResult(agent=self.name, output=answer, confidence=0.7)

        digest = "\n\n".join(
            f"### Step {r.get('id')}: {r.get('description')}\n{r.get('output')}"
            for r in step_results
        )
        model_label = ctx.scratch.get("brain_model") or getattr(ctx.brain, "model_name", "")
        provider_label = ctx.scratch.get("brain_provider") or getattr(ctx.brain, "name", "")
        brain_label = f"{provider_label} / {model_label}".strip(" /")

        resource_sections: list[str] = []
        for r in step_results:
            audit = r.get("resource_audit") or {}
            if audit.get("resources"):
                resource_sections.append(
                    format_resource_block(audit["resources"], model=brain_label)
                )
        resource_hint = ""
        if is_resource_task(task) or resource_sections:
            resource_hint = (
                "\n\nRESOURCE REPORTING RULES:\n"
                "- Include a **Resources** section in the final answer.\n"
                "- For EACH resource use exactly:\n"
                "  - **Name:** ...\n"
                "  - **Web URL:** ... (or NOT FOUND)\n"
                "  - **Local path:** ... (or NOT DOWNLOADED — explain why)\n"
                f"  - **Model:** {brain_label or 'N/A'}\n"
                "- NEVER claim a video/file was found, downloaded, or saved unless a team step "
                "verified it with a real URL or an existing workspace file.\n"
                "- If nothing was obtained, state **Status: NOT FOUND** clearly — do not invent paths."
            )
            if resource_sections:
                resource_hint += (
                    "\n\nAUDITED RESOURCES FROM TEAM STEPS:\n" + "\n\n".join(resource_sections)
                )

        system = (
            "You are the CEO synthesizing your team's work into a single, coherent, "
            "high-quality deliverable for the user. Integrate the step outputs, resolve "
            "contradictions, and present a clear final answer. Do not mention internal "
            "step mechanics unless useful. Be honest when the team did NOT actually obtain "
            "a requested resource — never embellish search snippets into confirmed downloads."
            f"{resource_hint}"
        )
        answer = await ctx.brain.chat(
            [Message.system(system), Message.user(f"GOAL: {task}\n\nTEAM OUTPUT:\n{digest}")]
        )
        return AgentResult(
            agent=self.name,
            success=True,
            output=answer.content,
            confidence=0.8,
            metadata={"synthesized_from": len(step_results)},
        )


class HumanApprovalAgent(BaseAgent):
    """Represents the human-in-the-loop gate. In an interactive deployment this
    surfaces a prompt; here it records the request and defers by default."""

    name = "human_approval"
    role = "Human Approval Gate"
    description = "Pauses execution to request explicit human approval for sensitive ops."

    async def run(self, task: str, ctx: AgentContext) -> AgentResult:
        operation = ctx.scratch.get("pending_operation", task)
        log.info("approval.requested", operation=operation)
        return AgentResult(
            agent=self.name,
            success=False,
            output=f"Awaiting human approval for: {operation}",
            confidence=0.0,
            needs_approval={"operation": operation},
        )
