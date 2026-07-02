"""Research-assistant workflow.

Given an ML problem, this graph:

    discover → decide → [hardware] → generate → [run?] → assemble → END

* **discover** — the ``scout`` agent searches HF / Papers With Code / Kaggle /
  arXiv / GitHub and we coerce its findings into a typed :class:`Discovery`.
* **decide** — pure Python picks a branch from what was found:
  ``both`` (model+data) · ``model_only`` · ``data_only`` · ``neither``.
* **hardware** — when a model exists, the ``hardware_analyst`` estimates VRAM.
* **generate** — the ``codesmith`` writes the right code (eval / inference /
  training / reproduce-from-paper) **plus** download & run instructions.
* **run** — *approval-gated*. By default we only generate; if the caller approved
  ``run_research_code`` we execute the code in the workspace and capture accuracy,
  otherwise we surface an approval request and stop.
* **assemble** — produce a structured :class:`ResearchResult` for the UI.

Stays provider-agnostic (all model calls via the injected :class:`Brain`) and
runs offline: discovery tools return empty results with no network, the mock
brain synthesizes the structured outputs, and every branch still completes.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from ai_employee.agents import AgentContext, AgentRegistry
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

log = get_logger("workflow.research")

#: Approval key the caller must grant (via ``approved_ops``) to actually execute
#: discovered code, rather than only generating it.
RUN_APPROVAL_KEY = "run_research_code"


# ── Structured findings ──────────────────────────────────────────────────────
class ModelHit(BaseModel):
    id: str = Field(description="Model id, e.g. 'google/vit-base-patch16-224'.")
    url: str = ""
    task: str = ""
    popularity: str = Field("", description="Downloads/likes/stars if known.")
    notes: str = ""


class DatasetHit(BaseModel):
    id: str = Field(description="Dataset id or ref, e.g. 'imagenet-1k'.")
    url: str = ""
    notes: str = ""


class PaperHit(BaseModel):
    title: str
    url: str = ""
    code_url: str = ""
    notes: str = ""


class Discovery(BaseModel):
    """What the scout actually found — only real, tool-returned resources."""

    models: list[ModelHit] = Field(default_factory=list)
    datasets: list[DatasetHit] = Field(default_factory=list)
    papers: list[PaperHit] = Field(default_factory=list)
    reasoning: str = ""


_BRANCH_BRIEF = {
    "both": (
        "A pretrained MODEL and a DATASET were both found. Write an EVALUATION script "
        "that loads the model and the dataset, runs inference over the dataset, and "
        "prints the accuracy (and other relevant metrics). Make it runnable as-is."
    ),
    "model_only": (
        "Only a pretrained MODEL was found (no ready dataset). Write a clean INFERENCE "
        "pipeline: load the model and run predictions on user-provided inputs, with a "
        "small example. Document expected input format."
    ),
    "data_only": (
        "Only a DATASET was found (no ready model). Write a TRAINING script for it "
        "(model definition + training loop + eval split), and clearly state the "
        "hardware needed and roughly how long training may take."
    ),
    "neither": (
        "No ready-made model or dataset was found. Use the located PAPER to reproduce "
        "the method: if pretrained weights exist, write an inference pipeline; "
        "otherwise write a training script. Include precise instructions on WHERE to "
        "download the dataset and HOW to train."
    ),
}


class ResearchState(TypedDict, total=False):
    goal: str
    session_id: str
    approved_ops: set[str]
    auto_run: bool
    discovery: Discovery
    branch: str
    hardware: dict[str, Any]
    generated: dict[str, Any]
    accuracy: dict[str, Any] | None
    needs_approval: dict[str, Any] | None
    result: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], lambda old, new: (old or []) + new]


class ResearchOrchestrator:
    """Builds and runs the research decision-tree graph."""

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
        g = StateGraph(ResearchState)
        g.add_node("discover", self._discover_node)
        g.add_node("decide", self._decide_node)
        g.add_node("hardware", self._hardware_node)
        g.add_node("generate", self._generate_node)
        g.add_node("run", self._run_node)
        g.add_node("assemble", self._assemble_node)

        g.set_entry_point("discover")
        g.add_edge("discover", "decide")
        g.add_conditional_edges(
            "decide",
            lambda s: "hardware" if s["discovery"].models else "generate",
            {"hardware": "hardware", "generate": "generate"},
        )
        g.add_edge("hardware", "generate")
        g.add_conditional_edges(
            "generate",
            self._route_after_generate,
            {"run": "run", "assemble": "assemble"},
        )
        g.add_edge("run", "assemble")
        g.add_edge("assemble", END)
        return g.compile()

    # ── context helper ───────────────────────────────────────────────────────
    def _context(self, state: ResearchState, extra: dict | None = None) -> AgentContext:
        ctx = AgentContext(
            brain=self._brain,
            tools=self._tools,
            memory=self._memory,
            session_id=state.get("session_id", "default"),
            approved_ops=set(state.get("approved_ops") or set()),
        )
        ctx.scratch["agent_catalog"] = self._agents.catalog()
        if extra:
            ctx.scratch.update(extra)
        return ctx

    # ── nodes ────────────────────────────────────────────────────────────────
    async def _discover_node(self, state: ResearchState) -> ResearchState:
        bind_context(node="discover", agent="scout", session=state.get("session_id"))
        ctx = self._context(state)
        task = (
            f"Find pretrained models, datasets and (if needed) the method paper for "
            f"this problem:\n\n{state['goal']}\n\n"
            "Search Hugging Face first, then Papers With Code, Kaggle, arXiv and GitHub "
            "(and Roboflow Universe for computer-vision problems). "
            "Report every real resource you found with its id/url."
        )
        narrative = ""
        try:
            res = await self._agents.get("scout").run(task, ctx)
            narrative = res.output
        except Exception as exc:  # noqa: BLE001
            log.warning("research.scout_failed", error=str(exc))
        discovery = await self._coerce_discovery(state["goal"], narrative)
        log.info(
            "research.discovered",
            models=len(discovery.models),
            datasets=len(discovery.datasets),
            papers=len(discovery.papers),
        )
        return {
            "discovery": discovery,
            "trace": [
                {
                    "node": "discover",
                    "models": len(discovery.models),
                    "datasets": len(discovery.datasets),
                    "papers": len(discovery.papers),
                }
            ],
        }

    async def _coerce_discovery(self, goal: str, narrative: str) -> Discovery:
        """Turn the scout's narrative into typed findings. Never invents resources —
        the prompt forbids it and an empty Discovery is a valid 'found nothing'."""
        if not narrative.strip():
            return Discovery(reasoning="scout produced no output")
        messages = [
            Message.system(
                "Extract ONLY the concrete models, datasets and papers explicitly "
                "named in the scout report below. Do not invent or guess ids/urls. "
                "If none of a category were found, return an empty list for it."
            ),
            Message.user(f"Problem: {goal}\n\nScout report:\n{narrative}"),
        ]
        try:
            return await self._brain.structured_output(messages, Discovery)
        except Exception as exc:  # noqa: BLE001
            log.warning("research.coerce_failed", error=str(exc))
            return Discovery(reasoning=f"could not parse findings: {exc}")

    def _decide_node(self, state: ResearchState) -> ResearchState:
        d = state["discovery"]
        has_model, has_data = bool(d.models), bool(d.datasets)
        if has_model and has_data:
            branch = "both"
        elif has_model:
            branch = "model_only"
        elif has_data:
            branch = "data_only"
        else:
            branch = "neither"
        log.info("research.branch", branch=branch)
        return {"branch": branch, "trace": [{"node": "decide", "branch": branch}]}

    async def _hardware_node(self, state: ResearchState) -> ResearchState:
        bind_context(node="hardware", agent="hardware_analyst")
        top = state["discovery"].models[0]
        ctx = self._context(state)
        task = (
            f"Estimate the GPU/VRAM needed for inference and full fine-tuning of the "
            f"model '{top.id}'. Use estimate_hardware (pass model_id='{top.id}'). "
            "State whether it fits the local GPU and whether to run locally or in cloud."
        )
        hardware: dict[str, Any] = {"model_id": top.id}
        try:
            res = await self._agents.get("hardware_analyst").run(task, ctx)
            hardware["summary"] = res.output
        except Exception as exc:  # noqa: BLE001
            hardware["summary"] = f"(hardware estimate unavailable: {exc})"
        return {"hardware": hardware, "trace": [{"node": "hardware", "model": top.id}]}

    async def _generate_node(self, state: ResearchState) -> ResearchState:
        bind_context(node="generate", agent="codesmith")
        branch = state["branch"]
        d = state["discovery"]
        ctx = self._context(state)
        resources = self._format_resources(d)
        hw = state.get("hardware", {}).get("summary", "")
        task = (
            f"{_BRANCH_BRIEF[branch]}\n\n"
            f"Problem: {state['goal']}\n\n"
            f"Resources found (use ONLY these — do not fabricate ids/urls):\n{resources}\n"
            + (f"\nHardware estimate:\n{hw}\n" if hw else "")
            + "\nWrite the code into the workspace (e.g. research/ ) and produce a "
            "README with: required dependencies, exactly where to download the dataset, "
            "install steps, and the single command to run. End your reply with the "
            "relative path of the main script to run."
        )
        generated: dict[str, Any] = {"branch": branch}
        try:
            res = await self._agents.get("codesmith").run(task, ctx)
            generated["instructions"] = res.output
            generated["files"] = res.metadata.get("written_files", [])
        except Exception as exc:  # noqa: BLE001
            generated["instructions"] = f"(code generation failed: {exc})"
        return {"generated": generated, "trace": [{"node": "generate", "branch": branch}]}

    def _route_after_generate(self, state: ResearchState) -> str:
        """Only run the generated code when the caller asked AND approved it."""
        if not state.get("auto_run"):
            return "assemble"
        if RUN_APPROVAL_KEY in (state.get("approved_ops") or set()):
            return "run"
        return "assemble"  # _run_node not reached; approval surfaced in assemble

    async def _run_node(self, state: ResearchState) -> ResearchState:
        bind_context(node="run", agent="codesmith")
        ctx = self._context(state)
        task = (
            "Run the evaluation/inference script you just generated in the workspace. "
            "Install any missing dependencies with pip_install, execute it, and report "
            "the accuracy / metrics it prints. If it fails, report the exact error."
        )
        accuracy: dict[str, Any] = {}
        try:
            res = await self._agents.get("codesmith").run(task, ctx)
            accuracy = {"report": res.output, "ran": True}
        except Exception as exc:  # noqa: BLE001
            accuracy = {"report": f"(run failed: {exc})", "ran": False}
        return {"accuracy": accuracy, "trace": [{"node": "run", "ran": accuracy.get("ran")}]}

    def _assemble_node(self, state: ResearchState) -> ResearchState:
        d = state["discovery"]
        branch = state["branch"]
        wants_run = bool(state.get("auto_run"))
        approved = RUN_APPROVAL_KEY in (state.get("approved_ops") or set())
        needs_approval = None
        if wants_run and not approved:
            needs_approval = {
                "operation": RUN_APPROVAL_KEY,
                "reason": "Executing discovered/generated code locally requires approval.",
                "hint": f"Re-run with approved_ops={{'{RUN_APPROVAL_KEY}'}} to execute.",
            }
        result = {
            "goal": state["goal"],
            "branch": branch,
            "found_model": [m.model_dump() for m in d.models],
            "found_dataset": [ds.model_dump() for ds in d.datasets],
            "papers": [p.model_dump() for p in d.papers],
            "hardware": state.get("hardware"),
            "generated": state.get("generated"),
            "accuracy": state.get("accuracy"),
            "summary": self._summary(state),
        }
        return {"result": result, "needs_approval": needs_approval}

    # ── formatting helpers ───────────────────────────────────────────────────
    @staticmethod
    def _format_resources(d: Discovery) -> str:
        lines: list[str] = []
        if d.models:
            lines.append("MODELS:")
            lines += [f"  - {m.id} ({m.url}) — {m.task} {m.popularity}".rstrip() for m in d.models]
        if d.datasets:
            lines.append("DATASETS:")
            lines += [f"  - {ds.id} ({ds.url})".rstrip() for ds in d.datasets]
        if d.papers:
            lines.append("PAPERS:")
            lines += [f"  - {p.title} ({p.url}) code={p.code_url}".rstrip() for p in d.papers]
        return "\n".join(lines) or "(nothing found)"

    @staticmethod
    def _summary(state: ResearchState) -> str:
        d = state["discovery"]
        branch = state["branch"]
        head = {
            "both": "Found a model and dataset — generated an eval pipeline to measure accuracy.",
            "model_only": "Found a model only — generated an inference pipeline.",
            "data_only": "Found a dataset only — generated a training script.",
            "neither": "No ready model/dataset — located the paper and generated a reproduction.",
        }[branch]
        bits = [head]
        if d.models:
            bits.append(f"Model: {d.models[0].id}.")
        if d.datasets:
            bits.append(f"Dataset: {d.datasets[0].id}.")
        if branch == "neither" and d.papers:
            bits.append(f"Paper: {d.papers[0].title}.")
        acc = state.get("accuracy")
        if acc and acc.get("ran"):
            bits.append("Ran the code locally — see accuracy report.")
        return " ".join(bits)

    # ── public API ───────────────────────────────────────────────────────────
    async def run(
        self,
        goal: str,
        *,
        session_id: str = "default",
        approved_ops: set[str] | None = None,
        auto_run: bool = False,
    ) -> dict[str, Any]:
        state: ResearchState = {
            "goal": goal,
            "session_id": session_id,
            "approved_ops": set(approved_ops or set()),
            "auto_run": auto_run,
        }
        begin_task_tool_history()
        try:
            with track_usage() as meter:
                final = await self._graph.ainvoke(state)
        finally:
            clear_task_tool_history()
        tokens = meter.snapshot()
        log.info("research.tokens", total=tokens["total_tokens"], calls=tokens["calls"])
        # Surface the total (and per-agent breakdown) on the result for the UI.
        if isinstance(final.get("result"), dict):
            final["result"]["tokens"] = tokens
        final["tokens"] = tokens
        return final


__all__ = ["ResearchOrchestrator", "Discovery", "RUN_APPROVAL_KEY"]
