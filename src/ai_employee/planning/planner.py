"""Hierarchical planner.

Turns a high-level goal into an ordered list of steps, each assigned to the most
appropriate agent. Uses the Brain's structured-output contract so the plan is a
validated Pydantic object regardless of provider.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip

log = get_logger("planner")


class PlanStep(BaseModel):
    id: int = Field(description="1-based step number.")
    description: str = Field(description="What to do in this step.")
    agent: str = Field(description="Name of the agent best suited to execute this step.")
    depends_on: list[int] = Field(
        default_factory=list, description="Step ids that must finish first."
    )
    success_criteria: str = Field(default="", description="How to tell this step succeeded.")


class Plan(BaseModel):
    goal: str
    reasoning: str = Field(default="", description="Brief rationale for the decomposition.")
    steps: list[PlanStep] = Field(default_factory=list)

    def ordered(self) -> list[PlanStep]:
        """Topologically-friendly ordering (stable by id, respecting declared order)."""
        return sorted(self.steps, key=lambda s: s.id)


class Planner:
    def __init__(self, brain: Brain) -> None:
        self._brain = brain

    async def make_plan(self, goal: str, *, agent_catalog: list[dict[str, str]]) -> Plan:
        """Decompose ``goal`` into steps assigned to agents from ``agent_catalog``."""
        roster = "\n".join(
            f"- {a['name']}: {a['role']} — {a['description']}" for a in agent_catalog
        )
        valid_names = {a["name"] for a in agent_catalog}
        # The planner's *input*: the goal, and the agents it may delegate to.
        log.info("planner.input", goal=clip(goal, 1200), agents=sorted(valid_names))

        system = (
            "You are the planning brain of an AI Employee. Decompose the user's goal "
            "into the smallest number of concrete, ordered steps. Assign each step to "
            "exactly one agent from the roster (use the agent's `name`). Add depends_on "
            "links where order matters. Keep it minimal but complete.\n\n"
            "BEFORE adding any acquisition step, check these two shortcuts:\n"
            "- FILE ALREADY IN THE WORKSPACE (the goal names a local file such as "
            "sales.csv): do NOT add an acquire step. Go straight to data_analysis.\n"
            "- FETCHABLE PUBLIC DATA (stock/ETF/index/FX/crypto prices, ticker "
            "fundamentals, weather, or any documented public API): do NOT use "
            "resource_acquirer or hub search — those look for pre-made dataset files. "
            "Assign ONE data_analysis step that writes Python to fetch it directly "
            "(e.g. yfinance for market history), saves it as CSV, and analyzes it.\n\n"
            "RESOURCE TASKS — pick ONE modality. Never mix them up "
            "(a sales CSV is not a video; LibriSpeech is not Roboflow images).\n"
            "TABULAR (csv, xlsx, kaggle tables, sales/finance):\n"
            "- Only when the data must be FOUND as an existing dataset file: acquire "
            "via resource_acquirer + hub_search_all kind=tabular (no YouTube). "
            "Analyze with data_analysis, report with documentation.\n"
            "VIDEO (cctv, youtube, mp4, people counting):\n"
            "- Acquire via resource_acquirer / research + hub_search_all kind=video. "
            "Process with coding; document results.\n"
            "IMAGE (photos, png/jpg, detection/segmentation, Roboflow, COCO-style):\n"
            "- Acquire via resource_acquirer + hub_search_all kind=image "
            "(Roboflow/HF/Kaggle; no YouTube). Inventory/labels with data_analysis "
            "or coding; training/inference scripts with coding.\n"
            "SPEECH / audio (wav, mp3, ASR, TTS, LibriSpeech-style corpora):\n"
            "- Acquire via resource_acquirer + hub_search_all kind=speech "
            "(HF/Kaggle; no YouTube/Roboflow). Profile with data_analysis/coding "
            "(durations, sample rate, transcripts); models via scout+coding.\n"
            "GENERIC (data, dataset, file, model weights):\n"
            "- Step 1 should LOCATE suitable resources, ATTEMPT automatic download, and "
            "record the original URL plus local path when download succeeds.\n"
            "- If automatic download cannot complete (403, login, CAPTCHA, etc.), the "
            "step succeeds by returning the download URL and entering WAITING_FOR_USER "
            "for a manual local path — do NOT treat that as failure.\n"
            "- Use research or scout to FIND and verify URLs across hubs.\n"
            "- Use resource_acquirer to search, download, or pause for manual download.\n"
            "- Do not assign download work to research alone.\n"
            "- Success criteria must NOT require a workspace file before download is "
            "attempted. Accept: (found + downloaded) OR (found + manual download needed).\n"
            "- Never mark a step failed when a suitable URL exists but automation was blocked.\n\n"
            "CHARTS: assign plotting to data_analysis (or coding) — both run Python and "
            "save PNGs. Prefer folding charts into the analysis step rather than adding a "
            "separate one; never assign charting to an agent without python_exec.\n\n"
            "AGENT ROSTER:\n"
            f"{roster}"
        )
        plan = await self._brain.structured_output(
            [Message.system(system), Message.user(f"GOAL: {goal}")],
            Plan,
        )
        plan.goal = goal
        # Repair any invalid agent assignments by routing to a sensible default.
        for step in plan.steps:
            if step.agent not in valid_names:
                log.debug("planner.invalid_agent", agent=step.agent)
                step.agent = "research" if "research" in valid_names else next(iter(valid_names))
        if not plan.steps:
            plan.steps = [PlanStep(id=1, description=goal, agent=next(iter(valid_names)))]
        # The planner's *output*: its rationale and the concrete step breakdown.
        log.info(
            "planner.plan_ready",
            steps=len(plan.steps),
            reasoning=clip(plan.reasoning, 1200),
            plan=[
                {
                    "id": s.id,
                    "agent": s.agent,
                    "description": clip(s.description, 300),
                    "depends_on": s.depends_on,
                    "success_criteria": clip(s.success_criteria, 300),
                }
                for s in plan.ordered()
            ],
        )
        return plan
