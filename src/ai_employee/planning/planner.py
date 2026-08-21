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


def prune_redundant_dependencies(steps: list[PlanStep]) -> list[PlanStep]:
    """Drop dependency links that are already implied, so steps can run in parallel.

    Planners habitually over-link — writing ``depends_on=[1, 2, 3]`` when 3 already
    depends on 1 and 2. Those transitive edges don't change execution order, but the
    scheduler treats each one as a real constraint, so an over-linked plan serializes
    work that could overlap.

    Removes, in order:
      * self-references and ids that aren't in the plan,
      * edges that would create a cycle (kept only if they point backwards),
      * transitive edges: if A→B→C, then C→A is redundant.

    Mutates ``steps`` in place and returns it.
    """
    by_id = {s.id: s for s in steps}

    # Normalize first: no self-links, no unknown ids, no duplicates, stable order.
    for step in steps:
        seen: set[int] = set()
        cleaned: list[int] = []
        for dep in step.depends_on or []:
            if dep == step.id or dep not in by_id or dep in seen:
                continue
            seen.add(dep)
            cleaned.append(dep)
        step.depends_on = cleaned

    # Break cycles: a plan that depends on itself in a loop can never schedule.
    # Keeping only backwards edges (dep id < step id) is enough and matches the
    # planner's own 1-based ordering.
    if _has_cycle(by_id):
        for step in steps:
            step.depends_on = [d for d in step.depends_on if d < step.id]

    # Transitive reduction: drop a direct edge that is already reachable indirectly.
    for step in steps:
        direct = list(step.depends_on)
        keep: list[int] = []
        for dep in direct:
            others = [d for d in direct if d != dep]
            if not any(_reaches(by_id, other, dep) for other in others):
                keep.append(dep)
        step.depends_on = keep
    return steps


def _reaches(by_id: dict[int, PlanStep], start: int, target: int) -> bool:
    """True if ``start`` depends on ``target``, directly or transitively."""
    stack = [start]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        node = by_id.get(cur)
        if node is None:
            continue
        for dep in node.depends_on or []:
            if dep == target:
                return True
            stack.append(dep)
    return False


def _has_cycle(by_id: dict[int, PlanStep]) -> bool:
    state: dict[int, int] = {}  # 0 = visiting, 1 = done

    def visit(node_id: int) -> bool:
        mark = state.get(node_id)
        if mark == 0:
            return True
        if mark == 1:
            return False
        state[node_id] = 0
        node = by_id.get(node_id)
        for dep in (node.depends_on if node else []) or []:
            if dep in by_id and visit(dep):
                return True
        state[node_id] = 1
        return False

    return any(visit(sid) for sid in by_id)


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
            "into the smallest number of concrete steps. Assign each step to "
            "exactly one agent from the roster (use the agent's `name`). "
            "Keep it minimal but complete.\n\n"
            "DEPENDENCIES — this decides what can run in parallel, so be strict.\n"
            "A step lists another in depends_on ONLY IF it literally cannot start "
            "without that step's OUTPUT as its input. Independent steps run at the "
            "same time; every unnecessary dependency makes the run slower for no "
            "reason.\n"
            "- Default to depends_on: [] and add a link only when you can name the "
            "  artifact that flows between the two steps (a file path, a model id, "
            "  a URL, measured numbers).\n"
            "- Do NOT add a dependency merely because one step is listed earlier, "
            "  is 'related', reads more naturally second, or is about the same topic. "
            "  Ordering is not dependency.\n"
            "- Searching different sources is independent: finding a MODEL, finding a "
            "  DATASET, and searching papers do not depend on each other. Run them "
            "  together.\n"
            "- Only list DIRECT dependencies. If C needs B and B needs A, write "
            "  C.depends_on=[B], never [A, B] — transitive links are already implied.\n"
            "- Genuine dependencies look like: analyse a file that an earlier step "
            "  downloaded; size the GPU for a model an earlier step chose; write code "
            "  against a dataset layout an earlier step measured; a final summary that "
            "  consumes several earlier outputs.\n"
            "Worked example — 'find a model and a dataset, size the GPU, write "
            "training code, document it':\n"
            "  1 scout (find model)              depends_on: []\n"
            "  2 resource_acquirer (find data)   depends_on: []      <- independent of 1\n"
            "  3 hardware_analyst (VRAM)         depends_on: [1]     <- needs the model\n"
            "  4 coding (training script)        depends_on: [1, 2]  <- needs both\n"
            "  5 documentation (README)          depends_on: [4]     <- not [1,2,3,4]\n"
            "Here steps 1 and 2 run at once, and 3 runs as soon as 1 lands.\n\n"
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
        prune_redundant_dependencies(plan.steps)
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
