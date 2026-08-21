"""Dependency skipping must not wipe an analysis pipeline after a weak step."""

from __future__ import annotations

from ai_employee.agents import AgentRegistry, AgentResult
from ai_employee.memory import MemoryManager
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.tools.registry import ToolRegistry
from ai_employee.workflows.employee_graph import Orchestrator, _looks_unusable


class _StubAgent:
    def __init__(self, name: str, handler):
        self.name = name
        self.role = name
        self.description = f"stub {name}"
        self._handler = handler

    async def run(self, task, ctx):
        return await self._handler(task, ctx)


def test_data_analysis_empty_output_is_not_unusable():
    assert not _looks_unusable({"id": 3, "agent": "data_analysis", "output": "", "verified": False})
    assert _looks_unusable({"id": 3, "agent": "data_analysis", "output": "", "skipped": True})


def test_resource_empty_output_is_still_unusable():
    assert _looks_unusable({"id": 1, "agent": "resource_acquirer", "output": ""})


async def test_analysis_pipeline_continues_after_empty_clean_step(brain, settings):
    ran: list[int] = []
    ws = settings.workspace_dir
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "sales.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    async def planner_run(goal, ctx):
        return AgentResult(
            agent="planner",
            success=True,
            metadata={
                "plan": {
                    "steps": [
                        {
                            "id": 1,
                            "agent": "resource_acquirer",
                            "description": "download dataset",
                            "depends_on": [],
                            "success_criteria": "file exists",
                        },
                        {
                            "id": 2,
                            "agent": "data_analysis",
                            "description": "clean the dataset and save a file",
                            "depends_on": [1],
                            "success_criteria": "cleaned csv",
                        },
                        {
                            "id": 3,
                            "agent": "data_analysis",
                            "description": "compute KPIs",
                            "depends_on": [2],
                            "success_criteria": "kpis",
                        },
                    ]
                }
            },
        )

    async def acquire_run(task, ctx):
        ran.append(1)
        return AgentResult(
            agent="resource_acquirer",
            output="Downloaded dataset to ./data/sales.csv",
            confidence=0.8,
        )

    async def analysis_run(task, ctx):
        if "clean" in task.lower():
            ran.append(2)
            return AgentResult(
                agent="data_analysis",
                output="",
                confidence=0.2,
                tool_trace=[
                    {
                        "tool": "python_exec",
                        "ok": True,
                        "output": {"stdout": "shape=(9994, 21)\n"},
                    }
                ],
            )
        ran.append(3)
        return AgentResult(agent="data_analysis", output="Total sales=2.3M", confidence=0.8)

    async def ceo_run(goal, ctx):
        return AgentResult(agent="ceo", output="done", confidence=0.8)

    agents = AgentRegistry()
    agents.register(_StubAgent("planner", planner_run))
    agents.register(_StubAgent("resource_acquirer", acquire_run))
    agents.register(_StubAgent("data_analysis", analysis_run))
    agents.register(_StubAgent("ceo", ceo_run))

    orch = Orchestrator(
        brain=brain,
        agents=agents,
        tools=ToolRegistry(),
        memory=MemoryManager(InMemoryVectorStore(), brain),
        max_retries_per_step=0,
    )
    state = await orch.run("analyze sales", session_id="dep-test")
    results = state.get("step_results") or []
    assert ran == [1, 2, 3]
    assert not any(r.get("skipped") for r in results)
    assert "shape=(9994, 21)" in (results[1].get("output") or "")
    assert "Total sales" in (results[2].get("output") or "")


# ── Dependency gating vs audit issues ────────────────────────────────────────
# A scout that found real model URLs but didn't download them gets an audit issue
# logged. Treating that as "produced nothing usable" skipped hardware_analyst,
# codesmith AND documentation — three agents never ran, and the Explain tab showed
# only the two steps that did.


def test_audit_issue_with_resources_still_feeds_dependents() -> None:
    from ai_employee.workflows.employee_graph import _looks_unusable

    record = {
        "id": 2,
        "agent": "scout",
        "output": "Found YOLO11 and yolov8n with verified URLs. " * 20,
        "resource_audit": {
            "applies": True,
            "outcome": "found_not_downloaded",
            "issues": ["automatic download was not attempted"],
            "has_urls": True,
            "resources": [
                {"name": "YOLO11", "web_url": "https://huggingface.co/Ultralytics/YOLO11"},
            ],
        },
    }
    assert _looks_unusable(record) is False


def test_audit_issue_without_any_resource_blocks_dependents() -> None:
    """No URLs and no resources really is nothing to build on — still blocked."""
    from ai_employee.workflows.employee_graph import _looks_unusable

    record = {
        "id": 2,
        "agent": "scout",
        "output": "I searched several hubs but found no suitable model at all. " * 5,
        "resource_audit": {
            "applies": True,
            "outcome": "not_found",
            "issues": ["no usable resource located"],
            "has_urls": False,
            "resources": [],
        },
    }
    assert _looks_unusable(record) is True


def test_skipped_and_empty_steps_remain_unusable() -> None:
    from ai_employee.workflows.employee_graph import _looks_unusable

    assert _looks_unusable({"id": 1, "agent": "scout", "skipped": True}) is True
    assert _looks_unusable({"id": 1, "agent": "scout", "output": "ok"}) is True
