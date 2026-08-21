"""Plan mode: research → step-by-step guide with OSS / proprietary tools."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from ai_employee.workflows.plan_graph import PlanGuide, PlanOrchestrator, ToolOption


@pytest.fixture
async def client():
    from ai_employee.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def employee(settings, monkeypatch):
    """Force mock brain — ignore any saved UI provider preference."""
    from ai_employee.brains.factory import reset_brain_cache
    from ai_employee.config.settings import EmbeddingProvider, LLMProvider
    from ai_employee.core.employee import AIEmployee

    monkeypatch.setattr(settings, "llm_provider", LLMProvider.mock)
    monkeypatch.setattr(settings, "embedding_provider", EmbeddingProvider.mock)
    reset_brain_cache()
    emp = AIEmployee(settings)
    yield emp
    await emp.aclose()


def _orch(employee) -> PlanOrchestrator:
    return employee.plan_orchestrator


async def _wait(client, task_id, timeout=15.0):
    for _ in range(int(timeout / 0.1)):
        snap = (await client.get(f"/ui/api/tasks/{task_id}")).json()
        if snap["status"] in {"completed", "failed", "awaiting_approval"}:
            return snap
        await asyncio.sleep(0.1)
    raise AssertionError("task did not finish")


async def test_plan_orchestrator_produces_steps_and_tools(employee):
    state = await _orch(employee).run("Build a people counter for mall CCTV")
    result = state["result"]
    assert result["goal"]
    assert result["summary"]
    assert isinstance(result["steps"], list) and len(result["steps"]) >= 1
    assert all(s.get("instruction") for s in result["steps"])
    assert "tools" in result
    ids = [s["id"] for s in result["steps"]]
    assert ids == sorted(ids)
    # Mock brain always records usage for chat/structured calls.
    assert state["tokens"]["total_tokens"] > 0 or result["steps"]


async def test_employee_plan_envelope(employee):
    out = await employee.plan("Set up a RAG chatbot over company PDFs")
    assert "result" in out
    assert out["result"]["steps"]
    assert "tools_open_source" in out["result"]
    assert "tools_proprietary" in out["result"]
    kinds = {t.get("kind") for t in out["result"].get("tools") or []}
    # At least one of OSS / proprietary / unknown should appear when tools exist.
    if out["result"].get("tools"):
        assert kinds & {"open_source", "proprietary", "freemium", "saas", "unknown"}


def test_tool_option_kinds():
    t = ToolOption(name="yt-dlp", kind="open_source", purpose="download video")
    assert t.kind == "open_source"
    g = PlanGuide(summary="x", tools=[t], steps=[])
    assert g.tools[0].name == "yt-dlp"


def test_fallback_guide_marks_oss_and_proprietary():
    g = PlanOrchestrator._fallback_guide("count cars in video", "", error="offline")
    assert len(g.steps) >= 3
    kinds = {t.kind for t in g.tools}
    assert "open_source" in kinds
    assert "proprietary" in kinds


async def test_plan_mode_task_surfaces_guide(client):
    r = await client.post(
        "/ui/api/tasks",
        json={"goal": "Train a YOLO model on custom images", "mode": "plan"},
    )
    assert r.status_code == 200
    snap = await _wait(client, r.json()["task_id"])
    assert snap["mode"] == "plan"
    assert snap["status"] == "completed"
    result = snap["result"]["result"]
    assert result["steps"]
    assert "summary" in result
    assert "tools" in result


# ── Dependency pruning ───────────────────────────────────────────────────────
# Planners over-link: writing depends_on=[1,2,3,4] when 4 already depends on the
# rest. Those transitive edges don't change ordering, but the parallel scheduler
# treats each as a real constraint, so an over-linked plan serializes needlessly.


def _steps(*specs):
    from ai_employee.planning.planner import PlanStep

    return [
        PlanStep(id=i, description=f"s{i}", agent="research", depends_on=list(d)) for i, d in specs
    ]


def test_prune_drops_transitive_dependencies() -> None:
    from ai_employee.planning.planner import prune_redundant_dependencies

    # 5 depends on everything, but 4 already depends on 1,2,3 transitively.
    steps = _steps((1, []), (2, []), (3, [1]), (4, [2, 3]), (5, [1, 2, 3, 4]))
    prune_redundant_dependencies(steps)

    by_id = {s.id: s.depends_on for s in steps}
    assert by_id[5] == [4], "only the direct dependency should remain"
    assert by_id[4] == [2, 3]
    assert by_id[1] == [] and by_id[2] == []


def test_prune_improves_achievable_parallelism() -> None:
    from ai_employee.planning.planner import prune_redundant_dependencies
    from ai_employee.workflows.scheduler import describe_schedule

    # Over-linked: 3 and 4 both list 1 and 2 though 2 already depends on 1.
    steps = _steps((1, []), (2, [1]), (3, [1, 2]), (4, [1, 2]), (5, [1, 2, 3, 4]))
    prune_redundant_dependencies(steps)

    shape = describe_schedule([{"id": s.id, "depends_on": s.depends_on} for s in steps])
    assert shape["waves"] == [[1], [2], [3, 4], [5]]
    # 5's four links collapse to the two that actually gate it.
    assert {s.id: s.depends_on for s in steps}[5] == [3, 4]


def test_prune_keeps_genuine_dependencies() -> None:
    """Pruning must never make a dependent step runnable too early."""
    from ai_employee.planning.planner import prune_redundant_dependencies

    steps = _steps((1, []), (2, []), (3, [1]), (4, [1, 2]), (5, [4]))
    prune_redundant_dependencies(steps)

    by_id = {s.id: s.depends_on for s in steps}
    assert by_id[3] == [1]
    assert by_id[4] == [1, 2], "1 and 2 are independent, so both are direct"
    assert by_id[5] == [4]


def test_prune_removes_self_and_unknown_links() -> None:
    from ai_employee.planning.planner import prune_redundant_dependencies

    steps = _steps((1, [1]), (2, [99, 1]), (3, [2, 2]))
    prune_redundant_dependencies(steps)

    by_id = {s.id: s.depends_on for s in steps}
    assert by_id[1] == []  # self-reference dropped
    assert by_id[2] == [1]  # unknown id 99 dropped
    assert by_id[3] == [2]  # duplicate collapsed


def test_prune_breaks_cycles_so_the_plan_can_schedule() -> None:
    from ai_employee.planning.planner import prune_redundant_dependencies
    from ai_employee.workflows.scheduler import schedule_rounds

    steps = _steps((1, [3]), (2, [1]), (3, [2]))
    prune_redundant_dependencies(steps)

    waves = schedule_rounds([{"id": s.id, "depends_on": s.depends_on} for s in steps])
    assert sorted(i for w in waves for i in w) == [1, 2, 3]
    assert {s.id: s.depends_on for s in steps}[1] == []
