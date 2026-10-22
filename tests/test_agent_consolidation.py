"""The 26 → 8 agent consolidation (2026-10-22).

Covers the roster, legacy-name aliases, Google-gated comms, coding's local-only
tool boundary, the per-run tool deny-list, and old names in configs/checkpoints.
"""

from __future__ import annotations

import pytest

from ai_employee.agents.registry import INTEGRATION_CHECKS, build_agent_registry
from ai_employee.agents.routing import (
    INTERNAL_AGENTS,
    LEGACY_AGENT_ALIASES,
    resolve_agent_name,
)
from ai_employee.agents.specs import AGENT_SPECS, NO_ACQUISITION_TOOLS, spec_by_name

WORKERS = {
    "research",
    "coding",
    "testing",
    "git",
    "data_analysis",
    "documentation",
    "reviewer",
    "comms",
}


# ── roster & aliases ─────────────────────────────────────────────────────────


def test_roster_is_the_eight_workers() -> None:
    assert {s.name for s in AGENT_SPECS} == WORKERS


def test_every_alias_targets_a_real_worker() -> None:
    assert set(LEGACY_AGENT_ALIASES.values()) <= WORKERS
    assert not set(LEGACY_AGENT_ALIASES) & WORKERS  # an alias never shadows a worker
    for old, new in {
        "scout": "research",
        "hardware_analyst": "research",
        "resource_downloader": "research",
        "plan_researcher": "research",
        "codesmith": "coding",
        "devops": "coding",
        "database": "coding",
        "qa": "testing",
        "security": "reviewer",
        "monitoring": "reviewer",
        "email": "comms",
        "calendar": "comms",
    }.items():
        assert resolve_agent_name(old) == new


def test_registry_resolves_legacy_names_transparently() -> None:
    reg = build_agent_registry()
    assert reg.get("codesmith") is reg.get("coding")
    assert reg.get("hardware_analyst") is reg.get("research")
    assert "monitoring" in reg
    assert "not_an_agent" not in reg
    with pytest.raises(KeyError):
        reg.get("not_an_agent")
    assert resolve_agent_name("research") == "research"  # real names pass through


# ── comms is offered only when Google is connected ──────────────────────────


def test_comms_is_hidden_from_the_roster_until_google_connects(monkeypatch) -> None:
    reg = build_agent_registry()

    monkeypatch.setitem(INTEGRATION_CHECKS, "google", lambda: False)
    assert "comms" not in {a["name"] for a in reg.catalog()}
    full = {a["name"]: a for a in reg.catalog(include_unavailable=True)}
    assert full["comms"]["available"] == "no"
    assert "comms" in reg  # still registered: legacy `email` steps can resolve

    monkeypatch.setitem(INTEGRATION_CHECKS, "google", lambda: True)
    assert "comms" in {a["name"] for a in reg.catalog()}


async def test_planner_cannot_assign_internal_roles_or_unconnected_comms(monkeypatch) -> None:
    from ai_employee.planning.planner import Planner

    monkeypatch.setitem(INTEGRATION_CHECKS, "google", lambda: False)
    seen: dict = {}

    class _Brain:
        async def structured_output(self, messages, schema, **kwargs):
            seen["enum"] = schema.model_json_schema()
            return schema.model_validate({"goal": "g", "steps": []})

    await Planner(_Brain()).make_plan("g", agent_catalog=build_agent_registry().catalog())
    text = str(seen["enum"])
    for name in INTERNAL_AGENTS | {"comms"}:
        assert f"'{name}'" not in text
    assert "'research'" in text and "'coding'" in text


# ── tool boundaries ──────────────────────────────────────────────────────────


def test_coding_is_local_only() -> None:
    tools = set(spec_by_name()["coding"].tools or ())
    external = {
        "hf_search",
        "fetch_page",
        "web_search",
        "http_request",
        "hub_search_all",
        "github_search",
        "download_resource",
    }
    assert not tools & external
    assert {"python_exec", "shell", "write_file"} <= tools
    prompt = spec_by_name()["coding"].system_prompt
    for folded in ("INFRA / DEVOPS", "DATABASES", "ML CODE", "NO WEB ACCESS"):
        assert folded in prompt


def test_research_owns_hubs_downloads_and_hardware() -> None:
    tools = set(spec_by_name()["research"].tools or ())
    assert {"hf_search", "arxiv_search", "download_resource", "estimate_hardware"} <= tools
    assert not tools & {"python_exec", "shell"}  # never runs local code


def test_merged_agents_keep_their_duties() -> None:
    assert "pass/fail" in spec_by_name()["testing"].system_prompt
    reviewer = spec_by_name()["reviewer"].system_prompt
    assert "SECURITY" in reviewer and "OPERATIONS" in reviewer
    comms = spec_by_name()["comms"]
    assert {"gmail_send", "calendar_create_event"} <= set(comms.tools or ())
    assert comms.requires == "google"


# ── per-run deny-list & refusing unexposed tools ─────────────────────────────


class _CallsThenAnswers:
    """Calls the given tools once each, then answers in text."""

    def __init__(self, *tools: str) -> None:
        self.tools = list(tools)
        self.seen_tools: list[set[str]] = []

    async def chat(self, messages, **kwargs):
        from ai_employee.brains.schemas import ChatResult, ToolCall, Usage

        self.seen_tools.append({t.name for t in kwargs.get("tools") or []})
        if self.tools:
            name = self.tools.pop(0)
            return ChatResult(
                content="",
                tool_calls=[ToolCall(id=name, name=name, arguments={"url": "https://x"})],
                usage=Usage(),
            )
        return ChatResult(content="Done: reported the findings only.", usage=Usage())


def _registry(settings, executed: list[str]):
    from pydantic import BaseModel

    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.registry import ToolRegistry

    def make(tool_name: str):
        class _T(Tool):
            name = tool_name
            description = tool_name
            category = "research"

            class Args(BaseModel):
                url: str = ""

            async def _run(self, args) -> ToolResult:
                executed.append(tool_name)
                return ToolResult(ok=True, output={"ok": True})

        return _T()

    reg = ToolRegistry(settings)
    for name in ("fetch_page", "download_resource", "hf_search"):
        reg.register(make(name))
    return reg


async def _run(agent_tools, task, brain, settings, *, denied=frozenset()):
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )

    class _Agent(ToolUsingAgent):
        name = "research"
        allowed_tools = list(agent_tools)
        max_iterations = 4

    executed: list[str] = []
    ctx = AgentContext(brain=brain, tools=_registry(settings, executed), denied_tools=denied)
    begin_task_tool_history()
    try:
        result = await _Agent().run(task, ctx)
    finally:
        clear_task_tool_history()
    return result, executed


async def test_denied_tools_are_hidden_and_refused(settings) -> None:
    brain = _CallsThenAnswers("download_resource")
    result, executed = await _run(
        ["fetch_page", "download_resource"],
        "download a dataset",
        brain,
        settings,
        denied=NO_ACQUISITION_TOOLS,
    )
    assert "download_resource" not in brain.seen_tools[0]
    assert executed == []  # the call was refused, not executed
    refused = [t for t in result.tool_trace if t["tool"] == "download_resource"]
    assert refused and not refused[0]["ok"]


async def test_tools_outside_the_agent_list_are_refused(settings) -> None:
    # e.g. coding hallucinating hf_search: refused instead of silently executed.
    brain = _CallsThenAnswers("hf_search")
    _, executed = await _run(["fetch_page"], "write the script", brain, settings)
    assert executed == []


async def test_find_only_step_is_not_pushed_to_download(settings) -> None:
    brain = _CallsThenAnswers("fetch_page")
    result, _ = await _run(
        ["fetch_page", "download_resource"], "find a pretrained model", brain, settings
    )
    assert "download_resource" not in {t["tool"] for t in result.tool_trace}


# ── legacy names in configs, checkpoints and modes ──────────────────────────


def test_legacy_agent_limits_fold_into_successors() -> None:
    from ai_employee.config.settings import parse_agent_limits

    limits = parse_agent_limits("devops=20, codesmith=30, scout=9, research=11, qa=5")
    assert limits == {"coding": 30, "research": 11, "testing": 5}


async def test_resumed_checkpoint_with_retired_agent_runs_successor(employee) -> None:
    graph = employee.orchestrator
    assert graph._resolve_agent("scout").name == "research"
    assert graph._resolve_agent("devops").name == "coding"


def test_download_mode_runs_research_with_a_download_brief() -> None:
    from ai_employee.agents.routing import base_agent_for_mode
    from ai_employee.core.employee import _normalize_download_goal

    assert base_agent_for_mode("download") == "research"
    task = _normalize_download_goal("https://huggingface.co/org/model")
    assert task.startswith("DOWNLOAD MODE")
    assert "https://huggingface.co/org/model" in task


def test_plan_and_research_modes_deny_acquisition() -> None:
    import inspect

    from ai_employee.workflows import plan_graph, research_graph

    assert "denied_tools=NO_ACQUISITION_TOOLS" in inspect.getsource(plan_graph)
    src = inspect.getsource(research_graph.ResearchOrchestrator._discover_node)
    assert "NO_ACQUISITION_TOOLS" in src
    assert '"scout"' not in inspect.getsource(research_graph)
    assert "download_resource" in NO_ACQUISITION_TOOLS
