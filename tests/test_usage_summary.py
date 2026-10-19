"""The run/agent output must expose which agents and tools were used.

`agents_used` / `tools_used` are derived from a run's step results and execute
trace so the CLI (and API/Result tab) can show an at-a-glance usage section.
"""

from __future__ import annotations

from ai_employee.observability.usage_summary import summarize_usage


def test_summarize_usage_from_workflow() -> None:
    step_results = [
        {"id": 1, "agent": "scout", "output": "found repos"},
        {"id": 2, "agent": "coding", "output": "wrote script"},
        {"id": 3, "agent": "qa", "output": "(skipped)", "skipped": True},
    ]
    trace = [
        {
            "node": "execute",
            "step": 1,
            "agent": "scout",
            "tools": [
                {"tool": "web_search", "ok": True},
                {"tool": "web_search", "ok": True},
                {"tool": "fetch_page", "ok": False, "error": "403"},
            ],
        },
        {
            "node": "execute",
            "step": 2,
            "agent": "coding",
            "tools": [{"tool": "write_file", "ok": True}],
        },
    ]

    usage = summarize_usage(step_results=step_results, trace=trace)

    agents = {a["agent"]: a for a in usage["agents_used"]}
    assert set(agents) == {"scout", "coding", "qa"}
    assert agents["scout"]["steps"] == 1
    assert agents["scout"]["tools"] == ["fetch_page", "web_search"]
    assert agents["scout"]["ok"] == 2
    assert agents["scout"]["failed"] == 1
    assert agents["qa"]["steps"] == 1
    assert agents["qa"]["tools"] == []

    tools = {t["tool"]: t for t in usage["tools_used"]}
    assert tools["web_search"]["calls"] == 2
    assert tools["web_search"]["ok"] == 2
    assert tools["fetch_page"]["failed"] == 1
    assert tools["fetch_page"]["agents"] == ["scout"]
    assert tools["write_file"]["agents"] == ["coding"]
    # Most-used first.
    assert usage["tools_used"][0]["tool"] == "web_search"


def test_summarize_usage_single_agent() -> None:
    usage = summarize_usage(
        agent="documentation",
        tool_trace=[
            {"tool": "read_file", "ok": True},
            {"tool": "write_file", "ok": True},
        ],
    )
    assert usage["agents_used"] == [
        {
            "agent": "documentation",
            "steps": 0,
            "tools": ["read_file", "write_file"],
            "ok": 2,
            "failed": 0,
        }
    ]
    assert [t["tool"] for t in usage["tools_used"]] == ["read_file", "write_file"]


def test_summarize_usage_agent_from_trace_without_step_record() -> None:
    """A step that halted (approval gate) has a trace entry but no record."""
    usage = summarize_usage(
        step_results=[],
        trace=[{"node": "execute", "step": 1, "agent": "coding", "tools": []}],
    )
    assert [a["agent"] for a in usage["agents_used"]] == ["coding"]


async def test_run_result_includes_usage(employee) -> None:
    result = await employee.run("Research async python and summarize", session_id="usage1")
    assert "agents_used" in result
    assert "tools_used" in result
    agents = {a["agent"] for a in result["agents_used"]}
    # The planner always runs, and the synthesizer (ceo) closes the run.
    assert "planner" in agents
    assert isinstance(result["tools_used"], list)


async def test_ask_agent_result_includes_usage(employee) -> None:
    out = await employee.ask_agent("documentation", "Document a function")
    assert "agents_used" in out
    assert "tools_used" in out
    assert out["agents_used"][0]["agent"] == "documentation"
