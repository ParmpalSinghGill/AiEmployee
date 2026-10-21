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
    # It still counts as one attempted step, not zero.
    assert usage["agents_used"][0]["steps"] == 1


def test_summarize_usage_counts_planner_and_ceo_phases() -> None:
    """Auxiliary phases are real work; they must not render as "0 steps"."""
    usage = summarize_usage(
        step_results=[{"id": 1, "agent": "coding", "verified": True}],
        trace=[
            {"node": "plan", "agent": "planner", "steps": 1},
            {"node": "execute", "step": 1, "agent": "coding", "tools": []},
            {"node": "synthesize", "agent": "ceo"},
        ],
    )

    agents = {a["agent"]: a for a in usage["agents_used"]}
    assert agents["planner"]["steps"] == 1
    assert agents["ceo"]["steps"] == 1
    # A trace entry for a step already counted in step_results is not double-counted.
    assert agents["coding"]["steps"] == 1


def test_summarize_usage_keeps_tools_for_failed_and_halted_steps() -> None:
    usage = summarize_usage(
        step_results=[
            {"id": 1, "agent": "coding", "verified": False},
            # A halted step has no record; its trace carries the agent + calls.
        ],
        trace=[
            {
                "node": "execute",
                "step": 1,
                "agent": "coding",
                "verified": False,
                "tools": [
                    {"tool": "shell", "ok": True},
                    {"tool": "shell", "ok": False},
                ],
            },
            {
                "node": "execute",
                "step": 2,
                "agent": "qa",
                "halted": "approval",
                "tools": [{"tool": "python_exec", "ok": True}],
            },
        ],
    )

    agents = {a["agent"]: a for a in usage["agents_used"]}
    assert agents["coding"]["steps"] == 1
    assert agents["coding"]["tools"] == ["shell"]
    assert agents["coding"]["ok"] == 1
    assert agents["coding"]["failed"] == 1
    assert agents["qa"]["steps"] == 1
    assert agents["qa"]["tools"] == ["python_exec"]
    tools = {t["tool"]: t for t in usage["tools_used"]}
    assert tools["shell"]["calls"] == 2
    assert tools["python_exec"]["agents"] == ["qa"]


def test_summarize_usage_accepts_tool_trace_key() -> None:
    """Older snapshots may name the per-step key ``tool_trace``."""
    usage = summarize_usage(
        step_results=[{"id": 1, "agent": "coding", "verified": False}],
        trace=[
            {
                "node": "execute",
                "step": 1,
                "agent": "coding",
                "tool_trace": [{"tool": "read_file", "ok": True}],
            }
        ],
    )
    assert usage["tools_used"][0]["tool"] == "read_file"
    assert usage["tools_used"][0]["agents"] == ["coding"]


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
