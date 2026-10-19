"""GUI 🎚 Limits modal: per-agent iteration & loop settings over the UI API."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

_LIMIT_FIELDS = (
    "agent_max_iterations",
    "agent_max_iterations_by_name",
    "agent_loop_extensions_max",
    "agent_loop_extensions_by_name",
    "agent_loop_check_enabled",
    "explain_max_chars",
    "explain_tool_max_chars",
)


@pytest.fixture
async def client():
    from ai_employee.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _restore_agent_limits():
    """The endpoints mutate the process-wide settings; restore them afterwards."""
    from ai_employee.config import get_settings

    s = get_settings()
    saved = {f: getattr(s, f) for f in _LIMIT_FIELDS}
    yield
    for f, v in saved.items():
        setattr(s, f, v)


async def test_get_returns_limits_and_resolved_agents(client) -> None:
    r = await client.get("/ui/api/agents/limits")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "max_iterations",
        "loop_extensions_max",
        "explain_max_chars",
        "explain_tool_max_chars",
        "agents",
    ):
        assert key in body
    # Explain text is kept in full by default.
    assert body["explain_max_chars"] == 0 and body["explain_tool_max_chars"] == 0
    names = {a["name"] for a in body["agents"]}
    assert "coding" in names and "research" in names
    assert "planner" not in names  # planner/ceo don't run the tool loop
    coding = next(a for a in body["agents"] if a["name"] == "coding")
    assert coding["iterations"] >= 1
    assert coding["max_rounds"] >= coding["iterations"]


async def test_post_sets_limits_live_and_normalizes(client) -> None:
    r = await client.post(
        "/ui/api/agents/limits",
        json={
            "max_iterations": 0,
            "max_iterations_by_name": " Coding = 30 , research=9 ; junk ",
            "loop_extensions_max": 1,
            "loop_extensions_by_name": "coding=4",
            "loop_check_enabled": True,
            "save_default": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["max_iterations_by_name"] == "coding=30,research=9"
    assert body["loop_extensions_by_name"] == "coding=4"

    coding = next(a for a in body["agents"] if a["name"] == "coding")
    assert coding["iterations"] == 30
    assert coding["loops"] == 4
    assert coding["max_rounds"] == 30 * (1 + 4)
    research = next(a for a in body["agents"] if a["name"] == "research")
    assert research["iterations"] == 9  # per-agent override

    # The change is live: a follow-up GET reflects it.
    again = (await client.get("/ui/api/agents/limits")).json()
    assert again["max_iterations_by_name"] == "coding=30,research=9"


async def test_post_clamps_negatives_and_ignores_junk(client) -> None:
    r = await client.post(
        "/ui/api/agents/limits",
        json={
            "max_iterations": -5,
            "max_iterations_by_name": "coding=abc, =2, x, research=3",
            "loop_extensions_max": -1,
            "loop_extensions_by_name": "",
            "loop_check_enabled": False,
        },
    )
    body = r.json()
    assert body["max_iterations"] == 0
    assert body["loop_extensions_max"] == 0
    assert body["max_iterations_by_name"] == "research=3"
    # Loop check off => agents are shown with no extra windows.
    assert all(a["loops"] == 0 for a in body["agents"])


async def test_global_iterations_apply_to_every_agent(client) -> None:
    r = await client.post(
        "/ui/api/agents/limits",
        json={
            "max_iterations": 6,
            "max_iterations_by_name": "",
            "loop_extensions_max": 0,
            "loop_extensions_by_name": "",
            "loop_check_enabled": True,
        },
    )
    body = r.json()
    assert all(a["iterations"] == 6 for a in body["agents"])


async def test_save_default_persists_and_reapplies(client, tmp_path: Path, monkeypatch) -> None:
    from ai_employee.config import get_settings
    from ai_employee.config.ui_preferences import apply_ui_preferences, read_preferences

    s = get_settings()
    monkeypatch.setattr(s, "ui_preferences_file", tmp_path / "prefs.json")

    r = await client.post(
        "/ui/api/agents/limits",
        json={
            "max_iterations": 15,
            "max_iterations_by_name": "coding=40",
            "loop_extensions_max": 3,
            "loop_extensions_by_name": "research=1",
            "loop_check_enabled": True,
            "save_default": True,
        },
    )
    assert r.status_code == 200
    saved = read_preferences(s)
    assert saved["agent_max_iterations"] == 15
    assert saved["agent_max_iterations_by_name"] == "coding=40"
    assert saved["agent_loop_extensions_max"] == 3
    assert saved["agent_loop_extensions_by_name"] == "research=1"
    assert saved["agent_loop_check_enabled"] is True

    # Simulate a server restart: fresh in-memory values, then re-apply.
    s.agent_max_iterations = 0
    s.agent_max_iterations_by_name = ""
    s.agent_loop_extensions_by_name = ""
    apply_ui_preferences(s)
    assert s.agent_max_iterations == 15
    assert s.agent_max_iterations_by_name == "coding=40"
    assert s.agent_loop_extensions_by_name == "research=1"
