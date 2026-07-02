"""External coding-agent CLI integration."""

from __future__ import annotations

import pytest

from ai_employee.integrations.code_agents import cli_available, run_external_code_agent


@pytest.mark.asyncio
async def test_unknown_backend_returns_error() -> None:
    res = await run_external_code_agent("not-a-backend", "do something")
    assert res["ok"] is False
    assert res["error"] == "bad_backend"


def test_cli_available_unknown_backend() -> None:
    assert cli_available("not-a-backend") is False


def test_cli_available_cursor_uses_configured_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_employee.config import get_settings

    s = get_settings()
    s.cursor_cli_cmd = "cursor-agent-test-cmd"
    monkeypatch.setattr(
        "ai_employee.integrations.code_agents.shutil.which",
        lambda cmd: "/usr/bin/cursor-agent-test-cmd" if cmd == "cursor-agent-test-cmd" else None,
    )
    assert cli_available("cursor") is True
