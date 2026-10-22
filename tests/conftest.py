"""Shared test fixtures. The whole suite runs offline against the mock provider."""

from __future__ import annotations

import os

import pytest

# Force a fully offline, deterministic configuration before settings load.
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("MEMORY_BACKEND", "memory")
os.environ.setdefault("WORKSPACE_DIR", "./data/test_workspace")
os.environ.setdefault("SESSIONS_DIR", "./data/test_sessions")
os.environ.setdefault("SECRETS_FILE", "./data/test_secrets.json")
os.environ.setdefault("PERSIST_GUEST_SESSIONS", "false")  # tests assume guests are memory-only
os.environ.setdefault("USAGE_LEDGER_FILE", "./data/test_usage.jsonl")
os.environ.setdefault("UI_PREFERENCES_FILE", "./data/test_ui_preferences.json")
os.environ.setdefault("MCP_CONFIG_FILE", "./data/test_mcp_servers.json")
os.environ.setdefault("GOOGLE_OAUTH_CLIENT_FILE", "./data/test_google_oauth_client.json")
os.environ.setdefault("GOOGLE_ACCOUNTS_FILE", "./data/test_google_accounts.json")
os.environ.setdefault("SCHEDULES_FILE", "./data/test_schedules.json")
os.environ.setdefault("USERS_FILE", "./data/test_users.json")
os.environ.setdefault("AUTH_SECRET_FILE", "./data/test_auth_secret.key")
os.environ.setdefault("APPROVAL_POLICIES_FILE", "./data/test_approval_policies.json")
os.environ.setdefault("AUDIT_LOG_FILE", "./data/test_audit_log.jsonl")
os.environ.setdefault("LOG_LEVEL", "ERROR")


@pytest.fixture(autouse=True)
def _clean_secrets():
    """Keep the UI-managed key store isolated between tests."""
    import contextlib
    from pathlib import Path

    paths = [
        Path(os.environ["SECRETS_FILE"]),
        Path(os.environ["USAGE_LEDGER_FILE"]),
        Path(os.environ["MCP_CONFIG_FILE"]),
        Path(os.environ["GOOGLE_OAUTH_CLIENT_FILE"]),
        Path(os.environ["GOOGLE_ACCOUNTS_FILE"]),
        Path(os.environ["SCHEDULES_FILE"]),
        Path(os.environ["APPROVAL_POLICIES_FILE"]),
        Path(os.environ["AUDIT_LOG_FILE"]),
        Path(os.environ["USERS_FILE"]),
    ]
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            p.unlink()
    yield
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


@pytest.fixture
def settings():
    from ai_employee.config import reload_settings

    return reload_settings()


@pytest.fixture
def brain():
    from ai_employee.brains.providers.mock import MockBrain

    return MockBrain(embedding_dim=64)


@pytest.fixture
def registry(settings):
    from ai_employee.tools import default_registry

    return default_registry(settings)


@pytest.fixture
async def employee():
    from ai_employee.brains.factory import reset_brain_cache
    from ai_employee.core.employee import AIEmployee

    reset_brain_cache()
    emp = AIEmployee()
    yield emp
    await emp.aclose()


async def drain_background_tasks(timeout: float = 10.0) -> None:
    """Stop UI task runs a test started but did not wait for.

    ``TaskManager`` runs each task as a background asyncio task. A test that only
    checks the immediate API response leaves it running; if it is still inside a
    subprocess spawn when pytest-asyncio closes the loop, loop shutdown can
    deadlock. Cancel and await them here, while this test's loop is still alive.
    """
    import asyncio

    current = asyncio.current_task()
    leftovers = [
        t
        for t in asyncio.all_tasks()
        if t is not current
        and not t.done()
        and getattr(t.get_coro(), "__qualname__", "") == "TaskManager._execute"
    ]
    for task in leftovers:
        task.cancel()
    if leftovers:
        await asyncio.wait(leftovers, timeout=timeout)


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    from ai_employee.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await drain_background_tasks()
