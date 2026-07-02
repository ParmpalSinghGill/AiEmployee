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
os.environ.setdefault("USAGE_LEDGER_FILE", "./data/test_usage.jsonl")
os.environ.setdefault("LOG_LEVEL", "ERROR")


@pytest.fixture(autouse=True)
def _clean_secrets():
    """Keep the UI-managed key store isolated between tests."""
    import contextlib
    from pathlib import Path

    paths = [Path(os.environ["SECRETS_FILE"]), Path(os.environ["USAGE_LEDGER_FILE"])]
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


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    from ai_employee.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
