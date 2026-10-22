"""End-to-end orchestration and API tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


async def test_employee_run_pipeline(employee):
    result = await employee.run("Research async python and summarize", session_id="t1")
    assert result["answer"]
    assert result["plan"]["steps"]
    assert result["steps"]
    assert result["confidence"] is not None


async def test_employee_chat(employee):
    reply = await employee.chat("Hello", session_id="t2")
    assert reply


async def test_ask_single_agent(employee):
    out = await employee.ask_agent("documentation", "Document a function")
    assert out["agent"] == "documentation"
    assert out["success"]


@pytest.fixture
async def client():
    from ai_employee.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_api_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_api_agents_and_tools(client):
    names = {a["name"] for a in (await client.get("/agents")).json()["agents"]}
    # The full listing includes comms even before Google is connected.
    assert {"research", "coding", "testing", "git", "reviewer", "comms"} <= names
    assert len((await client.get("/tools")).json()["tools"]) >= 8


async def test_api_chat(client):
    r = await client.post("/chat", json={"message": "hi", "session_id": "api"})
    assert r.status_code == 200 and r.json()["reply"]


async def test_api_task(client):
    r = await client.post("/task", json={"goal": "Summarize REST APIs"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["task_id"]


async def test_api_knowledge_roundtrip(client):
    ing = await client.post(
        "/knowledge/ingest",
        json={"source": "GraphQL is a query language for APIs developed by Facebook."},
    )
    assert ing.json()["chunks"] >= 1
    q = await client.post("/knowledge/query", json={"question": "Who developed GraphQL?"})
    assert q.json()["used_chunks"] >= 1
