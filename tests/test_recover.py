"""Recover a lost run from the structured server log.

Guest sessions were memory-only, so a restart lost them. These tests cover the
log parser, the reconstruction into a task snapshot, and importing it as a
session owned by the requester (so it reappears in Saved tasks).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from ai_employee.observability.recover import (
    existing_task_ids,
    import_all_new,
    parse_events,
    recover_runs,
)
from ai_employee.observability.store import SessionStore

_LOG = """\
2026-10-18T10:00:00.000000Z [info     ] task.started                   goal='do the thing' task_id=AAA
2026-10-18T10:00:01.000000Z [info     ] planner.plan_ready             node=plan plan=[{'id': 1, 'agent': 'coding', 'description': 'Write code'}] task_id=AAA
2026-10-18T10:00:02.000000Z [info     ] agent.context                  agent=coding node=execute retry=0 session=S1 step=1 task='Write code' task_id=AAA
2026-10-18T10:00:03.000000Z [info     ] tool.execute                   args={'cmd': 'echo hi'} node=execute retry=0 step=1 task_id=AAA tool=shell
2026-10-18T10:00:04.000000Z [info     ] tool.result                    node=execute ok=True output='hi' retry=0 step=1 task_id=AAA tool=shell
2026-10-18T10:00:05.000000Z [info     ] agent.answer                   agent=coding node=execute output='first try' retry=0 session=S1 step=1 task_id=AAA
2026-10-18T10:00:06.000000Z [info     ] verifier.verdict               confidence=0.3 issues=['missing test'] node=execute passed=False retry=0 step=1 task_id=AAA
2026-10-18T10:00:07.000000Z [info     ] agent.answer                   agent=coding node=execute output='fixed' retry=1 session=S1 step=1 task_id=AAA
2026-10-18T10:00:08.000000Z [info     ] verifier.verdict               confidence=0.9 issues=[] node=execute passed=True retry=1 step=1 task_id=AAA
2026-10-18T10:00:09.000000Z [info     ] brain.response                 content='final answer' node=synthesize purpose=synthesize task_id=AAA
2026-10-18T10:00:10.000000Z [info     ] task.completed                 confidence=0.8 task_id=AAA
INFO:     127.0.0.1:50980 - "GET /ui HTTP/1.1" 200 OK
HTTP Request: POST https://api.example/chat "HTTP/1.1 200 OK"
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "server.log"
    p.write_text(_LOG, encoding="utf-8")
    return p


def test_parse_events_extracts_typed_fields(tmp_path: Path) -> None:
    events = parse_events(_write(tmp_path))
    assert [e["event"] for e in events][0] == "task.started"
    started = events[0]
    assert started["fields"]["goal"] == "do the thing"
    assert started["fields"]["task_id"] == "AAA"
    tool = next(e for e in events if e["event"] == "tool.execute")
    assert tool["fields"]["args"] == {"cmd": "echo hi"}
    assert tool["fields"]["retry"] == 0
    tool_res = next(e for e in events if e["event"] == "tool.result")
    assert tool_res["fields"]["ok"] is True
    # Non-structlog lines (uvicorn / httpx) are ignored.
    assert all("HTTP" not in e["event"] for e in events)


def test_recover_run_reconstructs_result_and_retry(tmp_path: Path) -> None:
    runs = recover_runs(_write(tmp_path))
    assert len(runs) == 1
    run = runs[0]
    assert run.task_id == "AAA" and run.session_id == "S1"
    assert run.goal == "do the thing"
    assert run.status == "completed"

    res = run.result
    assert res["answer"] == "final answer"
    assert res["confidence"] == 0.8
    step = res["steps"][0]
    assert step["agent"] == "coding"
    assert step["verified"] is True
    assert step["retries"] == 1
    assert step["output"] == "fixed"  # the retry's answer wins
    # Retry tagging survives so Explain can split the attempts.
    assert any(e["fields"].get("retry") == 1 for e in run.events)


def test_import_all_new_writes_and_dedups(tmp_path: Path) -> None:
    log = _write(tmp_path)
    store = SessionStore(tmp_path / "sessions")
    imported = import_all_new(store, log, owner_id="usr_x", owner_name="X")
    assert len(imported) == 1
    meta, snap = imported[0]
    assert meta["user_id"] == "usr_x" and meta["run_ids"] == ["AAA"]

    assert (tmp_path / "sessions" / "S1" / "session.json").exists()
    assert (tmp_path / "sessions" / "S1" / "runs" / "AAA.json").exists()
    assert existing_task_ids(store) == {"AAA"}

    # Second run is a no-op — the task is already saved.
    assert import_all_new(store, log, owner_id="usr_x", owner_name="X") == []


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    import ai_employee.observability.tasks as tasks_mod
    from ai_employee.api.app import create_app
    from ai_employee.config import get_settings

    # Point recovery at a temp log + sessions dir so the real data/ isn't touched.
    s = get_settings()
    monkeypatch.setattr(s, "sessions_dir", tmp_path / "sessions")
    monkeypatch.setattr(s, "persist_guest_sessions", True, raising=False)
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "server.log").write_text(_LOG, encoding="utf-8")

    # Force a fresh task-manager bound to the temp dir, and drop it afterwards so
    # other tests rebuild one from their own settings.
    tasks_mod._MANAGER = None
    app = create_app()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        tasks_mod._MANAGER = None


async def test_recover_endpoint_imports_for_requester(client) -> None:
    r = await client.post("/ui/api/sessions/recover")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["imported"][0]["task_id"] == "AAA"

    # Owned by the requesting (guest) user, so it now appears in Saved tasks.
    listed = (await client.get("/ui/api/sessions")).json()["sessions"]
    entry = next(s for s in listed if s["id"] == "S1")
    assert entry["runs"][0]["id"] == "AAA"

    # And it opens with its events intact.
    detail = (await client.get("/ui/api/sessions/S1")).json()
    snap = detail["runs"][0]
    assert snap["result"]["answer"] == "final answer"
    assert any(e["event"] == "agent.answer" for e in snap["events"])

    # Recovering again is a no-op.
    assert (await client.post("/ui/api/sessions/recover")).json()["count"] == 0


async def test_guest_sessions_persist_when_enabled(tmp_path, employee, monkeypatch) -> None:
    """The fix for 'restarts lose my task': guest runs now hit the store."""
    from ai_employee.config import get_settings
    from ai_employee.observability.tasks import TaskManager

    monkeypatch.setattr(get_settings(), "persist_guest_sessions", True, raising=False)
    tm = TaskManager(SessionStore(tmp_path / "sessions"))
    task = tm.start(
        employee, "guest goal", owner={"id": "guest:abc", "name": "Test1", "guest": True}
    )
    try:
        assert (tmp_path / "sessions" / task.session_id / "session.json").exists()
    finally:
        tm.cancel(task.id)


async def test_guest_sessions_memory_only_when_disabled(tmp_path, employee, monkeypatch) -> None:
    from ai_employee.config import get_settings
    from ai_employee.observability.tasks import TaskManager

    monkeypatch.setattr(get_settings(), "persist_guest_sessions", False, raising=False)
    tm = TaskManager(SessionStore(tmp_path / "sessions"))
    task = tm.start(
        employee, "guest goal", owner={"id": "guest:abc", "name": "Test1", "guest": True}
    )
    try:
        assert not (tmp_path / "sessions" / task.session_id / "session.json").exists()
    finally:
        tm.cancel(task.id)
