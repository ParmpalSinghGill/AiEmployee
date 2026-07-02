"""Save & resume: session/task/memory persistence."""

from __future__ import annotations

from ai_employee.memory import MemoryNamespace
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.memory.manager import MemoryManager
from ai_employee.observability.store import SessionStore
from ai_employee.observability.tasks import TaskRecord


def test_session_store_roundtrip(tmp_path):
    store = SessionStore(tmp_path)
    store.save_session({"id": "s1", "title": "Build X", "created_at": "2026-06-28T00:00:00",
                        "run_ids": ["t1"]})
    store.save_task("s1", {
        "id": "t1", "goal": "Build X", "session_id": "s1", "parent_id": None,
        "status": "completed", "created_at": "2026-06-28T00:00:00", "finished_at": None,
        "events": [{"seq": 1, "ts": "t", "level": "info", "event": "agent.start", "fields": {"agent": "coding"}}],
        "last_seq": 1, "result": {"answer": "done"}, "error": None,
    })
    sessions, tasks = store.load_all()
    assert len(sessions) == 1 and sessions[0]["title"] == "Build X"
    assert "t1" in tasks and tasks["t1"]["result"]["answer"] == "done"
    assert tasks["t1"]["events"][0]["event"] == "agent.start"


def test_memory_file_roundtrip(tmp_path):
    store = SessionStore(tmp_path)
    store.save_memory([{"id": "m1", "namespace": "long_term", "content": "hi",
                        "metadata": {}, "embedding": None, "session_id": "s1",
                        "created_at": "2026-06-28T00:00:00", "score": None}])
    loaded = store.load_memory()
    assert loaded and loaded[0]["content"] == "hi"


async def test_memory_manager_export_import(brain):
    src = MemoryManager(InMemoryVectorStore(), brain)
    await src.record("Paris is the capital of France.", namespace=MemoryNamespace.long_term)
    dump = src.export_records()
    assert dump and src.is_persistable

    dst = MemoryManager(InMemoryVectorStore(), brain)
    dst.import_records(dump)
    hits = await dst.recall("capital of France", k=1)
    assert hits and "Paris" in hits[0].content


def test_taskrecord_from_snapshot_clamps_running():
    # A 'running' task with no result (process died) is kept as a resumable
    # 'pending' so the app can relaunch it on startup instead of losing it.
    rec = TaskRecord.from_snapshot({
        "id": "t9", "goal": "g", "session_id": "s9", "parent_id": None,
        "status": "running", "events": [], "last_seq": 0, "result": None, "error": None,
    })
    assert rec.status.value == "pending"
    assert rec.resumable is True

    # A 'running' task that had produced a result becomes 'completed'.
    rec2 = TaskRecord.from_snapshot({
        "id": "t10", "goal": "g", "session_id": "s9", "parent_id": None,
        "status": "running", "events": [], "last_seq": 0, "result": {"answer": "x"}, "error": None,
    })
    assert rec2.status.value == "completed"
    assert rec2.resumable is False

    # A run interrupted mid-flight (has events, no result) is NOT auto-resumed —
    # it's marked failed so a partial execution isn't blindly re-run.
    rec3 = TaskRecord.from_snapshot({
        "id": "t11", "goal": "g", "session_id": "s9", "parent_id": None,
        "status": "running", "last_seq": 1, "result": None, "error": None,
        "events": [{"seq": 1, "ts": "t", "event": "agent.start", "fields": {}}],
    })
    assert rec3.status.value == "failed"
    assert rec3.resumable is False
    assert "interrupted" in (rec3.error or "")


async def test_taskmanager_loads_persisted_session(tmp_path, employee):
    """A new TaskManager over the same store sees previously-saved sessions."""
    from ai_employee.observability.tasks import TaskManager

    store = SessionStore(tmp_path)
    tm = TaskManager(store)
    task = tm.start(employee, "Persisted goal")
    # Wait for it to finish so the completed snapshot is written.
    import asyncio
    for _ in range(100):
        if tm.get_task(task.id).status.value in {"completed", "failed", "awaiting_approval"}:
            break
        await asyncio.sleep(0.05)

    # Simulate a fresh process: brand-new manager, same store on disk.
    tm2 = TaskManager(SessionStore(tmp_path))
    assert task.session_id in {s["id"] for s in tm2.list_sessions()}
    reloaded = tm2.get_task(task.id)
    assert reloaded is not None
    assert reloaded.goal == "Persisted goal"
    detail = tm2.session_detail(task.session_id)
    assert detail and detail["runs"]
