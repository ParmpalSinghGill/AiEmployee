"""Web UI: task manager + dashboard API tests."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    from ai_employee.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _wait_done(client, task_id, timeout=10.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        snap = (await client.get(f"/ui/api/tasks/{task_id}")).json()
        if snap["status"] in {"completed", "failed", "awaiting_approval"}:
            return snap
        await asyncio.sleep(0.1)
    raise AssertionError("task did not finish in time")


async def test_ui_index_served(client):
    r = await client.get("/ui")
    assert r.status_code == 200
    assert "AI Employee" in r.text


async def test_root_redirects_to_ui(client):
    r = await client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/ui"


async def test_create_task_runs_and_captures_events(client):
    r = await client.post("/ui/api/tasks", json={"goal": "Summarize async python"})
    assert r.status_code == 200
    body = r.json()
    task_id = body["task_id"]
    assert body["session_id"]

    snap = await _wait_done(client, task_id)
    assert snap["status"] == "completed"
    assert snap["result"]["answer"]
    # Live event capture: agent/tool/verifier activity attributed to this task.
    events = {e["event"] for e in snap["events"]}
    assert any(e.startswith("agent") for e in events)
    assert "verifier.verdict" in events


async def test_download_mode_routes_to_employee_download():
    from ai_employee.api.app import create_app
    from ai_employee.api.deps import get_employee

    calls = []

    class _FakeEmployee:
        async def download(
            self, goal, *, session_id=None, approved_ops=None, resource_decision=None
        ):
            calls.append(
                {
                    "goal": goal,
                    "session_id": session_id,
                    "approved_ops": set(approved_ops or ()),
                    "resource_decision": resource_decision,
                }
            )
            return {
                "goal": goal,
                "answer": "downloaded ok",
                "confidence": 1.0,
                "steps": [],
                "tokens": {"calls": 0},
                "needs_approval": None,
                "trace": [],
            }

        async def run(self, *args, **kwargs):
            raise AssertionError("run() should not be used for download mode")

        async def research(self, *args, **kwargs):
            raise AssertionError("research() should not be used for download mode")

        async def name_task(self, goal):
            return "Download URL"

        def persist_memory(self):
            return None

    app = create_app()
    app.dependency_overrides[get_employee] = lambda: _FakeEmployee()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ui/api/tasks",
            json={"goal": "https://github.com/org/repo", "mode": "download"},
        )
        assert r.status_code == 200
        task_id = r.json()["task_id"]
        snap = await _wait_done(client, task_id)
        assert snap["mode"] == "download"
        assert snap["result"]["answer"] == "downloaded ok"
        assert calls and calls[0]["goal"] == "https://github.com/org/repo"


async def test_sessions_listed(client):
    await client.post("/ui/api/tasks", json={"goal": "List me later"})
    sessions = (await client.get("/ui/api/sessions")).json()["sessions"]
    assert sessions
    assert all("runs" in s for s in sessions)


async def test_followup_shares_session(client):
    start = (await client.post("/ui/api/tasks", json={"goal": "First goal"})).json()
    sid = start["session_id"]
    await _wait_done(client, start["task_id"])

    fu = await client.post(f"/ui/api/sessions/{sid}/followup", json={"goal": "Follow-up goal"})
    assert fu.status_code == 200
    follow = fu.json()
    assert follow["session_id"] == sid

    snap = await _wait_done(client, follow["task_id"])
    assert snap["session_id"] == sid
    assert snap["parent_id"] == start["task_id"]


async def test_followup_unknown_session_404(client):
    r = await client.post("/ui/api/sessions/nope/followup", json={"goal": "x"})
    assert r.status_code == 404


async def test_brain_options_lists_providers(client):
    r = await client.get("/ui/api/brain/options")
    assert r.status_code == 200
    body = r.json()
    assert "local" in body["providers"] and "deepseek" in body["providers"]
    assert "current" in body and body["models"]


async def test_change_brain_applies(client):
    # Switch the model on the mock provider (keeps the suite offline & uncontaminated).
    r = await client.post("/ui/api/brain", json={"provider": "mock", "model": "mock-1"})
    assert r.status_code == 200
    assert r.json()["provider"] == "mock"
    # The change is reflected live via /config.
    cfg = (await client.get("/config")).json()
    assert cfg["provider"] == "mock"


async def test_brain_save_default_persisted(client, tmp_path):
    from ai_employee.config import get_settings

    s = get_settings()
    prefs = tmp_path / "ui_prefs.json"
    s.ui_preferences_file = prefs
    r = await client.post(
        "/ui/api/brain",
        json={"provider": "mock", "model": "mock-1", "save_default": True},
    )
    assert r.status_code == 200
    assert r.json().get("saved_default") is True
    opts = (await client.get("/ui/api/brain/options")).json()
    assert opts["saved"]["brain"]["provider"] == "mock"


async def test_coding_save_default_persisted(client, tmp_path):
    from ai_employee.config import get_settings

    s = get_settings()
    s.ui_preferences_file = tmp_path / "ui_prefs.json"
    r = await client.post("/ui/api/coding", json={"backend": "cursor", "save_default": True})
    assert r.status_code == 200
    opts = (await client.get("/ui/api/coding/options")).json()
    assert opts["saved"]["coding_backend"] == "cursor"


async def test_change_brain_unknown_provider_400(client):
    r = await client.post("/ui/api/brain", json={"provider": "nope"})
    assert r.status_code == 400


async def test_brain_credits_local_unsupported(client):
    r = await client.post("/ui/api/brain/credits", json={"provider": "local"})
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is False


async def test_brain_credits_unknown_provider_400(client):
    r = await client.post("/ui/api/brain/credits", json={"provider": "nope"})
    assert r.status_code == 400


async def test_brain_credits_hosted_provider(client):
    r = await client.post("/ui/api/brain/credits", json={"provider": "openai"})
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    # No key → ok=false; configured key on a provider w/o balance API → supported=false.
    assert body["ok"] is False or body.get("balance") or body["supported"] is False


async def test_clear_memories(client):
    from ai_employee.api.deps import get_employee
    from ai_employee.memory import MemoryNamespace

    emp = get_employee()
    await emp.memory.record("unique recall token abc123", namespace=MemoryNamespace.long_term)
    r = await client.delete("/ui/api/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["cleared"] is True
    assert body.get("backup_saved") is True
    hits = await emp.memory.recall("unique recall token")
    assert not hits
    st = await client.get("/ui/api/memory/settings")
    assert st.status_code == 200
    assert st.json()["backup_available"] is True


async def test_restore_memories(client):
    from ai_employee.api.deps import get_employee
    from ai_employee.memory import MemoryNamespace

    emp = get_employee()
    await emp.memory.record("restore me token xyz", namespace=MemoryNamespace.long_term)
    await client.delete("/ui/api/memory")
    hits = await emp.memory.recall("restore me token")
    assert not hits
    r = await client.post("/ui/api/memory/restore")
    assert r.status_code == 200
    assert r.json()["records_restored"] >= 1
    hits = await emp.memory.recall("restore me token")
    assert hits


async def test_memory_settings_toggle(client):
    r = await client.post("/ui/api/memory/settings", json={"use_stored_memory": False})
    assert r.status_code == 200
    assert r.json()["use_stored_memory"] is False
    st = await client.get("/ui/api/memory/settings")
    assert st.json()["use_stored_memory"] is False
    await client.post("/ui/api/memory/settings", json={"use_stored_memory": True})


async def test_clear_session_memory_keeps_task(client):
    from ai_employee.api.deps import get_employee
    from ai_employee.memory import MemoryNamespace

    r = await client.post("/ui/api/tasks", json={"goal": "session memory probe"})
    sid = r.json()["session_id"]
    emp = get_employee()
    await emp.memory.record(
        "session scoped token qwe789",
        namespace=MemoryNamespace.long_term,
        session_id=sid,
    )
    clr = await client.delete(f"/ui/api/sessions/{sid}/memory")
    assert clr.status_code == 200
    assert clr.json()["records_removed"] >= 1
    hits = await emp.memory.recall("session scoped token", session_id=sid)
    assert not hits
    detail = await client.get(f"/ui/api/sessions/{sid}")
    assert detail.status_code == 200


async def test_delete_all_sessions(client):
    await client.post("/ui/api/tasks", json={"goal": "task one"})
    await client.post("/ui/api/tasks", json={"goal": "task two"})
    r = await client.delete("/ui/api/sessions")
    assert r.status_code == 200
    assert r.json()["deleted"] >= 2
    listed = (await client.get("/ui/api/sessions")).json()["sessions"]
    assert listed == []


async def test_delete_followup_run(client):
    r1 = await client.post("/ui/api/tasks", json={"goal": "main task"})
    sid = r1.json()["session_id"]
    first_id = r1.json()["task_id"]
    r2 = await client.post(f"/ui/api/sessions/{sid}/followup", json={"goal": "extra follow up"})
    follow_id = r2.json()["task_id"]
    listed = (await client.get("/ui/api/sessions")).json()["sessions"]
    sess = next(s for s in listed if s["id"] == sid)
    assert len(sess["runs"]) == 2

    dr = await client.delete(f"/ui/api/sessions/{sid}/runs/{follow_id}")
    assert dr.status_code == 200
    assert dr.json()["deleted"] == follow_id

    listed = (await client.get("/ui/api/sessions")).json()["sessions"]
    sess = next(s for s in listed if s["id"] == sid)
    assert len(sess["runs"]) == 1
    assert sess["runs"][0]["id"] == first_id

    bad = await client.delete(f"/ui/api/sessions/{sid}/runs/{first_id}")
    assert bad.status_code == 404


# ── Cancellation ───────────────────────────────────────────────────────────
class _SlowEmployee:
    """Fake employee whose run() blocks long enough to be cancelled mid-flight."""

    async def run(self, goal, *, session_id=None, approved_ops=None, resource_decision=None, **_):
        await asyncio.sleep(30)
        return {"answer": "done"}

    async def name_task(self, goal):
        return "Slow task"

    def persist_memory(self):
        pass


async def test_cancel_stops_running_task():
    from ai_employee.observability.tasks import TaskManager, TaskStatus

    tm = TaskManager(None)
    task = tm.start(_SlowEmployee(), "a long-running goal")

    await asyncio.sleep(0.05)  # let it enter the running state
    assert tm.get_task(task.id).status == TaskStatus.running

    assert tm.cancel(task.id) is True
    await asyncio.sleep(0.05)  # let the CancelledError land

    rec = tm.get_task(task.id)
    assert rec.status == TaskStatus.cancelled
    assert rec.error == "cancelled by user"
    assert rec.finished_at is not None

    # Already terminal → cancelling again is a no-op.
    assert tm.cancel(task.id) is False


async def test_cancel_unknown_task_404(client):
    r = await client.post("/ui/api/tasks/nope/cancel")
    assert r.status_code == 404


async def test_resume_endpoint_resumes_interrupted_task(monkeypatch):
    from ai_employee.api import ui as ui_module
    from ai_employee.api.app import create_app
    from ai_employee.api.deps import get_employee
    from ai_employee.observability.tasks import TaskManager, TaskRecord, _Session

    tm = TaskManager(None)
    rec = TaskRecord.from_snapshot(
        {
            "id": "resume-task-1",
            "goal": "resume interrupted task",
            "session_id": "sess-1",
            "status": "running",
            "events": [{"seq": 1, "ts": "t", "event": "approval.granted", "fields": {}}],
            "result": {
                "answer": "",
                "needs_approval": None,
                "workflow_checkpoint": {
                    "goal": "resume interrupted task",
                    "session_id": "sess-1",
                    "plan": {"steps": [{"id": 1}]},
                    "cursor": 0,
                    "step_results": [],
                },
            },
            "approved_ops": ["shell_write"],
        }
    )
    # Sessions are owner-scoped now — act as the guest who owns this one.
    from ai_employee.api.users import COOKIE_NAME, encode_cookie, mint_guest
    from ai_employee.config import get_settings

    guest = mint_guest()
    session = _Session("Resume interrupted task", user_id=guest.id, user_name=guest.name)
    session.id = "sess-1"
    session.task_ids = [rec.id]
    tm._sessions[session.id] = session
    tm._tasks[rec.id] = rec

    class _ResumeEmployee:
        async def run(
            self,
            goal,
            *,
            session_id=None,
            approved_ops=None,
            resource_decision=None,
            workflow_checkpoint=None,
            session_context=None,
        ):
            assert workflow_checkpoint is not None
            return {
                "answer": "resumed from api",
                "confidence": 0.9,
                "needs_approval": None,
                "workflow_checkpoint": workflow_checkpoint,
            }

        async def name_task(self, goal):
            return "Resume task"

        def persist_memory(self):
            pass

    monkeypatch.setattr(ui_module, "get_task_manager", lambda: tm)
    app = create_app()
    app.dependency_overrides[get_employee] = lambda: _ResumeEmployee()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.cookies.set(COOKIE_NAME, encode_cookie(get_settings(), guest))
        r = await client.post(f"/ui/api/tasks/{rec.id}/resume")
        assert r.status_code == 200
        snap = await _wait_done(client, rec.id)
        assert snap["status"] == "completed"
        assert snap["result"]["answer"] == "resumed from api"


# ── Approval: resume in place + resume-on-startup ──────────────────────────
class _ApprovalEmployee:
    """Pauses for approval on the first run, completes once the op is approved."""

    _PLAN = {
        "steps": [
            {
                "id": 1,
                "agent": "coding",
                "description": "run gated shell",
                "success_criteria": "done",
            },
        ],
    }

    def __init__(self):
        self.calls = 0
        self.last_checkpoint = None

    def _checkpoint(self, goal, session_id):
        return {
            "goal": goal,
            "session_id": session_id or "default",
            "plan": self._PLAN,
            "cursor": 0,
            "step_results": [],
        }

    async def run(
        self,
        goal,
        *,
        session_id=None,
        approved_ops=None,
        resource_decision=None,
        workflow_checkpoint=None,
        session_context=None,
    ):
        self.calls += 1
        self.last_checkpoint = workflow_checkpoint
        if not approved_ops:
            return {
                "answer": "",
                "confidence": 0.0,
                "needs_approval": {
                    "operation": "shell_write",
                    "tool": "shell",
                    "args": {"command": "rm -rf build"},
                },
                "workflow_checkpoint": self._checkpoint(goal, session_id),
            }
        if workflow_checkpoint:
            assert workflow_checkpoint.get("plan")
        return {
            "answer": "done",
            "confidence": 0.9,
            "needs_approval": None,
            "workflow_checkpoint": workflow_checkpoint,
        }

    async def name_task(self, goal):
        return "Approval task"

    def persist_memory(self):
        pass


async def _wait_status(tm, task_id, status, tries=200):
    from ai_employee.observability.tasks import TaskStatus

    for _ in range(tries):
        await asyncio.sleep(0.01)
        if tm.get_task(task_id).status == TaskStatus(status):
            return
    raise AssertionError(f"task never reached {status}")


async def test_approve_resumes_in_place_same_run():
    from ai_employee.observability.tasks import TaskManager, TaskStatus

    tm = TaskManager(None)
    emp = _ApprovalEmployee()
    task = tm.start(emp, "do a gated thing")
    await _wait_status(tm, task.id, "awaiting_approval")
    n_before = len(tm.get_task(task.id).events)

    same = tm.approve(emp, task.id)
    assert same is task  # SAME record — no new run/tab is created
    assert tm.get_task(task.id).status == TaskStatus.running
    assert tm.get_task(task.id).result.get("needs_approval") is None

    await _wait_status(tm, task.id, "completed")
    rec = tm.get_task(task.id)
    assert rec.status == TaskStatus.completed
    assert "shell_write" in rec.approved_ops
    # The approval marker is appended into the SAME run's event stream.
    assert any(e.event == "approval.granted" for e in rec.events)
    assert len(rec.events) > n_before
    assert emp.calls == 2
    assert emp.last_checkpoint is not None
    assert emp.last_checkpoint.get("plan")


async def test_auto_approve_never_pauses_for_gated_ops():
    """Auto Approve pre-grants every approval key so the run never waits."""
    from ai_employee.observability.tasks import TaskManager, TaskStatus

    tm = TaskManager(None)
    emp = _ApprovalEmployee()
    task = tm.start(emp, "do a gated thing", auto_approve=True)
    assert task.auto_approve is True
    assert "shell_write" in task.approved_ops
    assert "resource_download" in task.approved_ops

    await _wait_status(tm, task.id, "completed")
    rec = tm.get_task(task.id)
    assert rec.status == TaskStatus.completed
    assert rec.result.get("needs_approval") is None
    assert emp.calls == 1


async def test_ui_create_task_accepts_auto_approve(client):
    r = await client.post(
        "/ui/api/tasks",
        json={"goal": "auto approve smoke", "auto_approve": True},
    )
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    snap = await _wait_done(client, task_id)
    assert snap["auto_approve"] is True
    assert "shell_write" in snap["approved_ops"]


async def test_approve_after_restart_uses_saved_checkpoint(tmp_path):
    """Approve & Continue must resume from disk checkpoint after a process restart."""
    from ai_employee.observability.store import SessionStore
    from ai_employee.observability.tasks import TaskManager, TaskStatus

    store = SessionStore(tmp_path)
    emp = _ApprovalEmployee()
    tm1 = TaskManager(store)
    task = tm1.start(emp, "do a gated thing after restart")
    await _wait_status(tm1, task.id, "awaiting_approval")
    paused = tm1.get_task(task.id)
    assert paused.result and paused.result.get("workflow_checkpoint", {}).get("plan")

    # Simulate server restart — new TaskManager loads the same store.
    tm2 = TaskManager(SessionStore(tmp_path))
    reloaded = tm2.get_task(task.id)
    assert reloaded is not None
    assert reloaded.status == TaskStatus.awaiting_approval
    assert reloaded.result.get("workflow_checkpoint", {}).get("plan")

    emp2 = _ApprovalEmployee()
    same = tm2.approve(emp2, task.id)
    assert same is reloaded
    await _wait_status(tm2, task.id, "completed")
    rec = tm2.get_task(task.id)
    assert rec.status == TaskStatus.completed
    assert emp2.last_checkpoint is not None
    assert emp2.last_checkpoint.get("cursor") == 0
    assert emp2.calls == 1  # only the resume run in this process


async def test_approve_non_paused_returns_none():
    from ai_employee.observability.tasks import TaskManager

    assert TaskManager(None).approve(_ApprovalEmployee(), "unknown") is None


async def test_resume_interrupted_relaunches_pending():
    from ai_employee.observability.tasks import TaskManager, TaskRecord, TaskStatus

    tm = TaskManager(None)
    # A run persisted mid-flight (pending, no result) with a prior approval.
    rec = TaskRecord.from_snapshot(
        {
            "id": "abc123def456",
            "goal": "resume me",
            "session_id": "s1",
            "status": "pending",
            "events": [],
            "result": None,
            "approved_ops": ["shell_write"],
        }
    )
    assert rec.status == TaskStatus.pending and rec.resumable is True
    tm._tasks[rec.id] = rec

    resumed = tm.resume_interrupted(_ApprovalEmployee())
    assert resumed == ["abc123def456"]
    await _wait_status(tm, "abc123def456", "completed")
    # approved_ops survived the restart → it finishes without re-pausing.
    assert tm.get_task("abc123def456").status == TaskStatus.completed


async def test_cancel_finished_task_is_noop(client):
    start = (await client.post("/ui/api/tasks", json={"goal": "quick goal"})).json()
    await _wait_done(client, start["task_id"])
    r = await client.post(f"/ui/api/tasks/{start['task_id']}/cancel")
    assert r.status_code == 200
    assert r.json()["cancelled"] is False
