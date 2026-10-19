"""Task <-> source links, reference-counted deletion, and isolation.

Covers the behaviour the design asks for:

* every task is linked to the sources it owns/uses, and a source knows every task
  linked to it (both directions);
* deleting a task keeps a source while any *other* task still links it, and
  deletes it only when the removed task was its last link;
* "isolate" makes every shared source task-private (a real copy, re-downloading
  from the recorded URL when the shared bytes are gone), leaving no shared source.
"""

from __future__ import annotations

import pathlib

from ai_employee.resources import ResourceManager


def _mgr(tmp_path: pathlib.Path, name: str = "ws") -> tuple[ResourceManager, pathlib.Path]:
    ws = tmp_path / name
    ws.mkdir()
    return ResourceManager(ws, audit_file=tmp_path / "audit.jsonl"), ws


def _task_keys(mgr: ResourceManager, resource_id: str) -> set[str]:
    return {t["task_key"] or t["task_id"] for t in mgr.tasks_for(resource_id)}


# ── bidirectional links ──────────────────────────────────────────────────────
def test_link_is_bidirectional(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "s.bin"
    src.write_bytes(b"shared")
    rec = mgr.put(
        src, name="s.bin", kind="dataset", visibility="shared", task="task-a", session_id="task-a"
    )
    mgr.link(rec.id, task_key="task-b", session_id="task-b", role="consumer")

    # source -> tasks
    assert _task_keys(mgr, rec.id) == {"task-a", "task-b"}
    assert mgr.reference_count(rec.id) == 2
    # task -> sources
    assert rec.id in {r.id for r in mgr.resources_for("task-b")}
    # idempotent
    mgr.link(rec.id, task_key="task-b", role="consumer")
    assert mgr.reference_count(rec.id) == 2


def test_private_resource_owns_one_link(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "p.bin"
    src.write_bytes(b"priv")
    rec = mgr.put(src, name="p.bin", kind="file", visibility="private", task="alice")
    tasks = mgr.tasks_for(rec.id)
    assert len(tasks) == 1
    assert tasks[0]["role"] == "owner" and tasks[0]["task_key"] == "alice"


# ── reference-counted deletion ───────────────────────────────────────────────
def test_delete_keeps_source_until_last_link(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "m.pt"
    src.write_bytes(b"weights")
    rec = mgr.put(src, name="m.pt", kind="model", visibility="shared", task="t1", session_id="t1")
    mgr.link(rec.id, task_key="t2", session_id="t2", role="consumer")

    first = mgr.delete_for_task("t1")
    assert rec.id in first["kept"] and rec.id not in first["deleted"]
    assert mgr.get(rec.id) is not None
    assert _task_keys(mgr, rec.id) == {"t2"}

    second = mgr.delete_for_task("t2")
    assert rec.id in second["deleted"]
    assert mgr.get(rec.id) is None


def test_delete_one_private_task_leaves_the_other(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    a = ws / "a.bin"
    a.write_bytes(b"a")
    b = ws / "b.bin"
    b.write_bytes(b"b")
    ra = mgr.put(a, name="a.bin", kind="file", visibility="private", task="alice")
    rb = mgr.put(b, name="b.bin", kind="file", visibility="private", task="bob")

    report = mgr.delete_for_task("alice")
    assert ra.id in report["deleted"]
    assert mgr.get(ra.id) is None
    assert mgr.get(rb.id) is not None


def test_delete_for_session_removes_all_run_links(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "d.bin"
    src.write_bytes(b"d")
    rec = mgr.put(
        src, name="d.bin", kind="dataset", visibility="shared", task="sess-1", session_id="sess-1"
    )
    # A follow-up run in the same session links to the same source.
    mgr.link(rec.id, task_id="run-2", task_key="sess-1", session_id="sess-1", role="consumer")
    mgr.link(rec.id, task_key="other", session_id="other", role="consumer")

    report = mgr.delete_for_session("sess-1")
    assert rec.id in report["kept"]
    assert _task_keys(mgr, rec.id) == {"other"}
    assert mgr.get(rec.id) is not None


# ── isolation ────────────────────────────────────────────────────────────────
def test_isolate_task_makes_a_real_private_copy(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "shared_ds"
    src.mkdir()
    (src / "x.txt").write_text("payload")
    rec = mgr.put(
        src, name="shared_ds", kind="dataset", visibility="shared", task="t1", session_id="t1"
    )
    mgr.link(rec.id, task_key="t2", session_id="t2", role="consumer")
    shared_path = mgr.materialized_path(rec)
    shared_ino = (shared_path / "x.txt").stat().st_ino

    result = mgr.isolate_task("t1")
    assert result["isolated"], result
    priv = mgr.get(result["isolated"][0]["private"])
    assert priv is not None and priv.visibility == "private" and priv.owner_task == "t1"
    priv_path = mgr.materialized_path(priv)
    assert (priv_path / "x.txt").read_text() == "payload"
    # A genuine copy, not a hard-link of the shared source.
    assert (priv_path / "x.txt").stat().st_ino != shared_ino

    # t1's link left the shared record; t2's remains.
    assert _task_keys(mgr, rec.id) == {"t2"}


def test_isolate_all_leaves_no_shared_source(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "d.bin"
    src.write_bytes(b"d")
    rec = mgr.put(
        src, name="d.bin", kind="dataset", visibility="shared", task="t1", session_id="t1"
    )
    mgr.link(rec.id, task_key="t2", session_id="t2", role="consumer")

    result = mgr.isolate_all(copy=True, purge_orphans=True)
    assert rec.id in result["purged"]
    assert mgr.get(rec.id) is None
    owners = {r.owner_task for r in mgr.records().values() if r.visibility == "private"}
    assert owners == {"t1", "t2"}
    assert not [r for r in mgr.records().values() if r.is_shared]


async def test_isolate_redownloads_missing_bytes(tmp_path: pathlib.Path) -> None:
    from ai_employee.resources.isolation import isolate_task

    mgr, ws = _mgr(tmp_path)
    src = ws / "r.bin"
    src.write_bytes(b"original")
    rec = mgr.put(
        src,
        name="r.bin",
        kind="file",
        visibility="shared",
        task="t1",
        session_id="t1",
        url="https://example.com/r.bin",
    )
    mgr.materialized_path(rec).unlink()  # the shared bytes are gone
    assert mgr.exists(rec) is False

    calls: list[str] = []

    async def fake_fetch(url: str, dest: pathlib.Path) -> bool:
        calls.append(url)
        dest.write_bytes(b"from-network")  # noqa: ASYNC240 - test double
        return True

    result = await isolate_task(
        mgr, "t1", fetcher=fake_fetch, allow_redownload=True, purge_orphans=True
    )
    assert result["isolated"], result
    assert calls == ["https://example.com/r.bin"]
    priv = [r for r in mgr.records().values() if r.visibility == "private"]
    assert priv and mgr.materialized_path(priv[0]).read_bytes() == b"from-network"


async def test_isolate_reports_when_redownload_disabled(tmp_path: pathlib.Path) -> None:
    from ai_employee.resources.isolation import isolate_task

    mgr, ws = _mgr(tmp_path)
    src = ws / "r.bin"
    src.write_bytes(b"original")
    rec = mgr.put(
        src,
        name="r.bin",
        kind="file",
        visibility="shared",
        task="t1",
        session_id="t1",
        url="https://example.com/r.bin",
    )
    mgr.materialized_path(rec).unlink()

    result = await isolate_task(mgr, "t1", allow_redownload=False, fetcher=None)
    assert result["isolated"] == []
    assert result["needs_redownload"]  # reported, bytes intact elsewhere
    assert mgr.get(rec.id) is not None  # shared source not lost


# ── facade methods ───────────────────────────────────────────────────────────
async def test_employee_reference_counted_delete(employee) -> None:
    ws = pathlib.Path(employee.settings.workspace_dir).resolve()  # noqa: ASYNC240 - test setup
    ws.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test setup
    mgr = ResourceManager(ws, audit_file=employee.settings.audit_log_file)
    f = ws / "emp_shared.bin"
    f.write_bytes(b"emp")
    rec = mgr.put(
        f, name="emp_shared.bin", kind="file", visibility="shared", task="s1", session_id="s1"
    )
    mgr.link(rec.id, task_key="s2", session_id="s2", role="consumer")

    first = employee.delete_task_resources(session_id="s1")
    assert rec.id in first["kept"]
    second = employee.delete_task_resources(session_id="s2")
    assert rec.id in second["deleted"]


async def test_employee_isolate_and_overview(employee) -> None:
    ws = pathlib.Path(employee.settings.workspace_dir).resolve()  # noqa: ASYNC240 - test setup
    ws.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test setup
    mgr = ResourceManager(ws, audit_file=employee.settings.audit_log_file)
    f = ws / "emp_ds.bin"
    f.write_bytes(b"emp-ds")
    rec = mgr.put(
        f, name="emp_ds.bin", kind="dataset", visibility="shared", task="e1", session_id="e1"
    )

    overview = employee.resources_overview()
    assert any(r["id"] == rec.id for r in overview["resources"])

    report = await employee.isolate_resources(all=True, allow_redownload=False)
    assert report["isolated"]
    assert mgr.get(rec.id) is None  # shared source purged after isolation


# ── API + CLI + UI ───────────────────────────────────────────────────────────
async def test_api_resources_list_and_isolate(client) -> None:
    from ai_employee.config import get_settings

    s = get_settings()
    ws = pathlib.Path(s.workspace_dir).resolve()  # noqa: ASYNC240 - test setup
    ws.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test setup
    mgr = ResourceManager(ws, audit_file=s.audit_log_file)
    f = ws / "api_shared.bin"
    f.write_bytes(b"api-bytes")
    rec = mgr.put(
        f, name="api_shared.bin", kind="file", visibility="shared", task="t1", session_id="t1"
    )
    mgr.link(rec.id, task_key="t2", session_id="t2", role="consumer")

    resp = await client.get("/ui/api/resources")
    assert resp.status_code == 200
    data = resp.json()
    item = next((r for r in data["resources"] if r["id"] == rec.id), None)
    assert item is not None and item["task_count"] == 2

    resp = await client.post(
        "/ui/api/resources/isolate", json={"all": True, "allow_redownload": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["isolated"]
    assert mgr.get(rec.id) is None


class _CLIStubEmployee:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def aclose(self) -> None:
        pass


async def test_cli_resources_links_and_isolate(tmp_path, capsys, monkeypatch) -> None:
    from argparse import Namespace

    from ai_employee.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "ws"
    settings.workspace_dir.mkdir(parents=True)
    settings.audit_log_file = tmp_path / "audit.jsonl"

    ws = settings.workspace_dir
    f = ws / "cli.bin"
    f.write_bytes(b"cli")
    mgr = ResourceManager(ws, audit_file=settings.audit_log_file)
    rec = mgr.put(f, name="cli.bin", kind="file", visibility="shared", task="t1", session_id="t1")
    mgr.link(rec.id, task_key="t2", session_id="t2", role="consumer")

    import ai_employee.core.employee as employee_module
    from ai_employee.cli import _run_async

    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _CLIStubEmployee(settings))

    code = await _run_async(Namespace(command="resources", resources_command="links", task=None))
    out = capsys.readouterr().out
    assert code == 0
    assert "Task <-> source" in out
    assert "t1" in out and "t2" in out

    code = await _run_async(
        Namespace(
            command="resources",
            resources_command="isolate",
            all=False,
            task="t1",
            no_redownload=True,
        )
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Isolation report" in out


def test_ui_exposes_resources_badge_and_isolate() -> None:
    static = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "ai_employee" / "api" / "static"
    )  # noqa: ASYNC240 - test setup
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    assert 'id="resources-badge"' in html
    assert "openResourcesModal" in js
    assert "/ui/api/resources/isolate" in js
    assert "/ui/api/resources/purge" in js
    assert "res-purge-all" in js
    assert "res-wipe-all" in js
    assert "wipe_workspace" in js


# ── delete all resources (fresh start) ───────────────────────────────────────
def test_purge_all_removes_records_and_bytes(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    a = ws / "a.bin"
    a.write_bytes(b"a")
    b = ws / "b.bin"
    b.write_bytes(b"b")
    mgr.put(a, name="a.bin", kind="file", visibility="shared")
    mgr.put(b, name="b.bin", kind="file", visibility="private", task="t1")
    assert len(mgr.records()) == 2

    report = mgr.purge_all(purge_bytes=True)
    assert report["deleted"] == 2
    assert mgr.records() == {}
    assert not (ws / ".resources" / "objects").exists()
    assert not (ws / "shared").exists()
    assert not (ws / "tasks").exists()


def test_purge_all_keep_bytes(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    a = ws / "keep.bin"
    a.write_bytes(b"keep")
    rec = mgr.put(a, name="keep.bin", kind="file", visibility="shared")
    materialized = mgr.materialized_path(rec)

    report = mgr.purge_all(purge_bytes=False)
    assert report["deleted"] == 1
    assert mgr.records() == {}
    assert materialized.exists()  # files deliberately left on disk


async def test_api_resources_purge(client) -> None:
    from ai_employee.config import get_settings

    s = get_settings()
    ws = pathlib.Path(s.workspace_dir).resolve()  # noqa: ASYNC240 - test setup
    ws.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test setup
    mgr = ResourceManager(ws, audit_file=s.audit_log_file)
    f = ws / "purge_me.bin"
    f.write_bytes(b"purge")
    mgr.put(f, name="purge_me.bin", kind="file", visibility="shared", task="t1", session_id="t1")

    resp = await client.post("/ui/api/resources/purge", json={"purge_bytes": True})
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 1
    assert (await client.get("/ui/api/resources")).json()["count"] == 0


async def test_cli_resources_purge(tmp_path, capsys, monkeypatch) -> None:
    from argparse import Namespace

    from ai_employee.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "ws"
    settings.workspace_dir.mkdir(parents=True)
    settings.audit_log_file = tmp_path / "audit.jsonl"

    ws = settings.workspace_dir
    f = ws / "cli.bin"
    f.write_bytes(b"cli")
    ResourceManager(ws, audit_file=settings.audit_log_file).put(
        f, name="cli.bin", kind="file", visibility="shared"
    )

    import ai_employee.core.employee as employee_module
    from ai_employee.cli import _run_async

    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _CLIStubEmployee(settings))
    code = await _run_async(
        Namespace(command="resources", resources_command="purge", keep_bytes=False)
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "purged" in out.lower()
    assert "deleted" in out.lower()


def test_wipe_workspace_clears_everything(tmp_path: pathlib.Path) -> None:
    mgr, ws = _mgr(tmp_path)
    (ws / "models").mkdir()
    (ws / "models" / "m.pt").write_bytes(b"m" * 10)
    (ws / "data").mkdir()
    (ws / "data" / "d.csv").write_text("x")
    (ws / "stray.bin").write_bytes(b"stray")
    assert any(ws.iterdir())

    report = mgr.wipe_workspace()
    assert report["removed_count"] >= 3
    assert report["bytes_freed"] > 0
    assert list(ws.iterdir()) == []  # workspace is empty (dir kept)


async def test_cli_resources_purge_workspace(tmp_path, capsys, monkeypatch) -> None:
    from argparse import Namespace

    from ai_employee.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "ws"
    settings.workspace_dir.mkdir(parents=True)
    settings.audit_log_file = tmp_path / "audit.jsonl"
    (settings.workspace_dir / "junk").mkdir()
    (settings.workspace_dir / "junk" / "big.bin").write_bytes(b"x" * 100)

    import ai_employee.core.employee as employee_module
    from ai_employee.cli import _run_async

    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _CLIStubEmployee(settings))
    code = await _run_async(
        Namespace(command="resources", resources_command="purge", keep_bytes=False, workspace=True)
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "workspace" in out.lower()
    assert not (settings.workspace_dir / "junk").exists()
