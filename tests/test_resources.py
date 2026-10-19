"""Tests for the secure, task-isolated resource management subsystem.

Covers the eleven requirements: private/shared resources, content dedup, task &
workspace isolation, authorization, date-based organization, safe names,
versioning & provenance, concurrency, migration, audit logging, and the agent
tools + CLI that expose it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ai_employee.resources import ResourceManager, Visibility, migrate_workspace
from ai_employee.resources.errors import (
    ResourceAccessDenied,
    ResourcePathError,
)
from ai_employee.resources.hashing import hash_file, hash_path, normalize_hash
from ai_employee.resources.names import (
    contained,
    safe_resource_name,
    safe_slug,
    task_date_key,
)


def _mgr(tmp_path: Path, name: str = "ws") -> tuple[ResourceManager, Path]:
    ws = tmp_path / name
    ws.mkdir()
    return ResourceManager(ws, audit_file=tmp_path / "audit.jsonl"), ws


# ── safe names & isolation ───────────────────────────────────────────────────
def test_safe_slug_strips_traversal() -> None:
    assert safe_slug("../../etc/passwd") == "etc-passwd"
    assert safe_slug("ÜnÍcode Name!") == "n-code-name"
    assert safe_slug("") == "item"
    assert safe_slug("...") == "item"


def test_safe_resource_name_keeps_one_extension() -> None:
    assert safe_resource_name("yolo26n.pt") == "yolo26n.pt"
    assert safe_resource_name("a/b/c.tar.gz") == "c-tar.gz"
    assert safe_resource_name("../../evil.sh") == "evil.sh"


def test_contained_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ResourcePathError):
        contained(root, "..", "outside")
    # A symlink pointing out of the root is refused too.
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(ResourcePathError):
        contained(root, "link", "x")


def test_task_identity_is_stable_and_date_is_separate(tmp_path: Path) -> None:
    mgr, _ = _mgr(tmp_path)
    assert mgr.task_identity("task-one") == "task-one"
    root_today = mgr.task_root("task-one")
    assert root_today.parent.parent.parent.name == "task-one"
    assert root_today.parent.parent.parent.parent.name == "tasks"
    assert root_today.name.isdigit() and root_today.parent.name.isdigit()  # .../YYYY/MM/DD
    assert task_date_key().count("/") == 2


# ── private vs shared + authorization ────────────────────────────────────────
def test_private_and_shared_visibility(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    priv_src = ws / "src1.bin"
    priv_src.write_bytes(b"private-bytes")
    shared_src = ws / "src2.bin"
    shared_src.write_bytes(b"shared-bytes")

    priv = mgr.put(priv_src, name="secret.bin", kind="file", visibility="private", task="alice")
    shared = mgr.put(shared_src, name="common.bin", kind="file", visibility="shared")

    assert priv.visibility == Visibility.private.value
    assert priv.owner_task == "alice"
    assert priv.path.startswith("tasks/alice/")
    assert shared.visibility == Visibility.shared.value
    assert shared.owner_task is None
    assert shared.path.startswith("shared/file/")

    # alice sees her private resource + the shared one.
    names = {r.name for r in mgr.list(task="alice")}
    assert names == {"secret.bin", "common.bin"}
    # bob sees only the shared one.
    names_bob = {r.name for r in mgr.list(task="bob")}
    assert names_bob == {"common.bin"}


def test_authorization_matrix(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "x.bin"
    src.write_bytes(b"x")
    priv = mgr.put(src, name="x.bin", kind="file", visibility="private", task="alice")

    mgr.authorize(priv, task="alice", mode="read")  # owner reads
    mgr.authorize(priv, task="alice", mode="write")
    with pytest.raises(ResourceAccessDenied):
        mgr.authorize(priv, task="bob", mode="read")
    with pytest.raises(ResourceAccessDenied):
        mgr.authorize(priv, task="bob", mode="delete")


def test_resolve_denies_cross_task_read(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "secret.bin"
    src.write_bytes(b"top-secret")
    priv = mgr.put(src, name="secret.bin", kind="file", visibility="private", task="alice")

    with pytest.raises(ResourceAccessDenied):
        mgr.resolve(priv.name, task="bob", mode="read")
    # The owner and a system actor (task=None) can resolve it.
    assert mgr.resolve(priv.name, task="alice").id == priv.id
    assert mgr.resolve(priv.id, task=None).id == priv.id


def test_delete_and_promote_are_owner_only(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "a.bin"
    src.write_bytes(b"a")
    priv = mgr.put(src, name="a.bin", kind="file", visibility="private", task="alice")

    with pytest.raises(ResourceAccessDenied):
        mgr.promote(priv.id, task="bob")
    with pytest.raises(ResourceAccessDenied):
        mgr.delete(priv.id, task="bob")

    shared = mgr.promote(priv.id, task="alice")
    assert shared.is_shared
    assert shared.path.startswith("shared/")


# ── content-based deduplication ──────────────────────────────────────────────
def test_content_dedup_shares_one_object(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    a = ws / "a.bin"
    a.write_bytes(b"identical-content")
    b = ws / "b.bin"
    b.write_bytes(b"identical-content")

    r1 = mgr.put(a, name="one.bin", kind="file", visibility="private", task="t1")
    r2 = mgr.put(b, name="two.bin", kind="file", visibility="shared")

    assert normalize_hash(r1.hash) == normalize_hash(r2.hash)
    assert r2.provenance.dedup_of == r1.id
    objects = [p for p in (ws / ".resources" / "objects").rglob("*") if p.is_file()]
    assert len(objects) == 1, "identical content must not be stored twice"
    assert mgr.find_by_hash(r1.hash).id == r1.id


def test_hash_file_and_dir(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    digest, size = hash_file(f)
    assert size == 3
    assert digest == hash_path(f)["hash"].split(":", 1)[1]

    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "x.txt").write_text("hello")
    ident = hash_path(d)
    assert ident["kind"] == "content"
    assert ident["files"] == 1
    # A changed file changes the directory identity.
    (d / "sub" / "x.txt").write_text("world")
    assert hash_path(d)["hash"] != ident["hash"]


# ── versioning + provenance ──────────────────────────────────────────────────
def test_versioning_and_provenance(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "model.pt"
    src.write_bytes(b"v1-weights")
    r1 = mgr.put(
        src, name="model.pt", kind="model", visibility="shared", source="http", url="https://x/m.pt"
    )
    assert r1.version == 1
    assert r1.provenance.url == "https://x/m.pt"
    assert r1.provenance.source == "http"

    src.write_bytes(b"v2-weights")
    r2 = mgr.put(src, name="model.pt", kind="model", visibility="shared", source="http")
    assert r2.id == r1.id
    assert r2.version == 2
    assert len(r2.versions) == 2
    hashes = {v.hash for v in r2.versions}
    assert len(hashes) == 2


def test_register_does_not_move_bytes(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    legacy = ws / "models" / "x.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"weights")
    rec = mgr.register(legacy, visibility="shared", source="migration", tool="migration")
    assert legacy.exists()  # register only indexes
    assert rec.path == "models/x.pt"


# ── concurrency protection ───────────────────────────────────────────────────
def test_concurrent_puts_do_not_corrupt_catalog(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    n = 12
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            p = ws / f"f{i}.bin"
            p.write_bytes(f"payload-{i}".encode())
            mgr.put(p, name=f"r{i}.bin", kind="file", visibility="shared")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    records = mgr.records()
    assert len(records) == n
    # The catalog is valid JSON after concurrent writes.
    from ai_employee.resources.catalog import SCHEMA_VERSION

    raw = json.loads((ws / ".resources" / "catalog.json").read_text())
    assert raw["version"] == SCHEMA_VERSION
    assert len(raw["resources"]) == n


def test_catalog_tolerates_corrupt_file(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    cat = ws / ".resources" / "catalog.json"
    cat.parent.mkdir(parents=True, exist_ok=True)
    cat.write_text("{not valid json")
    assert mgr.records() == {}  # falls back to empty rather than raising


# ── verify ───────────────────────────────────────────────────────────────────
def test_verify_detects_deleted_and_changed_bytes(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    a = ws / "a.bin"
    a.write_bytes(b"a")
    b = ws / "b.bin"
    b.write_bytes(b"b")
    ra = mgr.put(a, name="a.bin", kind="file", visibility="shared")
    rb = mgr.put(b, name="b.bin", kind="file", visibility="shared")
    assert ra.id  # registered

    assert mgr.verify()["ok"] is True

    mgr.materialized_path(rb).unlink()
    report = mgr.verify()
    assert report["ok"] is False
    assert any(m["id"] == rb.id for m in report["missing"])

    # Recreate but change the content → mismatch.
    p = mgr.materialized_path(rb)
    p.write_bytes(b"tampered")
    report = mgr.verify()
    assert any(m["id"] == rb.id for m in report["mismatched"])


# ── audit logging ────────────────────────────────────────────────────────────
def test_audit_records_events(tmp_path: Path) -> None:
    mgr, ws = _mgr(tmp_path)
    src = ws / "a.bin"
    src.write_bytes(b"a")
    rec = mgr.put(src, name="a.bin", kind="file", visibility="private", task="alice")
    with pytest.raises(ResourceAccessDenied):
        mgr.authorize(rec, task="bob", mode="read")

    lines = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    actions = {e["action"] for e in lines}
    assert "create" in actions
    assert "access_denied" in actions
    assert all(e["kind"] == "resource" for e in lines)


# ── migration of existing data ───────────────────────────────────────────────
def test_migration_indexes_legacy_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "models").mkdir(parents=True)
    (ws / "data").mkdir()
    (ws / "sessions" / "s1").mkdir(parents=True)
    (ws / "models" / "yolo.pt").write_bytes(b"weights")
    (ws / "data" / "pets.csv").write_text("a,b\n1,2\n")
    (ws / "sessions" / "s1" / "clip.mp4").write_bytes(b"video")
    # A legacy asset ledger that also points at a stray file.
    (ws / "stray.bin").write_bytes(b"stray")
    (ws / "assets.json").write_text(
        json.dumps(
            {
                "version": 1,
                "assets": [
                    {
                        "path": "stray.bin",
                        "kind": "file",
                        "source": "http",
                        "url": "https://x/stray",
                        "session_id": "s1",
                    }
                ],
            }
        )
    )

    mgr = ResourceManager(ws, audit_file=tmp_path / "audit.jsonl")
    dry = migrate_workspace(mgr, apply=False)
    assert dry["applied"] is False
    assert dry["registered"] == 0
    assert dry["candidates"] >= 4
    assert mgr.records() == {}

    report = migrate_workspace(mgr, apply=True)
    assert report["registered"] >= 4
    records = mgr.records()
    by_name = {r.name: r for r in records.values()}
    assert by_name["yolo.pt"].visibility == "shared"
    assert by_name["pets.csv"].visibility == "shared"
    assert by_name["clip.mp4"].visibility == "private"
    assert by_name["clip.mp4"].owner_task == "s1"
    assert by_name["stray.bin"].source == "http"

    # Re-running is a no-op.
    again = migrate_workspace(mgr, apply=True)
    assert again["registered"] == 0
    assert again["skipped"] == again["candidates"]


# ── agent tools ──────────────────────────────────────────────────────────────
@pytest.fixture
def resource_settings(tmp_path: Path):
    from ai_employee.config import get_settings

    s = get_settings().model_copy(deep=True)
    s.workspace_dir = tmp_path / "ws"
    s.workspace_dir.mkdir(parents=True, exist_ok=True)
    s.audit_log_file = tmp_path / "audit.jsonl"
    return s


async def test_resource_tools_list_get_promote(resource_settings) -> None:
    from ai_employee.observability.resource_ctx import (
        bind_resource_session,
        clear_resource_decision,
    )
    from ai_employee.tools.builtins.resource_manager import (
        GetResourceTool,
        ListResourcesTool,
    )

    ws = resource_settings.workspace_dir
    src = ws / "ds.bin"
    src.write_bytes(b"dataset-bytes")
    mgr = ResourceManager(ws, audit_file=resource_settings.audit_log_file)
    rec = mgr.put(src, name="ds.bin", kind="dataset", visibility="private", task="alice")

    bind_resource_session("alice")
    try:
        listed = await ListResourcesTool(resource_settings).execute(query="ds")
        assert listed.ok
        assert any(r["id"] == rec.id for r in listed.output["resources"])

        got = await GetResourceTool(resource_settings).execute(ref="ds.bin")
        assert got.ok
        assert got.output["resource"]["id"] == rec.id

        from ai_employee.core.exceptions import ApprovalRequired
        from ai_employee.tools import default_registry

        reg = default_registry(resource_settings)
        with pytest.raises(ApprovalRequired):
            await reg.execute("promote_resource", {"resource_id": rec.id})
        promoted = await reg.execute("promote_resource", {"resource_id": rec.id}, approved=True)
        assert promoted.ok
        assert promoted.output["resource"]["visibility"] == "shared"
    finally:
        clear_resource_decision()


async def test_resource_tools_deny_cross_task(resource_settings) -> None:
    from ai_employee.observability.resource_ctx import (
        bind_resource_session,
        clear_resource_decision,
    )
    from ai_employee.tools.builtins.resource_manager import GetResourceTool

    ws = resource_settings.workspace_dir
    src = ws / "secret.bin"
    src.write_bytes(b"secret")
    ResourceManager(ws, audit_file=resource_settings.audit_log_file).put(
        src, name="secret.bin", kind="file", visibility="private", task="alice"
    )

    bind_resource_session("bob")  # a different task
    try:
        out = await GetResourceTool(resource_settings).execute(ref="secret.bin")
        assert out.ok is False
        assert "not authorized" in (out.error or "").lower()
    finally:
        clear_resource_decision()


def test_resource_tools_registered(resource_settings) -> None:
    from ai_employee.tools import default_registry

    reg = default_registry(resource_settings)
    for name in ("list_resources", "get_resource", "promote_resource"):
        assert name in reg.names()


# ── CLI ──────────────────────────────────────────────────────────────────────
class _CLIStubEmployee:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def aclose(self) -> None:
        pass


async def test_cli_resources_list_and_migrate(tmp_path, capsys, monkeypatch) -> None:
    from argparse import Namespace

    from ai_employee.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "ws"
    settings.workspace_dir.mkdir(parents=True)
    settings.audit_log_file = tmp_path / "audit.jsonl"

    import ai_employee.core.employee as employee_module
    from ai_employee.cli import _run_async

    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _CLIStubEmployee(settings))

    (settings.workspace_dir / "models").mkdir()
    (settings.workspace_dir / "models" / "m.pt").write_bytes(b"w")

    # migrate --apply
    code = await _run_async(Namespace(command="resources", resources_command="migrate", apply=True))
    out = capsys.readouterr().out
    assert code == 0
    assert "migrate" in out.lower() or "registered" in out.lower()

    # list shows the migrated resource
    code = await _run_async(
        Namespace(
            command="resources", resources_command="list", task=None, shared=True, kind="", query=""
        )
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "m.pt" in out


async def test_cli_resources_verify(tmp_path, capsys, monkeypatch) -> None:
    from argparse import Namespace

    from ai_employee.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "ws"
    settings.workspace_dir.mkdir(parents=True)
    settings.audit_log_file = tmp_path / "audit.jsonl"

    ws = settings.workspace_dir
    f = ws / "ok.bin"
    f.write_bytes(b"ok")
    ResourceManager(ws, audit_file=settings.audit_log_file).put(
        f, name="ok.bin", kind="file", visibility="shared"
    )

    import ai_employee.core.employee as employee_module
    from ai_employee.cli import _run_async

    monkeypatch.setattr(employee_module, "AIEmployee", lambda: _CLIStubEmployee(settings))
    code = await _run_async(Namespace(command="resources", resources_command="verify", task=None))
    out = capsys.readouterr().out
    assert code == 0
    assert "ok" in out.lower()


# ── DB mirror mapping ────────────────────────────────────────────────────────
def test_mirror_row_mapping(tmp_path: Path) -> None:
    from ai_employee.resources.mirror import row_from_record

    mgr, ws = _mgr(tmp_path)
    src = ws / "m.pt"
    src.write_bytes(b"weights")
    rec = mgr.put(src, name="m.pt", kind="model", visibility="shared", source="http")
    row = row_from_record(rec)
    assert row["content_hash"] == rec.hash
    assert row["visibility"] == "shared"
    assert row["logical_path"] == rec.path
    assert "provenance" in row["meta"]
