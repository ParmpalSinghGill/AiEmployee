"""The resource manager: private/shared storage, dedup, versions, authorization.

Layout under the workspace root:

    .resources/
      catalog.json          # the index (atomic, locked)
      objects/<aa>/<hash>   # content-addressed store — one copy per content hash
    shared/<kind>/<name>    # shared, reusable resources (any task may read)
    tasks/<YYYY>/<MM>/<DD>/<slug>/       # date-based private task area
      resources/<name>      # private resources (only the owning task may read)
      manifest.json         # the task's resource manifest

The five concepts stay separate: task identity (date-based key), resource identity
(content hash), ownership (``owner_task`` / ``owner_user``), visibility
(``private`` / ``shared``) and physical storage (object store vs logical path).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger
from ai_employee.observability.asset_registry import classify_asset
from ai_employee.resources.audit import record_resource_event
from ai_employee.resources.catalog import Catalog
from ai_employee.resources.errors import (
    ResourceAccessDenied,
    ResourceNotFound,
    ResourcePathError,
)
from ai_employee.resources.hashing import hash_path, normalize_hash
from ai_employee.resources.models import (
    Provenance,
    ResourceLink,
    ResourceRecord,
    ResourceVersion,
    Visibility,
    now_iso,
)
from ai_employee.resources.names import (
    contained,
    relative_to,
    safe_resource_name,
    safe_task_identity,
    task_date_parts,
    validate_relative,
)

log = get_logger("resources")

META_DIR = ".resources"
OBJECTS_DIR = "objects"
CATALOG_FILE = "catalog.json"
SHARED_DIR = "shared"
TASKS_DIR = "tasks"


class ResourceManager:
    """Owns the resource catalog and the bytes behind it.

    Filesystem-first so the whole framework keeps working offline; the optional
    SQLAlchemy mirror (``ai_employee.resources.mirror``) lets a deployed app also
    query resources from Postgres without changing this class.
    """

    def __init__(self, workspace: str | Path, *, audit_file: str | Path | None = None) -> None:
        self._ws = Path(workspace).expanduser().resolve()
        self._ws.mkdir(parents=True, exist_ok=True)
        self._meta = self._ws / META_DIR
        self._objects = self._meta / OBJECTS_DIR
        self._catalog = Catalog(self._meta / CATALOG_FILE)
        self._audit_file = Path(audit_file) if audit_file else None

    # ── paths ────────────────────────────────────────────────────────────────
    @property
    def workspace(self) -> Path:
        return self._ws

    def task_identity(self, task_id: str | None) -> str:
        """Stable task identity used for ownership (does not change with date)."""
        return safe_task_identity(task_id)

    def task_identity_root(self, task: str) -> Path:
        rel = validate_relative(task)
        return contained(self._ws, TASKS_DIR, *rel.parts)

    def task_root(self, task: str, *, when: Any = None) -> Path:
        """Date-based storage root: ``tasks/<slug>/<YYYY>/<MM>/<DD>``."""
        year, month, day = task_date_parts(when)
        return contained(self.task_identity_root(task), year, month, day)

    def private_root(self, task: str, *, when: Any = None) -> Path:
        return contained(self.task_root(task, when=when), "resources")

    def shared_root(self) -> Path:
        return contained(self._ws, SHARED_DIR)

    def object_path(self, object_key: str) -> Path:
        rel = validate_relative(object_key)
        return contained(self._objects, *rel.parts)

    # ── attribution helpers ──────────────────────────────────────────────────
    def _audit(self, action: str, *, record: ResourceRecord | None = None, **kw: Any) -> None:
        record_resource_event(
            self._audit_file,
            action=action,
            resource_id=record.id if record else kw.get("resource_id", ""),
            name=record.name if record else kw.get("name", ""),
            visibility=record.visibility if record else kw.get("visibility", ""),
            task=kw.get("task"),
            actor=kw.get("actor"),
            ok=kw.get("ok", True),
            detail=kw.get("detail", ""),
        )

    # ── reads ────────────────────────────────────────────────────────────────
    def records(self) -> dict[str, ResourceRecord]:
        return self._catalog.load()

    def get(self, resource_id: str) -> ResourceRecord | None:
        return self._catalog.load().get(resource_id)

    def materialized_path(self, record: ResourceRecord) -> Path:
        """Absolute logical path of a record's current version (may not exist)."""
        rel = record.path
        if not rel:
            raise ResourcePathError(f"resource {record.id} has no storage path")
        return contained(self._ws, *validate_relative(rel).parts)

    def exists(self, record: ResourceRecord) -> bool:
        try:
            return self.materialized_path(record).exists()
        except ResourcePathError:
            return False

    def find_by_hash(
        self, content_hash: str, *, include_missing: bool = False
    ) -> ResourceRecord | None:
        target = normalize_hash(content_hash)
        if not target:
            return None
        for rec in self._catalog.load().values():
            if normalize_hash(rec.hash) != target:
                continue
            if include_missing or self.exists(rec):
                return rec
        return None

    def find_by_name(self, name: str, *, include_missing: bool = False) -> list[ResourceRecord]:
        target = safe_resource_name(name)
        out = [
            r
            for r in self._catalog.load().values()
            if r.name == target or r.name.startswith(target)
        ]
        if not include_missing:
            out = [r for r in out if self.exists(r)]
        return sorted(out, key=lambda r: r.updated_at, reverse=True)

    def list(
        self,
        *,
        task: str | None = None,
        visibility: str | None = None,
        kind: str | None = None,
        query: str = "",
        include_missing: bool = False,
        shared_only: bool = False,
    ) -> list[ResourceRecord]:
        """Records visible to ``task``: its own private ones plus every shared one.

        With ``task=None`` the caller is treated as a system actor and sees all.
        """
        q = (query or "").strip().lower()
        out: list[ResourceRecord] = []
        for rec in self._catalog.load().values():
            if not include_missing and not self.exists(rec):
                continue
            if shared_only and not rec.is_shared:
                continue
            if visibility and rec.visibility != visibility:
                continue
            if kind and rec.kind != kind:
                continue
            if task is not None and not (rec.is_shared or rec.owner_task == task):
                continue
            if q and q not in f"{rec.name} {rec.kind} {rec.url} {rec.hash}".lower():
                continue
            out.append(rec)
        out.sort(key=lambda r: r.last_used_at or r.updated_at, reverse=True)
        return out

    def resolve(self, ref: str, *, task: str | None = None, mode: str = "read") -> ResourceRecord:
        """Find one record by id, name or hash, then authorize access."""
        ref = (ref or "").strip()
        if not ref:
            raise ResourceNotFound("empty resource reference")
        records = self._catalog.load()
        rec = records.get(ref)
        if rec is None:
            matches = [r for r in records.values() if r.name == safe_resource_name(ref)]
            if not matches:
                target = normalize_hash(ref)
                matches = [r for r in records.values() if normalize_hash(r.hash) == target]
            if not matches:
                raise ResourceNotFound(f"no resource matches {ref!r}")
            if len(matches) > 1:
                # Prefer one visible to the caller.
                visible = [r for r in matches if self._is_visible(r, task)]
                if len(visible) == 1:
                    rec = visible[0]
                else:
                    raise ResourceNotFound(
                        f"{ref!r} is ambiguous ({len(matches)} matches); use the resource id"
                    )
            else:
                rec = matches[0]
        self.authorize(rec, task=task, mode=mode)
        return rec

    # ── authorization ────────────────────────────────────────────────────────
    def _is_visible(self, record: ResourceRecord, task: str | None) -> bool:
        return record.is_shared or task is None or record.owner_task == task

    def authorize(self, record: ResourceRecord, *, task: str | None, mode: str = "read") -> None:
        """Raise :class:`ResourceAccessDenied` unless ``task`` may act on ``record``."""
        if task is None:
            return  # system/CLI actor
        if mode == "read":
            if self._is_visible(record, task):
                return
        else:  # write / delete / promote — owner only
            if record.owner_task == task:
                return
        self._audit(
            "access_denied",
            record=record,
            task=task,
            ok=False,
            detail=f"mode={mode}",
        )
        raise ResourceAccessDenied(
            f"task '{task}' is not authorized to {mode} resource '{record.name}'",
            details={"resource_id": record.id, "owner_task": record.owner_task, "mode": mode},
        )

    # ── writes ───────────────────────────────────────────────────────────────
    def register(
        self,
        path: str | Path,
        *,
        name: str = "",
        kind: str = "",
        visibility: str = Visibility.private.value,
        task: str | None = None,
        task_id: str = "",
        owner_user: str | None = None,
        source: str = "local",
        url: str = "",
        tool: str = "",
        agent: str = "",
        session_id: str = "",
        note: str = "",
        dedupe: bool = True,
    ) -> ResourceRecord:
        """Index content already on disk as a resource (no bytes are moved).

        Used after a download to give the artifact identity, ownership, visibility,
        provenance and version history. Identical content is referenced, never
        copied.
        """
        return self._add(
            path,
            dest=None,
            name=name,
            kind=kind,
            visibility=visibility,
            task=task,
            task_id=task_id,
            owner_user=owner_user,
            source=source,
            url=url,
            tool=tool,
            agent=agent,
            session_id=session_id,
            note=note,
            dedupe=dedupe,
        )

    def put(
        self,
        path: str | Path,
        *,
        name: str = "",
        kind: str = "",
        visibility: str = Visibility.private.value,
        task: str | None = None,
        task_id: str = "",
        owner_user: str | None = None,
        source: str = "local",
        url: str = "",
        session_id: str = "",
        note: str = "",
        dedupe: bool = True,
    ) -> ResourceRecord:
        """Move content into the managed store, deduping against existing bytes.

        Private resources land under ``tasks/<date>/<slug>/resources/``; shared
        ones under ``shared/<kind>/``. Existing identical content is linked, not
        re-copied.
        """
        abs_src = self._resolve_input(path)
        if not abs_src.exists():
            raise ResourcePathError(f"source does not exist: {path}")
        final_name = safe_resource_name(name or abs_src.name)
        dest = self._dest_for(final_name, kind=kind, visibility=visibility, task=task)
        return self._add(
            abs_src,
            dest=dest,
            name=final_name,
            kind=kind,
            visibility=visibility,
            task=task,
            task_id=task_id,
            owner_user=owner_user,
            source=source,
            url=url,
            session_id=session_id,
            note=note,
            dedupe=dedupe,
        )

    def _add(
        self,
        path: str | Path,
        *,
        dest: Path | None,
        name: str,
        kind: str,
        visibility: str,
        task: str | None,
        owner_user: str | None,
        source: str,
        url: str,
        note: str,
        dedupe: bool,
        tool: str = "",
        agent: str = "",
        session_id: str = "",
        task_id: str = "",
    ) -> ResourceRecord:
        abs_path = self._resolve_input(path)
        if not abs_path.exists():
            raise ResourcePathError(f"path does not exist: {path}")
        if visibility not in (Visibility.private.value, Visibility.shared.value):
            raise ResourcePathError(f"invalid visibility: {visibility!r}")

        # Relocate into the managed layout when asked (put), else index in place.
        if dest is not None:
            dest = contained(self._ws, relative_to(self._ws, dest))
            if dest.resolve() != abs_path.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                _move_or_link(abs_path, dest)
                abs_path = dest

        ident = hash_path(abs_path)
        final_name = safe_resource_name(name or abs_path.name)
        if not kind:
            kind = classify_asset(relative_to(self._ws, abs_path), source)
        rel = relative_to(self._ws, abs_path)

        dedup_of = ""
        if dedupe:
            existing = self.find_by_hash(ident["hash"])
            if existing is not None:
                existing_path = self.materialized_path(existing)
                if (
                    existing_path.exists()
                    and existing_path.resolve() != abs_path.resolve()
                    and _dedupe_into(abs_path, existing_path)
                ):
                    dedup_of = existing.id
                    log.info(
                        "resource.deduped",
                        hash=ident["hash"],
                        reused=existing.id,
                        path=rel,
                    )

        storage = self._storage_for(abs_path, ident)
        prov = Provenance(
            source=source,
            url=(url or "").strip(),
            tool=tool,
            agent=agent,
            task_id=task or "",
            session_id=session_id,
            user_id=owner_user,
            note=note,
            dedup_of=dedup_of,
        )
        # Shared records stay global (owner_task=None) so every task touching the
        # same source links to ONE record; private records are owned by their task.
        # Either way the caller gets a link — that is what makes the task<->source
        # relationship bidirectional and lets the catalog see when one source is
        # shared by several tasks.
        owner_match = task if visibility == Visibility.private.value else None
        link = ResourceLink(
            task_id=(task_id or "").strip(),
            task_key=(task or "").strip(),
            session_id=(session_id or "").strip(),
            user_id=owner_user,
            role="owner" if (visibility == Visibility.private.value and task) else "consumer",
        )
        has_link = bool(link.task_id or link.task_key or link.session_id)

        def _apply(records: dict[str, ResourceRecord]) -> ResourceRecord:
            existing = self._find_logical(records, owner_match, final_name, kind, visibility)
            if existing is not None:
                if has_link:
                    self._attach_link(existing, link)
                if normalize_hash(existing.hash) == normalize_hash(ident["hash"]):
                    # Idempotent re-registration — refresh recency only.
                    existing.last_used_at = now_iso()
                    return existing
                existing.versions.append(
                    ResourceVersion(
                        version=existing.version + 1,
                        hash=ident["hash"],
                        hash_kind=ident["kind"],
                        size=ident["size"],
                        logical_path=rel,
                        provenance=prov,
                    )
                )
                existing.hash = ident["hash"]
                existing.hash_kind = ident["kind"]
                existing.size = ident["size"]
                existing.storage = storage
                existing.source = source
                existing.url = (url or "").strip() or existing.url
                existing.updated_at = now_iso()
                existing.last_used_at = now_iso()
                existing.provenance = prov
                log.info("resource.versioned", id=existing.id, version=existing.version)
                return existing

            record = ResourceRecord(
                id=uuid.uuid4().hex[:16],
                name=final_name,
                kind=kind,
                visibility=visibility,
                owner_task=owner_match,
                owner_user=owner_user,
                hash=ident["hash"],
                hash_kind=ident["kind"],
                size=ident["size"],
                storage=storage,
                source=source,
                url=(url or "").strip(),
                created_at=now_iso(),
                updated_at=now_iso(),
                provenance=prov,
            )
            record.versions.append(
                ResourceVersion(
                    version=1,
                    hash=ident["hash"],
                    hash_kind=ident["kind"],
                    size=ident["size"],
                    logical_path=rel,
                    provenance=prov,
                )
            )
            if has_link:
                record.links.append(link)
            records[record.id] = record
            return record

        record = self._catalog.mutate(_apply)
        if task and not record.is_shared:
            self._write_task_manifest(task)
        self._audit(
            "dedup" if dedup_of else "create",
            record=record,
            task=task,
            actor=owner_user,
            detail=f"hash={record.hash}",
        )
        return record

    def promote(
        self,
        resource_id: str,
        *,
        task: str | None,
        owner_user: str | None = None,
        actor: str | None = None,
    ) -> ResourceRecord:
        """Promote a private resource to shared (reusable by every task)."""
        current = self.get(resource_id)
        if current is None:
            raise ResourceNotFound(f"no resource with id {resource_id!r}")
        self.authorize(current, task=task, mode="write")
        if current.is_shared:
            return current

        src = self.materialized_path(current)
        if not src.exists():
            raise ResourceNotFound(f"resource {resource_id} has no bytes on disk")
        dest = self._dest_for(current.name, kind=current.kind, visibility=Visibility.shared.value)
        if src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _move_or_link(src, dest)
        rel = relative_to(self._ws, dest)

        def _apply(records: dict[str, ResourceRecord]) -> ResourceRecord:
            rec = records.get(resource_id)
            if rec is None:
                raise ResourceNotFound(f"no resource with id {resource_id!r}")
            rec.visibility = Visibility.shared.value
            rec.storage = {"mode": "path", "path": rel}
            rec.updated_at = now_iso()
            rec.last_used_at = now_iso()
            rec.provenance.note = (
                rec.provenance.note + " | promoted to shared"
                if rec.provenance.note
                else "promoted to shared"
            )
            return rec

        record = self._catalog.mutate(_apply)
        if task:
            self._write_task_manifest(task)
        self._audit("promote", record=record, task=task, actor=actor, detail="private->shared")
        return record

    def set_last_used(self, resource_id: str) -> None:
        def _apply(records: dict[str, ResourceRecord]) -> None:
            rec = records.get(resource_id)
            if rec is not None:
                rec.last_used_at = now_iso()

        self._catalog.mutate(_apply)

    def delete(
        self, resource_id: str, *, task: str | None, actor: str | None = None, purge: bool = False
    ) -> bool:
        """Remove a record. ``purge`` also deletes unreferenced bytes."""
        current = self.get(resource_id)
        if current is None:
            return False
        self.authorize(current, task=task, mode="write")

        removed: ResourceRecord | None = None

        def _apply(records: dict[str, ResourceRecord]) -> None:
            nonlocal removed
            removed = records.pop(resource_id, None)

        self._catalog.mutate(_apply)
        if removed is not None and purge:
            self._purge_record_bytes(removed)
        if task:
            self._write_task_manifest(task)
        self._audit("delete", record=removed, task=task, actor=actor, detail=f"purge={purge}")
        return removed is not None

    # ── task <-> source links ────────────────────────────────────────────────
    @staticmethod
    def _attach_link(record: ResourceRecord, link: ResourceLink) -> bool:
        """Add ``link`` unless an equivalent edge is already present (idempotent)."""
        for existing in record.links:
            if (
                existing.identity
                and existing.identity == link.identity
                and existing.role == link.role
            ):
                existing.session_id = existing.session_id or link.session_id
                existing.user_id = existing.user_id or link.user_id
                return False
        record.links.append(link)
        return True

    @staticmethod
    def _effective_links(record: ResourceRecord) -> list[ResourceLink]:
        """Explicit links, or a synthesized owner link for legacy records.

        Records created before links existed carry only ``owner_task``; treat that
        as an implicit owner edge so reference-counting stays correct.
        """
        if record.links:
            return list(record.links)
        if record.owner_task:
            return [
                ResourceLink(
                    task_key=record.owner_task,
                    session_id=record.provenance.session_id,
                    role="owner",
                )
            ]
        return []

    def link(
        self,
        resource_id: str,
        *,
        task_key: str = "",
        task_id: str = "",
        session_id: str = "",
        user_id: str | None = None,
        role: str = "consumer",
    ) -> ResourceRecord | None:
        """Link a task to a resource (idempotent). Returns the record (or None)."""
        new_link = ResourceLink(
            task_id=(task_id or "").strip(),
            task_key=(task_key or "").strip(),
            session_id=(session_id or "").strip(),
            user_id=user_id,
            role=role,
        )
        if not new_link.identity:
            return self.get(resource_id)

        def _apply(records: dict[str, ResourceRecord]) -> ResourceRecord | None:
            rec = records.get(resource_id)
            if rec is not None:
                self._attach_link(rec, new_link)
            return rec

        rec = self._catalog.mutate(_apply)
        if rec is not None:
            self._audit("link", record=rec, task=new_link.identity, detail=f"role={role}")
        return rec

    def unlink(self, resource_id: str, ref: str) -> bool:
        """Remove every edge matching ``ref`` (task id, task key or session id)."""
        removed = False

        def _apply(records: dict[str, ResourceRecord]) -> None:
            nonlocal removed
            rec = records.get(resource_id)
            if rec is None:
                return
            before = len(rec.links)
            rec.links = [link for link in rec.links if not link.matches(ref)]
            removed = len(rec.links) != before

        self._catalog.mutate(_apply)
        if removed:
            self._audit("unlink", resource_id=resource_id, task=ref)
        return removed

    def links_for(self, resource_id: str) -> list[dict[str, Any]]:
        """Every task edge on a resource (source -> tasks)."""
        rec = self.get(resource_id)
        if rec is None:
            return []
        return [link.to_dict() for link in self._effective_links(rec)]

    def tasks_for(self, resource_id: str) -> list[dict[str, Any]]:
        """Distinct tasks referencing a resource (deduped by identity)."""
        rec = self.get(resource_id)
        if rec is None:
            return []
        seen: dict[str, dict[str, Any]] = {}
        for link in self._effective_links(rec):
            ident = link.identity
            if ident and ident not in seen:
                seen[ident] = {
                    "task_id": link.task_id,
                    "task_key": link.task_key,
                    "session_id": link.session_id,
                    "user_id": link.user_id,
                    "role": link.role,
                }
        return list(seen.values())

    def resources_for(self, ref: str) -> list[ResourceRecord]:
        """Every resource a task references (task -> sources)."""
        ref = (ref or "").strip()
        if not ref:
            return []
        return [
            rec
            for rec in self._catalog.load().values()
            if any(link.matches(ref) for link in self._effective_links(rec))
        ]

    def reference_count(self, resource_id: str) -> int:
        return len(self.tasks_for(resource_id))

    # ── reference-counted deletion ───────────────────────────────────────────
    def _cascade(
        self, predicate: Any, *, purge: bool, action: str, task: str = ""
    ) -> dict[str, Any]:
        """Remove links matching ``predicate``; delete records left with no links.

        This is the core rule the design asks for: a **source** is kept while any
        *other* task still references it, and deleted only when the task being
        removed was its **last** link.
        """
        report: dict[str, Any] = {"unlinked": [], "deleted": [], "kept": [], "bytes_freed": 0}
        to_purge: list[ResourceRecord] = []

        def _apply(records: dict[str, ResourceRecord]) -> None:
            for rid, rec in list(records.items()):
                links = self._effective_links(rec)
                if not any(predicate(link) for link in links):
                    continue
                remaining = [link for link in links if not predicate(link)]
                report["unlinked"].append(rid)
                if remaining:
                    rec.links = remaining
                    report["kept"].append(rid)
                else:
                    rec.links = []
                    records.pop(rid, None)
                    report["deleted"].append(rid)
                    to_purge.append(rec)

        self._catalog.mutate(_apply)
        if purge:
            for rec in to_purge:
                report["bytes_freed"] += self._purge_record_bytes(rec)
        self._audit(
            action,
            task=task,
            detail=f"deleted={len(report['deleted'])} kept={len(report['kept'])}",
        )
        return report

    def delete_for_task(self, task_ref: str, *, purge: bool = True) -> dict[str, Any]:
        """Remove one run's links; delete a source only when it was the last link."""
        ref = (task_ref or "").strip()
        if not ref:
            return {"unlinked": [], "deleted": [], "kept": [], "bytes_freed": 0}
        return self._cascade(
            lambda link: ref in {link.task_id, link.task_key},
            purge=purge,
            action="delete_task",
            task=ref,
        )

    def delete_for_session(self, session_id: str, *, purge: bool = True) -> dict[str, Any]:
        """Remove all of a session's links (initial run + follow-ups), then cascade."""
        sid = (session_id or "").strip()
        key = self.task_identity(sid)
        return self._cascade(
            lambda link: link.session_id == sid or link.task_key == key,
            purge=purge,
            action="delete_session",
            task=sid,
        )

    def _purge_record_bytes(self, record: ResourceRecord) -> int:
        """Delete a record's bytes **only** if no other record still references them."""
        target = normalize_hash(record.hash)
        others = [r for r in self._catalog.load().values() if normalize_hash(r.hash) == target]
        if others:
            return 0  # content is still reachable through another resource record
        freed = 0
        paths: list[Path] = []
        with contextlib.suppress(ResourcePathError):
            paths.append(self.materialized_path(record))
        obj = self._object_abs(record)
        if obj is not None:
            paths.append(obj)
        for path in paths:
            freed += _remove_path(path)
        return freed

    # ── isolation: shared -> task-private ────────────────────────────────────
    def isolate_task(self, task_ref: str, *, copy: bool = True) -> dict[str, Any]:
        """Make every shared source a task uses private to that task.

        Copies the bytes into the task's private area (a real copy, never a
        hard-link, so the copy is genuinely independent). Sources whose bytes are
        gone but whose URL is known are returned in ``needs_redownload`` for the
        caller to re-fetch; sources with neither are reported ``unrecoverable``.
        """
        ref = (task_ref or "").strip()
        key = self.task_identity(ref)
        result: dict[str, Any] = {
            "task": key,
            "isolated": [],
            "needs_redownload": [],
            "unrecoverable": [],
            "orphaned": [],
        }
        for rec in list(self._catalog.load().values()):
            if not rec.is_shared:
                continue
            links = self._effective_links(rec)
            matched = [link for link in links if link.matches(ref) or link.matches(key)]
            if not matched:
                continue
            link = matched[0]
            private_dest = contained(self.private_root(key), rec.name)
            if copy and self.exists(rec):
                self._copy_path(self.materialized_path(rec), private_dest)
                priv = self.put(
                    private_dest,
                    name=rec.name,
                    kind=rec.kind,
                    visibility=Visibility.private.value,
                    task=key,
                    task_id=link.task_id,
                    session_id=link.session_id,
                    source=rec.source,
                    url=rec.url,
                    note=f"isolated from shared {rec.id}",
                    dedupe=False,
                )
                self.unlink(rec.id, ref)
                self.unlink(rec.id, key)
                result["isolated"].append({"shared": rec.id, "private": priv.id, "path": priv.path})
                if not self._effective_links(self.get(rec.id) or rec):
                    result["orphaned"].append(rec.id)
            elif rec.url:
                result["needs_redownload"].append(
                    {
                        "shared": rec.id,
                        "name": rec.name,
                        "kind": rec.kind,
                        "url": rec.url,
                        "source": rec.source,
                        "task_key": key,
                        "task_id": link.task_id,
                        "session_id": link.session_id,
                        "dest": relative_to(self._ws, private_dest),
                    }
                )
            else:
                result["unrecoverable"].append({"shared": rec.id, "name": rec.name})
        self._audit(
            "isolate_task",
            task=key,
            detail=(
                f"isolated={len(result['isolated'])} "
                f"redownload={len(result['needs_redownload'])}"
            ),
        )
        return result

    def isolate_all(self, *, copy: bool = True, purge_orphans: bool = True) -> dict[str, Any]:
        """Isolate every task, then drop shared records left with no task links."""
        keys: set[str] = set()
        for rec in self._catalog.load().values():
            for link in self._effective_links(rec):
                if link.task_key:
                    keys.add(link.task_key)
                elif link.task_id:
                    keys.add(link.task_id)
        combined: dict[str, Any] = {
            "tasks": sorted(keys),
            "isolated": [],
            "needs_redownload": [],
            "unrecoverable": [],
            "orphaned": [],
            "purged": [],
        }
        orphan_ids: list[str] = []
        for key in combined["tasks"]:
            res = self.isolate_task(key, copy=copy)
            combined["isolated"] += res["isolated"]
            combined["needs_redownload"] += res["needs_redownload"]
            combined["unrecoverable"] += res["unrecoverable"]
            orphan_ids += res["orphaned"]
        combined["orphaned"] = sorted(set(orphan_ids))
        if purge_orphans and combined["orphaned"]:
            combined["purged"] = self.purge_records(combined["orphaned"])
        self._audit(
            "isolate_all",
            detail=(
                f"tasks={len(combined['tasks'])} isolated={len(combined['isolated'])} "
                f"purged={len(combined['purged'])}"
            ),
        )
        return combined

    def purge_records(self, resource_ids: list[str], *, purge_bytes: bool = True) -> list[str]:
        """Delete records by id **only if they have no remaining links**."""
        wanted = set(resource_ids)
        removed: list[str] = []
        to_purge: list[ResourceRecord] = []

        def _apply(records: dict[str, ResourceRecord]) -> None:
            for rid in list(wanted):
                rec = records.get(rid)
                if rec is None:
                    continue
                if self._effective_links(rec):
                    continue  # still referenced — never orphan it
                records.pop(rid, None)
                removed.append(rid)
                to_purge.append(rec)

        self._catalog.mutate(_apply)
        if purge_bytes:
            for rec in to_purge:
                self._purge_record_bytes(rec)
        return removed

    def purge_all(self, *, purge_bytes: bool = True, clear_registry: bool = True) -> dict[str, Any]:
        """Delete **every** resource record and (optionally) its bytes — fresh start.

        Wipes the catalog, the bytes each record points at, the managed ``shared/``
        and ``tasks/`` trees, and the content-addressed object store. With
        ``clear_registry`` it also drops the legacy ``assets.json`` download cache so
        future runs re-acquire instead of reusing old files.
        """
        records = self._catalog.load()
        ids = sorted(records)
        paths: set[Path] = set()
        for rec in records.values():
            with contextlib.suppress(ResourcePathError):
                paths.add(self.materialized_path(rec))
            obj = self._object_abs(rec)
            if obj is not None:
                paths.add(obj)

        freed = 0
        # Wipe the index first so a crash mid-delete can never resurrect stale rows.
        self._catalog.save({})
        if purge_bytes:
            # Deepest paths first so removing a parent can't strand a child lookup.
            for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
                freed += _remove_path(path)
            freed += _remove_path(self._objects)
            freed += _remove_path(self.shared_root())
            freed += _remove_path(self._ws / TASKS_DIR)
            if clear_registry:
                freed += _remove_path(self._ws / "assets.json")
        self._audit(
            "purge_all",
            detail=f"deleted={len(ids)} purge_bytes={purge_bytes} clear_registry={clear_registry}",
        )
        log.info("resources.purge_all", deleted=len(ids), bytes_freed=freed)
        return {
            "deleted": len(ids),
            "resources": ids,
            "bytes_freed": freed,
            "purge_bytes": purge_bytes,
        }

    def wipe_workspace(self) -> dict[str, Any]:
        """Delete **everything** under the workspace — a truly fresh start.

        Unlike :meth:`purge_all` (which only removes *catalogued* resources), this
        clears every file and folder in the workspace, including downloads and task
        outputs that were never registered (``models/``, ``data/``, ``sessions/``,
        stray files, the catalog and the object store). The workspace directory
        itself is kept and recreated empty.
        """
        ws = self._ws
        ws.mkdir(parents=True, exist_ok=True)
        removed: list[str] = []
        freed = 0
        for child in sorted(ws.iterdir()):
            freed += _remove_path(child)
            removed.append(child.name)
        self._audit("wipe_workspace", detail=f"removed={len(removed)}")
        log.info("resources.wipe_workspace", removed=len(removed), bytes_freed=freed)
        return {"removed": removed, "removed_count": len(removed), "bytes_freed": freed}

    @staticmethod
    def _copy_path(src: Path, dest: Path) -> None:
        """Real (non-linked) copy of a file or directory tree into ``dest``.

        ``dest`` is removed first so that a pre-existing hard-link to ``src`` (the
        content-addressed object store shares inodes) can't make ``copytree`` /
        ``copy2`` fail with *same file* — the copy must be a genuinely independent
        inode, which is the whole point of isolation.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    dest.unlink()
        if src.is_dir():
            shutil.copytree(src, dest, symlinks=False)
        else:
            shutil.copy2(src, dest)

    # ── verification ─────────────────────────────────────────────────────────
    def verify(self, *, task: str | None = None, rehash: bool = True) -> dict[str, Any]:
        """Check every visible record's bytes are present (and unchanged)."""
        present: list[str] = []
        missing: list[dict[str, str]] = []
        mismatched: list[dict[str, str]] = []
        for rec in self.list(task=task, include_missing=True):
            path = self.materialized_path(rec)
            if not path.exists():
                missing.append({"id": rec.id, "name": rec.name, "path": rec.path})
                continue
            if rehash:
                try:
                    ident = hash_path(path)
                except OSError:
                    missing.append({"id": rec.id, "name": rec.name, "path": rec.path})
                    continue
                if normalize_hash(ident["hash"]) != normalize_hash(rec.hash):
                    mismatched.append(
                        {
                            "id": rec.id,
                            "name": rec.name,
                            "expected": rec.hash,
                            "actual": ident["hash"],
                        }
                    )
                    continue
            present.append(rec.id)
        return {
            "ok": not missing and not mismatched,
            "checked": len(present) + len(missing) + len(mismatched),
            "present": len(present),
            "missing": missing,
            "mismatched": mismatched,
        }

    # ── internals ────────────────────────────────────────────────────────────
    def _resolve_input(self, path: str | Path) -> Path:
        """Accept an absolute path inside the workspace, or a workspace/cwd-relative one."""
        p = Path(path)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self._ws / p)
            candidates.append(Path.cwd() / p)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                relative_to(self._ws, resolved)
                return resolved
            except (ResourcePathError, OSError):
                continue
        raise ResourcePathError(
            f"path is outside the workspace: {path}", details={"workspace": str(self._ws)}
        )

    def _dest_for(self, name: str, *, kind: str, visibility: str, task: str | None = None) -> Path:
        if visibility == Visibility.shared.value:
            return contained(self.shared_root(), kind or "file", name)
        if not task:
            raise ResourcePathError("private resources require a task key")
        return contained(self.private_root(task), name)

    def _storage_for(self, abs_path: Path, ident: dict[str, Any]) -> dict[str, Any]:
        rel = relative_to(self._ws, abs_path)
        storage: dict[str, Any] = {"mode": "path", "path": rel}
        # Files get a canonical object-store copy so identical content shares one inode.
        if abs_path.is_file():
            try:
                obj_key = self._ensure_object(ident["hash"], abs_path)
                storage = {"mode": "object", "object": obj_key, "path": rel}
            except OSError as exc:  # pragma: no cover - fall back to path storage
                log.warning("resource.object_store_failed", error=str(exc))
        return storage

    def _ensure_object(self, content_hash: str, content: Path) -> str:
        digest = normalize_hash(content_hash).split(":", 1)[1]
        key = f"{digest[:2]}/{digest}"
        obj = self.object_path(key)
        if obj.exists():
            return key
        obj.parent.mkdir(parents=True, exist_ok=True)
        # Copy to a UNIQUE temp name, then atomically link/replace so two processes
        # racing on the same content can't clobber each other's partial object.
        tmp = obj.with_name(f"{obj.name}.{uuid.uuid4().hex}.tmp")
        try:
            try:
                os.link(content, tmp)
            except OSError:
                shutil.copy2(content, tmp)
            os.replace(tmp, obj)
        finally:
            with contextlib.suppress(OSError):
                if tmp.exists() and tmp != obj:
                    tmp.unlink()
        return key

    def _object_abs(self, record: ResourceRecord) -> Path | None:
        key = record.storage.get("object")
        if not key:
            return None
        try:
            return self.object_path(str(key))
        except ResourcePathError:
            return None

    def _find_logical(
        self,
        records: dict[str, ResourceRecord],
        task: str | None,
        name: str,
        kind: str,
        visibility: str,
    ) -> ResourceRecord | None:
        for rec in records.values():
            if (
                rec.name == name
                and rec.kind == kind
                and rec.visibility == visibility
                and rec.owner_task == task
            ):
                return rec
        return None

    def _write_task_manifest(self, task: str) -> None:
        try:
            root = self.task_identity_root(task)
            root.mkdir(parents=True, exist_ok=True)
            rows = [
                {"id": r.id, "name": r.name, "kind": r.kind, "hash": r.hash, "path": r.path}
                for r in self._catalog.load().values()
                if r.owner_task == task
            ]
            import json

            tmp = root / f"manifest.json.{uuid.uuid4().hex}.tmp"
            tmp.write_text(
                json.dumps({"task": task, "resources": rows}, indent=2), encoding="utf-8"
            )
            os.replace(tmp, root / "manifest.json")
        except (OSError, ResourcePathError) as exc:
            log.warning("resource.manifest_failed", task=task, error=str(exc))


# ── byte-level helpers ───────────────────────────────────────────────────────
def _dedupe_into(target: Path, canonical: Path) -> bool:
    """Make ``target`` share ``canonical``'s bytes. Best effort; True on success."""
    try:
        if canonical.is_dir():
            return _link_tree(canonical, target, replace=True)
        if canonical.is_file() and target.is_file():
            _replace_with_link(target, canonical)
            return True
    except OSError:
        return False
    return False


def _link_tree(source: Path, dest: Path, *, replace: bool) -> bool:
    """Hard-link every file from ``source`` into ``dest`` (copies on failure)."""
    if replace and dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    for dirpath, _dirnames, filenames in os.walk(source):
        rel = Path(dirpath).relative_to(source)
        (dest / rel).mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            _replace_with_link(dest / rel / fn, Path(dirpath) / fn)
    return True


def _replace_with_link(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _move_or_link(src: Path, dest: Path) -> None:
    """Move ``src`` to ``dest`` (hardlink+unlink on one filesystem, else copy)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
        else:
            dest.unlink()
    try:
        os.link(src, dest)
        if src.is_dir():
            shutil.rmtree(src, ignore_errors=True)
        else:
            src.unlink()
        return
    except OSError:
        pass
    if src.is_dir():
        shutil.move(str(src), str(dest))
    else:
        shutil.move(str(src), str(dest))


def _remove_path(path: Path | None) -> int:
    """Delete a file or directory tree, returning the bytes reclaimed (0 on error)."""
    if path is None:
        return 0
    try:
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path, ignore_errors=True)
            return size
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            return size
    except OSError:
        return 0
    return 0


__all__ = ["ResourceManager"]
