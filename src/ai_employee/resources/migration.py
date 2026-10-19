"""Migrate pre-existing workspace data into the resource catalog.

Older installs kept their assets in three flat buckets — ``models/``, ``data/``
and ``sessions/<session>/`` — indexed only by ``assets.json``. This walks that
legacy layout (and the ledger) once, registers each artifact with a content hash,
dedup, provenance and correct visibility, and reports what it did.

Nothing is moved or deleted: migration only *indexes*. Re-running it is a no-op
because already-cataloged paths are skipped, so it is safe to call at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger
from ai_employee.observability.asset_registry import AssetRegistry, classify_asset
from ai_employee.resources.models import Visibility
from ai_employee.resources.names import relative_to, safe_task_identity
from ai_employee.resources.store import ResourceManager

log = get_logger("resources.migration")

#: Buckets that hold shared, reusable resources.
_SHARED_BUCKETS = ("models", "data", "datasets")
#: Buckets/dirs that are framework-managed or irrelevant — never migrated as content.
_SKIP = {
    ".resources",
    "tasks",
    "shared",
    ".hf_home",
    ".cache",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "assets.json",
}


def _iter_bucket(ws: Path, bucket: str) -> list[Path]:
    root = ws / bucket
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.name not in _SKIP)


def _candidates(manager: ResourceManager) -> list[dict[str, Any]]:
    """Every legacy artifact worth registering, with its intended visibility."""
    ws = manager.workspace
    found: list[dict[str, Any]] = []

    # 1. Shared buckets.
    for bucket in _SHARED_BUCKETS:
        for path in _iter_bucket(ws, bucket):
            found.append(
                {
                    "path": path,
                    "visibility": Visibility.shared.value,
                    "owner_task": None,
                    "source": "migration",
                    "url": "",
                    "session_id": "",
                }
            )

    # 2. Per-session (private) files.
    sessions = ws / "sessions"
    if sessions.is_dir():
        for sdir in sorted(p for p in sessions.iterdir() if p.is_dir()):
            if sdir.name in _SKIP:
                continue
            for path in sorted(p for p in sdir.iterdir() if p.name not in _SKIP):
                found.append(
                    {
                        "path": path,
                        "visibility": Visibility.private.value,
                        "owner_task": safe_task_identity(sdir.name),
                        "source": "migration",
                        "url": "",
                        "session_id": sdir.name,
                    }
                )

    # 3. Anything the old asset ledger knew about that the buckets missed.
    ledger = AssetRegistry(ws)
    for record in ledger.list_assets(verify=True):
        path = ws / record.path
        rel_parts = Path(record.path).parts
        if not rel_parts or rel_parts[0] in _SHARED_BUCKETS or rel_parts[0] == "sessions":
            continue  # already covered above
        if any(p in _SKIP for p in rel_parts):
            continue
        found.append(
            {
                "path": path,
                "visibility": Visibility.shared.value,
                "owner_task": None,
                "source": record.source or "migration",
                "url": record.url,
                "session_id": record.session_id,
            }
        )
    return found


def migrate_workspace(manager: ResourceManager, *, apply: bool = False) -> dict[str, Any]:
    """Index legacy workspace artifacts. ``apply=False`` is a dry run.

    Returns ``{"candidates", "registered", "skipped", "entries", "applied"}``.
    """
    existing_paths = {r.path for r in manager.records().values()}
    entries: list[dict[str, Any]] = []
    registered = 0
    skipped = 0

    for item in _candidates(manager):
        path: Path = item["path"]
        try:
            rel = relative_to(manager.workspace, path)
        except Exception:  # noqa: BLE001 - outside workspace, skip
            continue
        kind = classify_asset(rel, item["source"])
        already = rel in existing_paths
        entry = {
            "path": rel,
            "kind": kind,
            "visibility": item["visibility"],
            "owner_task": item["owner_task"],
            "action": "skip" if already else ("register" if apply else "plan"),
        }
        if already:
            skipped += 1
            entries.append(entry)
            continue
        if apply:
            record = manager.register(
                path,
                name=Path(rel).name,
                kind=kind,
                visibility=item["visibility"],
                task=item["owner_task"],
                source=item["source"],
                url=item["url"],
                session_id=item["session_id"],
                tool="migration",
                note="migrated from legacy workspace layout",
            )
            entry["resource_id"] = record.id
            existing_paths.add(rel)
            registered += 1
        entries.append(entry)

    report = {
        "applied": apply,
        "candidates": len(entries),
        "registered": registered,
        "skipped": skipped,
        "entries": entries,
    }
    log.info(
        "resources.migrate",
        applied=apply,
        candidates=len(entries),
        registered=registered,
        skipped=skipped,
    )
    return report


__all__ = ["migrate_workspace"]
