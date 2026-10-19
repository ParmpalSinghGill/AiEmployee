"""Optional mirror of the resource catalog into Postgres.

The filesystem catalog stays authoritative so the framework works offline; this
helper lets a deployed app also upsert a :class:`~ai_employee.database.models.Resource`
row per catalog entry for SQL reporting. Imported lazily so nothing pulls in the
DB layer unless a caller actually mirrors.
"""

from __future__ import annotations

from typing import Any

from ai_employee.resources.models import ResourceRecord


def row_from_record(record: ResourceRecord) -> dict[str, Any]:
    """Map a :class:`ResourceRecord` onto ``Resource`` table columns."""
    return {
        "id": record.id,
        "name": record.name,
        "kind": record.kind,
        "visibility": record.visibility,
        "owner_task": record.owner_task,
        "owner_user": record.owner_user,
        "content_hash": record.hash,
        "hash_kind": record.hash_kind,
        "size": record.size,
        "logical_path": record.path,
        "source": record.source,
        "url": record.url,
        "version": record.version,
        "meta": {
            "storage": record.storage,
            "tags": record.tags,
            "provenance": record.provenance.to_dict(),
            "versions": [v.to_dict() for v in record.versions],
        },
    }


async def upsert_resource(session: Any, record: ResourceRecord) -> None:
    """Insert or update the mirror row for ``record`` (caller commits)."""
    from ai_employee.database.models import Resource

    row = row_from_record(record)
    existing = await session.get(Resource, record.id)
    if existing is None:
        session.add(Resource(**row))
        return
    for key, value in row.items():
        setattr(existing, key, value)
    await session.flush()


__all__ = ["row_from_record", "upsert_resource"]
