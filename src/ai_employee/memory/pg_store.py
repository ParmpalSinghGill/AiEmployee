"""pgvector-backed VectorStore.

Production memory backend. Uses cosine distance over the ``memories`` table. The
constructor performs a quick connectivity check so the manager can transparently
fall back to the in-memory store when Postgres is absent.
"""

from __future__ import annotations

from ai_employee.config import Settings
from ai_employee.core.logging import get_logger
from ai_employee.database.session import db_session
from ai_employee.memory.base import MemoryNamespace, MemoryRecord, VectorStore

log = get_logger("memory.pg")


class PgVectorStore(VectorStore):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.postgres_host:
            raise RuntimeError("postgres not configured")
        # Fast TCP probe so we cleanly fall back to the in-memory store when no
        # database is running (e.g. local/offline development & tests).
        import socket

        try:
            with socket.create_connection(
                (settings.postgres_host, settings.postgres_port), timeout=0.5
            ):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"postgres unreachable at {settings.postgres_host}:{settings.postgres_port}"
            ) from exc

    async def upsert(self, records: list[MemoryRecord]) -> None:
        from ai_employee.database.models import MemoryItem

        async with db_session(self._settings) as session:
            for r in records:
                session.add(
                    MemoryItem(
                        id=r.id,
                        namespace=r.namespace.value,
                        content=r.content,
                        session_id=r.session_id,
                        meta=r.metadata,
                        embedding=r.embedding,
                    )
                )

    async def search(
        self,
        namespace: MemoryNamespace,
        query_embedding: list[float],
        *,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        from sqlalchemy import select

        from ai_employee.database.models import MemoryItem

        async with db_session(self._settings) as session:
            stmt = select(MemoryItem).where(MemoryItem.namespace == namespace.value)
            if session_id is not None:
                stmt = stmt.where(MemoryItem.session_id == session_id)
            stmt = stmt.order_by(MemoryItem.embedding.cosine_distance(query_embedding)).limit(k)
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_record(r) for r in rows]

    async def recent(
        self,
        namespace: MemoryNamespace,
        *,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        from sqlalchemy import select

        from ai_employee.database.models import MemoryItem

        async with db_session(self._settings) as session:
            stmt = select(MemoryItem).where(MemoryItem.namespace == namespace.value)
            if session_id is not None:
                stmt = stmt.where(MemoryItem.session_id == session_id)
            stmt = stmt.order_by(MemoryItem.created_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_record(r) for r in rows]

    async def delete(self, ids: list[str]) -> None:
        from sqlalchemy import delete

        from ai_employee.database.models import MemoryItem

        async with db_session(self._settings) as session:
            await session.execute(delete(MemoryItem).where(MemoryItem.id.in_(ids)))

    async def clear(self, namespace: MemoryNamespace | None = None) -> None:
        from sqlalchemy import delete

        from ai_employee.database.models import MemoryItem

        async with db_session(self._settings) as session:
            stmt = delete(MemoryItem)
            if namespace is not None:
                stmt = stmt.where(MemoryItem.namespace == namespace.value)
            await session.execute(stmt)

    async def clear_session(self, session_id: str) -> int:
        from sqlalchemy import delete

        from ai_employee.database.models import MemoryItem

        async with db_session(self._settings) as session:
            result = await session.execute(
                delete(MemoryItem).where(MemoryItem.session_id == session_id)
            )
            return int(result.rowcount or 0)


def _to_record(row) -> MemoryRecord:
    return MemoryRecord(
        id=row.id,
        namespace=MemoryNamespace(row.namespace),
        content=row.content,
        metadata=row.meta or {},
        session_id=row.session_id,
        embedding=list(row.embedding) if row.embedding is not None else None,
        created_at=row.created_at,
    )
