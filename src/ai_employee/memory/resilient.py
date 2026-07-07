"""A VectorStore wrapper that falls back to in-memory on backend failure.

Used in ``MEMORY_BACKEND=auto``: the framework prefers durable pgvector storage but
must never crash a task because the database is unavailable or misconfigured. On
the first backend error it logs a warning, switches permanently to the in-memory
store for the process, and continues.
"""

from __future__ import annotations

from ai_employee.core.logging import get_logger
from ai_employee.memory.base import MemoryNamespace, MemoryRecord, VectorStore
from ai_employee.memory.in_memory import InMemoryVectorStore

log = get_logger("memory.resilient")


class ResilientVectorStore(VectorStore):
    def __init__(self, primary: VectorStore) -> None:
        self._primary = primary
        self._fallback = InMemoryVectorStore()
        self._degraded = False

    @property
    def _active(self) -> VectorStore:
        return self._fallback if self._degraded else self._primary

    def _degrade(self, exc: Exception) -> None:
        if not self._degraded:
            log.warning("memory.degraded_to_in_memory", error=str(exc))
            self._degraded = True

    async def upsert(self, records: list[MemoryRecord]) -> None:
        try:
            await self._primary.upsert(records) if not self._degraded else None
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        # Always mirror into the fallback so reads remain consistent post-degrade.
        await self._fallback.upsert(records)

    async def search(self, namespace, query_embedding, *, k=5, session_id=None):
        try:
            if not self._degraded:
                return await self._primary.search(
                    namespace, query_embedding, k=k, session_id=session_id
                )
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        return await self._fallback.search(namespace, query_embedding, k=k, session_id=session_id)

    async def recent(self, namespace, *, limit=20, session_id=None):
        try:
            if not self._degraded:
                return await self._primary.recent(namespace, limit=limit, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        return await self._fallback.recent(namespace, limit=limit, session_id=session_id)

    async def delete(self, ids: list[str]) -> None:
        try:
            if not self._degraded:
                await self._primary.delete(ids)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        await self._fallback.delete(ids)

    async def clear(self, namespace: MemoryNamespace | None = None) -> None:
        try:
            if not self._degraded:
                await self._primary.clear(namespace)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        await self._fallback.clear(namespace)

    async def clear_session(self, session_id: str) -> int:
        removed = 0
        try:
            if not self._degraded and hasattr(self._primary, "clear_session"):
                removed = await self._primary.clear_session(session_id)
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
        removed = max(removed, await self._fallback.clear_session(session_id))
        return removed

    # The in-memory mirror always holds every record, so persistence delegates to it.
    def export_records(self) -> list[dict]:
        return self._fallback.export_records()

    def import_records(self, records: list[dict]) -> None:
        self._fallback.import_records(records)
