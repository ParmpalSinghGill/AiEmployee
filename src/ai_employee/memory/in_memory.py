"""In-memory vector store — the default offline backend.

Cosine similarity over numpy. Not durable, but lets the entire memory/RAG stack
run and be tested with zero infrastructure.
"""

from __future__ import annotations

import math

from ai_employee.memory.base import MemoryNamespace, MemoryRecord, VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    async def upsert(self, records: list[MemoryRecord]) -> None:
        for r in records:
            self._records[r.id] = r

    async def search(
        self,
        namespace: MemoryNamespace,
        query_embedding: list[float],
        *,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        candidates = [
            r
            for r in self._records.values()
            if r.namespace == namespace
            and r.embedding is not None
            and (session_id is None or r.session_id == session_id)
        ]
        scored: list[tuple[float, MemoryRecord]] = []
        for r in candidates:
            sim = _cosine(query_embedding, r.embedding or [])
            copy = r.model_copy(update={"score": sim})
            scored.append((sim, copy))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    async def recent(
        self,
        namespace: MemoryNamespace,
        *,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        items = [
            r
            for r in self._records.values()
            if r.namespace == namespace and (session_id is None or r.session_id == session_id)
        ]
        items.sort(key=lambda r: r.created_at, reverse=True)
        return items[:limit]

    async def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._records.pop(i, None)

    async def clear(self, namespace: MemoryNamespace | None = None) -> None:
        if namespace is None:
            self._records.clear()
        else:
            self._records = {k: v for k, v in self._records.items() if v.namespace != namespace}

    async def clear_session(self, session_id: str) -> int:
        to_remove = [k for k, v in self._records.items() if v.session_id == session_id]
        for k in to_remove:
            del self._records[k]
        return len(to_remove)

    # ── Persistence helpers ──────────────────────────────────────────────────
    def export_records(self) -> list[dict]:
        return [r.model_dump(mode="json") for r in self._records.values()]

    def import_records(self, records: list[dict]) -> None:
        for d in records:
            try:
                rec = MemoryRecord.model_validate(d)
            except Exception:  # noqa: BLE001 - skip malformed rows
                continue
            self._records[rec.id] = rec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
