"""MemoryManager — the high-level API agents use to remember and recall.

It coordinates a :class:`VectorStore` (semantic recall) with the embedder Brain,
and maintains lightweight short-term conversation buffers per session. All memory
types are namespaces, so one consistent API serves them all.
"""

from __future__ import annotations

from typing import Any

from ai_employee.brains.base import Brain
from ai_employee.config import Settings, get_settings
from ai_employee.core.logging import get_logger
from ai_employee.memory.base import MemoryNamespace, MemoryRecord, VectorStore
from ai_employee.memory.in_memory import InMemoryVectorStore

log = get_logger("memory")


class MemoryManager:
    def __init__(self, store: VectorStore, embedder: Brain) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    # ── Write ────────────────────────────────────────────────────────────────
    async def record(
        self,
        content: str,
        *,
        namespace: MemoryNamespace = MemoryNamespace.long_term,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        embed: bool = True,
    ) -> MemoryRecord:
        """Persist a memory, embedding it for later semantic recall."""
        embedding = None
        if embed:
            try:
                embedding = (await self._embedder.embed([content]))[0]
            except Exception as exc:  # noqa: BLE001 - degrade gracefully w/o embeddings
                log.warning("memory.embed_failed", error=str(exc))
        rec = MemoryRecord(
            namespace=namespace,
            content=content,
            metadata=metadata or {},
            session_id=session_id,
            embedding=embedding,
        )
        await self._store.upsert([rec])
        log.debug("memory.record", namespace=namespace.value, id=rec.id)
        return rec

    async def record_message(self, role: str, content: str, *, session_id: str) -> MemoryRecord:
        """Append a message to conversation memory (short-term + durable)."""
        return await self.record(
            content,
            namespace=MemoryNamespace.conversation,
            metadata={"role": role},
            session_id=session_id,
        )

    async def remember_skill(
        self, name: str, description: str, steps: list[str], *, tags: list[str] | None = None
    ) -> MemoryRecord:
        """Store a reusable skill/workflow the employee learned."""
        content = f"SKILL: {name}\n{description}\nSTEPS:\n" + "\n".join(f"- {s}" for s in steps)
        return await self.record(
            content,
            namespace=MemoryNamespace.skill,
            metadata={"name": name, "tags": tags or [], "steps": steps},
        )

    # ── Read ─────────────────────────────────────────────────────────────────
    async def recall(
        self,
        query: str,
        *,
        namespace: MemoryNamespace = MemoryNamespace.long_term,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Semantic recall from a namespace."""
        try:
            vec = (await self._embedder.embed([query]))[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.recall_embed_failed", error=str(exc))
            return await self._store.recent(namespace, limit=k, session_id=session_id)
        return await self._store.search(namespace, vec, k=k, session_id=session_id)

    async def conversation_window(self, session_id: str, *, limit: int = 20) -> list[MemoryRecord]:
        """The most recent messages — the short-term working memory."""
        records = await self._store.recent(
            MemoryNamespace.conversation, limit=limit, session_id=session_id
        )
        return list(reversed(records))  # chronological order

    async def find_skill(self, query: str, *, k: int = 3) -> list[MemoryRecord]:
        return await self.recall(query, namespace=MemoryNamespace.skill, k=k)

    async def clear(self, namespace: MemoryNamespace | None = None) -> None:
        await self._store.clear(namespace)

    async def clear_session(self, session_id: str) -> int:
        """Remove every memory record scoped to a saved task/session."""
        clear = getattr(self._store, "clear_session", None)
        if clear is None:
            return 0
        return await clear(session_id)

    # ── Persistence (only meaningful for the in-memory/resilient backends;
    #    pgvector is already durable in Postgres) ─────────────────────────────
    @property
    def is_persistable(self) -> bool:
        return hasattr(self._store, "export_records")

    def export_records(self) -> list[dict]:
        return self._store.export_records() if self.is_persistable else []

    def import_records(self, records: list[dict]) -> None:
        if self.is_persistable:
            self._store.import_records(records)


def build_memory_manager(embedder: Brain, settings: Settings | None = None) -> MemoryManager:
    """Construct a MemoryManager with the appropriate backend.

    Uses pgvector when a database is configured and reachable; otherwise falls
    back to the in-memory store so the framework always runs.
    """
    s = settings or get_settings()
    backend = (s.memory_backend or "auto").lower()
    store: VectorStore

    if backend == "memory":
        log.info("memory.backend", backend="in_memory")
        return MemoryManager(InMemoryVectorStore(), embedder)

    # Try to construct the pgvector store (cheap TCP probe inside).
    try:
        from ai_employee.memory.pg_store import PgVectorStore

        pg = PgVectorStore(s)
    except Exception as exc:  # noqa: BLE001
        if backend == "pgvector":
            raise  # explicit pgvector mode must fail loudly
        log.info("memory.backend", backend="in_memory", reason=str(exc))
        return MemoryManager(InMemoryVectorStore(), embedder)

    if backend == "pgvector":
        log.info("memory.backend", backend="pgvector")
        store = pg
    else:  # auto: durable when it works, graceful fallback when it doesn't
        from ai_employee.memory.resilient import ResilientVectorStore

        log.info("memory.backend", backend="pgvector+resilient")
        store = ResilientVectorStore(pg)
    return MemoryManager(store, embedder)
