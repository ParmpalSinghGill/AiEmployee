"""MemoryManager — the high-level API agents use to remember and recall.

It coordinates a :class:`VectorStore` (semantic recall) with the embedder Brain,
and maintains lightweight short-term conversation buffers per session. All memory
types are namespaces, so one consistent API serves them all.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from ai_employee.brains.base import Brain
from ai_employee.config import Settings, get_settings
from ai_employee.core.logging import get_logger
from ai_employee.memory.base import MemoryNamespace, MemoryRecord, VectorStore
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.memory.skill_policy import (
    SkillRejectedError,
    sanitize_skill,
    skill_name_from_content,
    skill_rejection_reason,
    skill_rejection_reason_from_dict,
    skill_rejection_reason_from_record,
    skill_slug,
    task_specific_reason,
)

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
        """Store a reusable skill/workflow the employee learned.

        Sanitizes against task-specific content, deduplicates by title, and raises
        :class:`SkillRejectedError` when the candidate is not a general skill.
        """
        return await self.save_skill(name=name, description=description, steps=steps, tags=tags)

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
        """Recall skills for prompt injection, skipping task-specific records."""
        # Over-fetch so filtering out banned skills still leaves up to ``k``.
        records = await self.recall(query, namespace=MemoryNamespace.skill, k=max(k * 3, 6))
        kept = [r for r in records if skill_rejection_reason_from_record(r) is None]
        return kept[:k]

    async def find_knowledge(
        self, query: str, *, k: int = 4, session_id: str | None = None
    ) -> list[MemoryRecord]:
        """Recall long-term memories for prompt injection, skipping task-specific ones.

        A stored ``GOAL: <raw benchmark prompt>`` would re-inject its instance id,
        commit SHA and issue dump into unrelated runs, so it is never surfaced.
        """
        records = await self.recall(
            query, namespace=MemoryNamespace.long_term, k=max(k * 3, 6), session_id=session_id
        )
        kept = [r for r in records if task_specific_reason(r.content or "") is None]
        return kept[:k]

    async def list_skills(self, *, limit: int = 500) -> list[MemoryRecord]:
        """Every learned skill, newest first (no semantic query needed)."""
        return await self._store.recent(MemoryNamespace.skill, limit=limit)

    async def get_skill(self, skill_id: str) -> MemoryRecord | None:
        for record in await self.list_skills(limit=5000):
            if record.id == skill_id:
                return record
        return None

    async def _find_skill_by_slug(self, slug: str) -> MemoryRecord | None:
        """Find an existing general skill by canonical title, for deduplication."""
        if not slug:
            return None
        for record in await self.list_skills(limit=5000):
            name = (record.metadata or {}).get("name") or skill_name_from_content(
                record.content or ""
            )
            if name and skill_slug(str(name)) == slug:
                return record
        return None

    async def save_skill(
        self,
        *,
        name: str,
        description: str,
        steps: list[str],
        tags: list[str] | None = None,
        skill_id: str | None = None,
    ) -> MemoryRecord:
        """Create a skill, or replace an existing one while keeping its id.

        Task-specific candidates are rejected with :class:`SkillRejectedError`.
        General skills are deduplicated by title (an existing skill is updated in
        place rather than appended). Editing deletes and re-inserts the record so
        the same code works for the in-memory store and pgvector (whose ``upsert``
        only inserts).
        """
        clean = sanitize_skill(name=name, description=description, steps=steps, tags=tags)
        if clean is None:
            reason = skill_rejection_reason(name, description, steps)
            log.warning("memory.skill_rejected", reason=reason or "task-specific")
            raise SkillRejectedError(reason or "task-specific skill cannot be saved")
        name = clean["name"]
        description = clean["description"]
        steps = clean["steps"]
        tags = clean["tags"]

        existing = await self.get_skill(skill_id) if skill_id else None
        if skill_id and existing is None:
            raise KeyError(skill_id)
        if existing is None:
            existing = await self._find_skill_by_slug(skill_slug(name))
            if existing is not None:
                log.info("memory.skill_merged", slug=skill_slug(name), id=existing.id)

        content = f"SKILL: {name}\n{description}\nSTEPS:\n" + "\n".join(f"- {s}" for s in steps)
        embedding = existing.embedding if existing is not None else None
        try:
            embedding = (await self._embedder.embed([content]))[0]
        except Exception as exc:  # noqa: BLE001 - keep the old vector if embedding fails
            log.warning("memory.skill_embed_failed", error=str(exc))

        metadata = dict(existing.metadata) if existing is not None else {}
        metadata.update({"name": name, "tags": tags or [], "steps": steps})
        record = MemoryRecord(
            id=existing.id if existing is not None else uuid.uuid4().hex,
            namespace=MemoryNamespace.skill,
            content=content,
            metadata=metadata,
            session_id=existing.session_id if existing is not None else None,
            created_at=existing.created_at if existing is not None else datetime.now(UTC),
            embedding=embedding,
        )
        if existing is not None:
            await self._store.delete([existing.id])
        await self._store.upsert([record])
        log.info("memory.skill_saved", id=record.id, updated=existing is not None)
        return record

    async def purge_task_specific_skills(self) -> int:
        """Delete every stored skill that fails the task-specific guard.

        Returns the number of records removed so callers can persist the change.
        """
        skills = await self.list_skills(limit=100000)
        bad = [r for r in skills if skill_rejection_reason_from_record(r) is not None]
        if bad:
            await self._store.delete([r.id for r in bad])
            for record in bad:
                name = (record.metadata or {}).get("name") or skill_name_from_content(
                    record.content or ""
                )
                log.info("memory.skill_purged", id=record.id, name=str(name)[:120])
        return len(bad)

    async def purge_task_specific_memories(self) -> int:
        """Purge task-specific skills *and* long-term memories (raw task dumps)."""
        removed = await self.purge_task_specific_skills()
        long_term = await self._store.recent(MemoryNamespace.long_term, limit=100000)
        bad = [r for r in long_term if task_specific_reason(r.content or "") is not None]
        if bad:
            await self._store.delete([r.id for r in bad])
            log.info("memory.long_term_purged", count=len(bad))
        return removed + len(bad)

    async def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill by id. Returns False when it is not a known skill."""
        if await self.get_skill(skill_id) is None:
            return False
        await self._store.delete([skill_id])
        log.info("memory.skill_deleted", id=skill_id)
        return True

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
        export = getattr(self._store, "export_records", None)
        return export() if export is not None else []

    def import_records(self, records: list[dict]) -> None:
        import_records = getattr(self._store, "import_records", None)
        if import_records is None:
            return
        # Never resurrect a legacy task-specific skill (or raw task-dump long-term
        # memory) from an old snapshot: the same policy that guards saves and
        # recall also guards what enters the live store.
        kept: list[dict] = []
        dropped = 0
        for record in records:
            namespace = record.get("namespace")
            if namespace == MemoryNamespace.skill.value and (
                skill_rejection_reason_from_dict(record) is not None
            ):
                dropped += 1
                continue
            if namespace == MemoryNamespace.long_term.value and (
                task_specific_reason(str(record.get("content") or "")) is not None
            ):
                dropped += 1
                continue
            kept.append(record)
        if dropped:
            log.info("memory.task_specific_dropped_on_import", count=dropped)
        import_records(kept)


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
