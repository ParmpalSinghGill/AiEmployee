"""Memory primitives and the VectorStore interface."""

from __future__ import annotations

import abc
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryNamespace(str, Enum):
    short_term = "short_term"
    long_term = "long_term"
    conversation = "conversation"
    task = "task"
    knowledge = "knowledge"
    project = "project"
    skill = "skill"


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


class MemoryRecord(BaseModel):
    """A single stored memory."""

    id: str = Field(default_factory=_uuid)
    namespace: MemoryNamespace = MemoryNamespace.long_term
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    # Optional scoping so memories can be filtered by who/what they belong to.
    session_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    # Populated by search results, not persisted.
    score: float | None = None


class VectorStore(abc.ABC):
    """Pluggable backing store for memory records (semantic + recency access)."""

    @abc.abstractmethod
    async def upsert(self, records: list[MemoryRecord]) -> None: ...

    @abc.abstractmethod
    async def search(
        self,
        namespace: MemoryNamespace,
        query_embedding: list[float],
        *,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[MemoryRecord]: ...

    @abc.abstractmethod
    async def recent(
        self,
        namespace: MemoryNamespace,
        *,
        limit: int = 20,
        session_id: str | None = None,
    ) -> list[MemoryRecord]: ...

    @abc.abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    @abc.abstractmethod
    async def clear(self, namespace: MemoryNamespace | None = None) -> None: ...

    async def aclose(self) -> None:
        return None
