"""ORM models for tasks, conversations, memory, knowledge and traces.

The vector columns use pgvector. ``EMBEDDING_DIM`` must match the embedding model
chosen in configuration (see Alembic migration / .env).
"""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ai_employee.config import get_settings
from ai_employee.database.base import Base, Timestamps, UUIDPrimaryKey

_DIM = get_settings().embedding_dim


class Task(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "tasks"

    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    steps: Mapped[list[TaskStep]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class TaskStep(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "task_steps"

    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    order: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str] = mapped_column(Text)
    agent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    output: Mapped[str | None] = mapped_column(Text, nullable=True)

    task: Mapped[Task] = relationship(back_populates="steps")


class Message(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "messages"

    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class MemoryItem(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "memories"

    namespace: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_DIM), nullable=True)

    __table_args__ = (
        Index(
            "ix_memories_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class KnowledgeDocument(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "knowledge_documents"

    source: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str] = mapped_column(String(32), default="text")
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    chunks: Mapped[list[KnowledgeChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class KnowledgeChunk(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "knowledge_chunks"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_DIM), nullable=True)

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class AgentEvent(Base, UUIDPrimaryKey, Timestamps):
    """Append-only trace of agent/tool activity for observability & learning."""

    __tablename__ = "agent_events"

    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    agent: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class Resource(Base, UUIDPrimaryKey, Timestamps):
    """Optional Postgres mirror of the filesystem resource catalog.

    The filesystem catalog is the source of truth (so the framework runs offline);
    a deployed app can mirror rows here for cross-task SQL queries and reporting.
    ``visibility`` / ``owner_task`` / ``content_hash`` carry the same meaning as in
    :class:`ai_employee.resources.models.ResourceRecord`.
    """

    __tablename__ = "resources"

    name: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="file")
    visibility: Mapped[str] = mapped_column(String(16), default="private", index=True)
    owner_task: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    owner_user: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content_hash: Mapped[str] = mapped_column(String(80), index=True)
    hash_kind: Mapped[str] = mapped_column(String(16), default="content")
    size: Mapped[int] = mapped_column(Integer, default=0)
    logical_path: Mapped[str] = mapped_column(String(1024), default="")
    source: Mapped[str] = mapped_column(String(32), default="local")
    url: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
