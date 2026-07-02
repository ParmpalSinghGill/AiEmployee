"""Memory layer.

Provides the AI Employee's persistence of experience across these memory types
(modeled as namespaces over a vector store + recency buffers):

  short_term     – rolling conversation buffer (per session)
  long_term      – durable semantic memory
  conversation   – full message history
  task           – goals, plans, outcomes
  knowledge      – RAG-ingested facts
  project        – project-scoped context
  skill          – reusable workflows/skills learned over time

Backed by an in-memory store offline, or PostgreSQL + pgvector in production —
selected by :func:`build_memory_manager` without changing caller code.
"""

from ai_employee.memory.base import MemoryNamespace, MemoryRecord, VectorStore
from ai_employee.memory.manager import MemoryManager, build_memory_manager

__all__ = [
    "MemoryRecord",
    "MemoryNamespace",
    "VectorStore",
    "MemoryManager",
    "build_memory_manager",
]
