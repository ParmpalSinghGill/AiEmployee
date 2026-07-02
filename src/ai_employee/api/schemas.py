"""Request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class TaskRequest(BaseModel):
    goal: str = Field(description="The high-level goal for the employee to accomplish.")
    session_id: str = "default"
    approved_ops: list[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    task_id: str
    goal: str
    answer: str
    confidence: float | None = None
    plan: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    needs_approval: dict[str, Any] | None = None


class AgentRunRequest(BaseModel):
    agent: str
    task: str
    session_id: str = "default"


class IngestRequest(BaseModel):
    source: str = Field(description="A file path, directory, URL, or raw text to ingest.")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    sources: list[str]


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    used_chunks: int


class MemoryRecordRequest(BaseModel):
    content: str
    namespace: str = "long_term"
    session_id: str | None = None


class MemoryQueryResponse(BaseModel):
    results: list[dict[str, Any]]
