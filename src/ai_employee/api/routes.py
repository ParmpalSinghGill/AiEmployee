"""API routes covering chat, tasks, agents, tools, memory, knowledge and config."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from ai_employee.api.deps import get_employee
from ai_employee.api.schemas import (
    AgentRunRequest,
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    MemoryQueryResponse,
    MemoryRecordRequest,
    QueryRequest,
    QueryResponse,
    TaskRequest,
    TaskResponse,
)
from ai_employee.brains.schemas import Message
from ai_employee.core.employee import AIEmployee
from ai_employee.memory import MemoryNamespace

router = APIRouter()


@router.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


@router.get("/config", tags=["system"])
async def config(emp: AIEmployee = Depends(get_employee)) -> dict:
    s = emp.settings
    provider = s.llm_provider.value
    # Is a usable credential present for the active provider? (local servers don't need one.)
    key_map = {
        "deepseek": s.deepseek_api_key,
        "openai": s.openai_api_key,
        "gemini": s.google_api_key,
        "glm": s.glm_api_key,
        "opencode": s.opencode_api_key,
    }
    needs_key = provider in key_map
    has_key = bool(key_map.get(provider)) if needs_key else True
    return {
        "provider": provider,
        "model": emp.brain.model_name,
        "offline_mock": provider == "mock",
        "needs_api_key": needs_key,
        "has_api_key": has_key,
        "embedding_provider": s.embedding_provider.value,
        "embedding_model": _embedding_model_name(s),
        "reasoning_supported": emp.brain.supports_reasoning,
        "embeddings_supported": emp.embedder.supports_embeddings,
        "environment": s.environment.value,
        "memory_backend": s.memory_backend,
        "use_stored_memory": s.use_stored_memory,
        "approval_required": sorted(s.approval_required),
    }


def _embedding_model_name(s) -> str:
    return {
        "openai": s.openai_embedding_model,
        "gemini": s.gemini_embedding_model,
        "glm": s.glm_embedding_model,
        "local": s.local_embedding_model,
        "mock": "mock",
    }.get(s.embedding_provider.value, "")


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest, emp: AIEmployee = Depends(get_employee)) -> ChatResponse:
    reply = await emp.chat(req.message, session_id=req.session_id)
    return ChatResponse(reply=reply, session_id=req.session_id)


@router.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest, emp: AIEmployee = Depends(get_employee)):
    async def event_gen():
        async for token in emp.brain.stream([Message.user(req.message)]):
            yield {"data": token}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_gen())


@router.post("/task", response_model=TaskResponse, tags=["tasks"])
async def run_task(req: TaskRequest, emp: AIEmployee = Depends(get_employee)) -> TaskResponse:
    result = await emp.run(
        req.goal, session_id=req.session_id, approved_ops=set(req.approved_ops)
    )
    return TaskResponse(
        task_id=uuid.uuid4().hex,
        goal=req.goal,
        answer=result["answer"],
        confidence=result.get("confidence"),
        plan=result.get("plan"),
        steps=result.get("steps", []),
        needs_approval=result.get("needs_approval"),
    )


@router.get("/agents", tags=["agents"])
async def list_agents(emp: AIEmployee = Depends(get_employee)) -> dict:
    return {"agents": emp.agents.catalog()}


@router.post("/agents/run", tags=["agents"])
async def run_agent(req: AgentRunRequest, emp: AIEmployee = Depends(get_employee)) -> dict:
    if req.agent not in emp.agents:
        raise HTTPException(404, f"unknown agent: {req.agent}")
    return await emp.ask_agent(req.agent, req.task, session_id=req.session_id)


@router.get("/tools", tags=["tools"])
async def list_tools(emp: AIEmployee = Depends(get_employee)) -> dict:
    return {
        "tools": [
            {"name": t.name, "category": t.category, "description": t.description,
             "approval_key": t.approval_key}
            for t in emp.tools.all()
        ]
    }


@router.post("/knowledge/ingest", response_model=IngestResponse, tags=["knowledge"])
async def ingest(req: IngestRequest, emp: AIEmployee = Depends(get_employee)) -> IngestResponse:
    report = await emp.knowledge.ingest(req.source)
    return IngestResponse(
        documents=report.documents, chunks=report.chunks, sources=report.sources
    )


@router.post("/knowledge/query", response_model=QueryResponse, tags=["knowledge"])
async def query(req: QueryRequest, emp: AIEmployee = Depends(get_employee)) -> QueryResponse:
    res = await emp.knowledge.answer(req.question, k=req.k)
    return QueryResponse(answer=res.answer, citations=res.citations, used_chunks=res.used_chunks)


@router.post("/memory", tags=["memory"])
async def record_memory(
    req: MemoryRecordRequest, emp: AIEmployee = Depends(get_employee)
) -> dict:
    try:
        ns = MemoryNamespace(req.namespace)
    except ValueError as exc:
        raise HTTPException(400, f"invalid namespace: {req.namespace}") from exc
    rec = await emp.memory.record(req.content, namespace=ns, session_id=req.session_id)
    return {"id": rec.id, "namespace": ns.value}


@router.get("/memory/search", response_model=MemoryQueryResponse, tags=["memory"])
async def search_memory(
    q: str, namespace: str = "long_term", k: int = 5,
    emp: AIEmployee = Depends(get_employee),
) -> MemoryQueryResponse:
    try:
        ns = MemoryNamespace(namespace)
    except ValueError as exc:
        raise HTTPException(400, f"invalid namespace: {namespace}") from exc
    hits = await emp.memory.recall(q, namespace=ns, k=k)
    return MemoryQueryResponse(
        results=[
            {"id": h.id, "content": h.content, "score": h.score, "metadata": h.metadata}
            for h in hits
        ]
    )


@router.get("/workflows", tags=["workflows"])
async def workflows() -> dict:
    """Describe the orchestration pipeline (static topology)."""
    return {
        "pipeline": ["plan", "execute*", "verify", "synthesize", "reflect"],
        "description": "Goal → decomposition → agent execution w/ verification → synthesis → memory.",
    }
