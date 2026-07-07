"""Web dashboard: routes for the task UI plus its JSON API.

The frontend (served from ``api/static``) is a dependency-free SPA that creates
tasks, opens a tab per task/session, renders the live agent/tool flow, streams
logs, shows results, and submits follow-up questions.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_employee.api.deps import get_employee, rebuild_employee
from ai_employee.config import LLMProvider, get_settings
from ai_employee.core.employee import AIEmployee
from ai_employee.observability import get_task_manager

router = APIRouter()
_STATIC = Path(__file__).parent / "static"

# Per-provider settings attributes + a few suggested models for the picker.
_CHAT_MODEL_ATTR = {
    "deepseek": "deepseek_chat_model",
    "openai": "openai_chat_model",
    "gemini": "gemini_chat_model",
    "glm": "glm_chat_model",
    "opencode": "opencode_chat_model",
    "local": "local_chat_model",
}
_KEY_ATTR = {
    "deepseek": "deepseek_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "glm": "glm_api_key",
    "opencode": "opencode_api_key",
    "local": "local_api_key",
}
_SUGGESTED = {
    "mock": ["mock-1"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "glm": ["glm-4-plus", "glm-4", "glm-4-air"],
    "opencode": ["deepseek-v4-pro", "claude-opus-4-8", "gpt-5.5", "glm-5.2"],
    "local": ["qwen3:8b"],
}


class NewTaskRequest(BaseModel):
    goal: str = Field(min_length=1)
    approved_ops: list[str] = Field(default_factory=list)
    mode: str = "task"  # "task", "research", or "download" (direct URL downloader)


class BrainChangeRequest(BaseModel):
    provider: str
    model: str | None = None
    api_key: str | None = None  # optional, in-memory only (not written to .env)
    save_default: bool = False


class BrainCreditsRequest(BaseModel):
    provider: str
    api_key: str | None = None  # optional override (typed in picker, else stored key)


class CodingBackendRequest(BaseModel):
    backend: str
    save_default: bool = False


class MemorySettingsRequest(BaseModel):
    use_stored_memory: bool
    save_default: bool = False


class KeyValueRequest(BaseModel):
    value: str = Field(min_length=1)


class KeyTestRequest(BaseModel):
    value: str | None = None  # test this value; falls back to the stored key


_CODING_BACKENDS = ["brain", "claude", "codex", "cursor"]


def _static_version() -> int:
    """Bump when app.js / styles.css change so browsers reload the dashboard assets."""
    mtimes = []
    for name in ("app.js", "styles.css"):
        p = _STATIC / name
        if p.exists():
            mtimes.append(int(p.stat().st_mtime))
    return max(mtimes) if mtimes else 0


# ── Pages ────────────────────────────────────────────────────────────────────
@router.get("/ui", include_in_schema=False)
async def ui_index() -> HTMLResponse:
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    v = _static_version()
    html = html.replace('href="/static/styles.css"', f'href="/static/styles.css?v={v}"')
    html = html.replace('src="/static/app.js"', f'src="/static/app.js?v={v}"')
    return HTMLResponse(html)


# ── JSON API ─────────────────────────────────────────────────────────────────
@router.post("/ui/api/tasks", tags=["ui"])
async def create_task(req: NewTaskRequest, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Start a brand-new task (creates a new session/tab)."""
    tm = get_task_manager()
    task = tm.start(emp, req.goal, approved_ops=set(req.approved_ops), mode=req.mode)
    return {"task_id": task.id, "session_id": task.session_id}


@router.post("/ui/api/sessions/{session_id}/followup", tags=["ui"])
async def followup(
    session_id: str, req: NewTaskRequest, emp: AIEmployee = Depends(get_employee)
) -> dict:
    """Add a follow-up run to an existing session/tab (shares memory & context)."""
    tm = get_task_manager()
    if not tm.session_runs(session_id):
        raise HTTPException(404, "unknown session")
    task = tm.start(
        emp, req.goal, session_id=session_id, approved_ops=set(req.approved_ops), mode=req.mode
    )
    return {"task_id": task.id, "session_id": session_id}


class ApproveTaskRequest(BaseModel):
    decision: str | None = None  # proceed_without | abort | provide_local_path | provide_url
    local_path: str | None = None
    resource_url: str | None = None
    note: str | None = None
    approve_all_similar: bool = True  # pre-approve this op + related (e.g. all downloads in run)


@router.post("/ui/api/tasks/{task_id}/approve", tags=["ui"])
async def approve_task(
    task_id: str,
    req: ApproveTaskRequest | None = None,
    emp: AIEmployee = Depends(get_employee),
) -> dict:
    """Approve a paused task and resume it *in place* — the run continues in the
    same tab (no new run), with the approved operation now permitted."""
    tm = get_task_manager()
    decision = None
    if req:
        decision = {
            k: v
            for k, v in {
                "decision": req.decision,
                "local_path": req.local_path,
                "resource_url": req.resource_url,
                "note": req.note,
            }.items()
            if v
        } or None
    task = tm.approve(
        emp,
        task_id,
        resource_decision=decision,
        approve_all_similar=req.approve_all_similar if req else True,
    )
    if task is None:
        raise HTTPException(409, "task is not awaiting approval or waiting for user input")
    return {"task_id": task.id, "session_id": task.session_id}


@router.get("/ui/api/brain/options", tags=["ui"])
async def brain_options() -> dict:
    """Available providers + suggested models. For local, queries the running
    Ollama/OpenAI-compatible server so its installed models show up live."""
    from ai_employee.config.ui_preferences import read_preferences

    s = get_settings()
    models = {k: list(v) for k, v in _SUGGESTED.items()}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(s.local_base_url.rstrip("/") + "/models")
        if r.is_success:
            ids = [m["id"] for m in r.json().get("data", [])]
            if ids:
                models["local"] = ids
    except Exception:  # noqa: BLE001 - local server may be offline; keep suggestion
        pass
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(s.opencode_base_url.rstrip("/") + "/models")
        if r.is_success:
            ids = [m["id"] for m in r.json().get("data", [])]
            if ids:
                models["opencode"] = ids
    except Exception:  # noqa: BLE001 - gateway unreachable; keep suggestion
        pass
    return {
        "providers": [p.value for p in LLMProvider],
        "models": models,
        "current": {"provider": s.llm_provider.value, "model": _current_model(s)},
        "saved": read_preferences(s),
    }


@router.post("/ui/api/brain", tags=["ui"])
async def change_brain(req: BrainChangeRequest) -> dict:
    """Switch the active brain (provider + model) at runtime — no restart."""
    if req.provider not in {p.value for p in LLMProvider}:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    s = get_settings()
    s.llm_provider = LLMProvider(req.provider)
    if req.model and req.provider in _CHAT_MODEL_ATTR:
        setattr(s, _CHAT_MODEL_ATTR[req.provider], req.model)
    if req.api_key and req.provider in _KEY_ATTR:
        setattr(s, _KEY_ATTR[req.provider], req.api_key)
    if req.save_default:
        from ai_employee.config.ui_preferences import save_brain_default

        save_brain_default(s, req.provider, req.model)
    # Embeddings are intentionally left as configured (changing them can cause
    # dimension mismatches); the mock embedder keeps memory/RAG working regardless.
    await rebuild_employee()
    emp = get_employee()
    return {
        "provider": emp.brain.name,
        "model": emp.brain.model_name,
        "embedding_provider": s.embedding_provider.value,
        "saved_default": req.save_default,
    }


@router.post("/ui/api/brain/credits", tags=["ui"])
async def brain_credits(req: BrainCreditsRequest) -> dict:
    """Remaining credit/balance for a hosted brain provider (not local/mock)."""
    from ai_employee.config.credits_checker import check_brain_credits

    if req.provider not in {p.value for p in LLMProvider}:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    return await check_brain_credits(get_settings(), req.provider, req.api_key)


def _current_model(s) -> str:
    attr = _CHAT_MODEL_ATTR.get(s.llm_provider.value)
    return getattr(s, attr) if attr else s.llm_provider.value


# ── API key management ───────────────────────────────────────────────────────
@router.get("/ui/api/keys", tags=["ui"])
async def list_api_keys() -> dict:
    """Managed API keys with *masked* values (raw secrets are never returned)."""
    from ai_employee.config.secrets_store import list_keys

    return {"keys": list_keys(get_settings())}


@router.put("/ui/api/keys/{name}", tags=["ui"])
async def set_api_key(name: str, req: KeyValueRequest) -> dict:
    """Add or update a key. LLM-provider keys rebuild the brain to take effect."""
    from ai_employee.config.secrets_store import set_key, spec_for

    try:
        spec = spec_for(name)
    except KeyError:
        raise HTTPException(404, f"unknown key: {name}") from None
    set_key(get_settings(), name, req.value.strip())
    if spec.requires_rebuild:
        await rebuild_employee()
    return {"ok": True, "name": name, "rebuilt": spec.requires_rebuild}


@router.post("/ui/api/keys/{name}/test", tags=["ui"])
async def test_api_key(name: str, req: KeyTestRequest) -> dict:
    """Live-check a key (the typed value, or the stored one). Returns ok + message."""
    from ai_employee.config.key_tester import test_key
    from ai_employee.config.secrets_store import spec_for

    try:
        spec_for(name)
    except KeyError:
        raise HTTPException(404, f"unknown key: {name}") from None
    return await test_key(get_settings(), name, req.value)


@router.post("/ui/api/keys/{name}/credits", tags=["ui"])
async def check_key_credits(name: str, req: KeyTestRequest) -> dict:
    """Check remaining credit/balance for an LLM key (typed value, or the stored
    one). Returns the real balance where a provider exposes one, otherwise a
    dashboard link. Never raises — the outcome is always in the payload."""
    from ai_employee.config.credits_checker import check_credits
    from ai_employee.config.secrets_store import spec_for

    try:
        spec_for(name)
    except KeyError:
        raise HTTPException(404, f"unknown key: {name}") from None
    return await check_credits(get_settings(), name, req.value)


@router.get("/ui/api/usage", tags=["ui"])
async def usage(start: str | None = None, end: str | None = None) -> dict:
    """Token usage per brain (provider), optionally within an inclusive date range
    (``YYYY-MM-DD``). Powers the Usage dashboard."""
    from ai_employee.observability.usage_ledger import query

    return query(get_settings().usage_ledger_file, start=start, end=end)


@router.delete("/ui/api/keys/{name}", tags=["ui"])
async def delete_api_key(name: str) -> dict:
    """Remove a key. LLM-provider keys rebuild the brain to take effect."""
    from ai_employee.config.secrets_store import delete_key, spec_for

    try:
        spec = spec_for(name)
    except KeyError:
        raise HTTPException(404, f"unknown key: {name}") from None
    delete_key(get_settings(), name)
    if spec.requires_rebuild:
        await rebuild_employee()
    return {"ok": True, "name": name, "rebuilt": spec.requires_rebuild}


@router.get("/ui/api/coding/options", tags=["ui"])
async def coding_options() -> dict:
    """Coding-agent backends + which external CLIs are actually installed."""
    from ai_employee.config.ui_preferences import read_preferences
    from ai_employee.integrations.code_agents import cli_available

    s = get_settings()
    return {
        "backends": _CODING_BACKENDS,
        "current": s.coding_backend,
        "saved": read_preferences(s),
        "available": {
            "brain": True,
            "claude": cli_available("claude"),
            "codex": cli_available("codex"),
            "cursor": cli_available("cursor"),
        },
        "labels": {
            "brain": "Current brain (LLM)",
            "claude": "Claude Code (CLI)",
            "codex": "Codex (CLI)",
            "cursor": "Cursor Agent (CLI)",
        },
    }


@router.post("/ui/api/coding", tags=["ui"])
async def set_coding_backend(req: CodingBackendRequest) -> dict:
    """Switch which backend the coding agent uses (brain / claude / codex / cursor).

    Takes effect immediately — the coding agent reads this per run, so no rebuild.
    """
    if req.backend not in _CODING_BACKENDS:
        raise HTTPException(400, f"unknown coding backend: {req.backend}")
    s = get_settings()
    s.coding_backend = req.backend
    if req.save_default:
        from ai_employee.config.ui_preferences import save_coding_default

        save_coding_default(s, req.backend)
    return {"backend": s.coding_backend, "saved_default": req.save_default}


@router.get("/ui/api/sessions", tags=["ui"])
async def list_sessions() -> dict:
    return {"sessions": get_task_manager().list_sessions()}


@router.get("/ui/api/sessions/{session_id}", tags=["ui"])
async def get_session(session_id: str) -> dict:
    """Full session (meta + every run's snapshot) — used to open/resume a saved task."""
    detail = get_task_manager().session_detail(session_id)
    if detail is None:
        raise HTTPException(404, "unknown session")
    return detail


@router.delete("/ui/api/sessions", tags=["ui"])
async def delete_all_sessions(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Delete every saved task and its session-scoped recall memory."""
    tm = get_task_manager()
    ids = [s["id"] for s in tm.list_sessions()]
    removed_memory = 0
    for sid in ids:
        mem = await emp.clear_session_memory(sid)
        removed_memory += int(mem.get("records_removed") or 0)
        tm.delete_session(sid)
    return {"deleted": len(ids), "session_ids": ids, "memory_records_removed": removed_memory}


@router.delete("/ui/api/sessions/{session_id}/runs/{task_id}", tags=["ui"])
async def delete_followup(session_id: str, task_id: str) -> dict:
    """Delete a single follow-up run. The session's initial task is kept."""
    tm = get_task_manager()
    if not tm.delete_run(session_id, task_id):
        raise HTTPException(404, "unknown session, run, or cannot delete the initial task")
    return {"deleted": task_id, "session_id": session_id}


@router.delete("/ui/api/sessions/{session_id}", tags=["ui"])
async def delete_session(session_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    tm = get_task_manager()
    if not tm.session_runs(session_id):
        raise HTTPException(404, "unknown session")
    mem = await emp.clear_session_memory(session_id)
    tm.delete_session(session_id)
    return {
        "deleted": session_id,
        "memory_records_removed": mem.get("records_removed", 0),
    }


@router.delete("/ui/api/sessions/{session_id}/memory", tags=["ui"])
async def clear_session_memory(session_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Clear recall memory for one task/session. Saved run history is kept."""
    tm = get_task_manager()
    if not tm.session_runs(session_id):
        raise HTTPException(404, "unknown session")
    return await emp.clear_session_memory(session_id)


@router.delete("/ui/api/memory", tags=["ui"])
async def clear_memories(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Remove all stored memories (long-term, skills, conversation, knowledge, …).

    Saved task history on disk is kept; only the recall context is wiped.
    A backup is written so you can restore the previous memory later.
    """
    return await emp.clear_memories()


@router.get("/ui/api/memory/settings", tags=["ui"])
async def get_memory_settings(emp: AIEmployee = Depends(get_employee)) -> dict:
    from ai_employee.config.ui_preferences import read_preferences

    s = get_settings()
    status = emp.memory_status()
    saved = read_preferences(s).get("use_stored_memory")
    status["saved_default"] = saved if isinstance(saved, bool) else None
    return status


@router.post("/ui/api/memory/settings", tags=["ui"])
async def set_memory_settings(req: MemorySettingsRequest) -> dict:
    """Toggle whether agents recall and persist stored memory."""
    from ai_employee.config.ui_preferences import save_memory_default

    s = get_settings()
    s.use_stored_memory = req.use_stored_memory
    if req.save_default:
        save_memory_default(s, req.use_stored_memory)
    return {
        "use_stored_memory": s.use_stored_memory,
        "saved_default": req.save_default,
    }


@router.post("/ui/api/memory/restore", tags=["ui"])
async def restore_memories(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Reload memory from the archive saved before the last clear."""
    result = await emp.restore_memories()
    if not result.get("restored"):
        raise HTTPException(404, "no memory backup to restore")
    return result


@router.post("/ui/api/tasks/{task_id}/cancel", tags=["ui"])
async def cancel_task(task_id: str) -> dict:
    """Stop a running/pending task. Idempotent: cancelling an already-finished
    task just reports ``cancelled: false``."""
    tm = get_task_manager()
    if tm.get_task(task_id) is None:
        raise HTTPException(404, "unknown task")
    return {"task_id": task_id, "cancelled": tm.cancel(task_id)}


@router.post("/ui/api/tasks/{task_id}/resume", tags=["ui"])
async def resume_task(task_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Resume an interrupted run from its last saved workflow checkpoint."""
    tm = get_task_manager()
    if tm.get_task(task_id) is None:
        raise HTTPException(404, "unknown task")
    task = tm.resume(emp, task_id)
    if task is None:
        raise HTTPException(409, "task is not resumable")
    return {"task_id": task.id, "session_id": task.session_id}


@router.get("/ui/api/tasks/{task_id}", tags=["ui"])
async def get_task(task_id: str, since: int = 0) -> dict:
    tm = get_task_manager()
    task = tm.get_task(task_id)
    if task is None:
        raise HTTPException(404, "unknown task")
    snap = task.snapshot(since=since)
    snap["session_title"] = tm.session_title(task.session_id)
    return snap
