"""Web dashboard: routes for the task UI plus its JSON API.

The frontend (served from ``api/static``) is a dependency-free SPA that creates
tasks, opens a tab per task/session, renders the live agent/tool flow, streams
logs, shows results, and submits follow-up questions.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ai_employee.api.deps import get_employee, rebuild_employee
from ai_employee.api.users import CurrentUser, get_current_user
from ai_employee.api.workspace_files import (
    collect_deliverables,
    guess_media_type,
    resolve_workspace_path,
    sample_media,
)
from ai_employee.config import LLMProvider, get_settings
from ai_employee.core.employee import AIEmployee
from ai_employee.memory import MemoryRecord
from ai_employee.memory.skill_policy import SkillRejectedError
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
    "gemini": [
        # Current chat lineup: newest stable Flash first, then 2.5 GA and previews.
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-pro-latest",
        "gemini-flash-latest",
    ],
    "glm": ["glm-4-plus", "glm-4", "glm-4-air"],
    "opencode": ["deepseek-v4-pro", "claude-opus-4-8", "gpt-5.5", "glm-5.2"],
    "local": ["qwen3:8b"],
}


class NewTaskRequest(BaseModel):
    goal: str = Field(min_length=1)
    approved_ops: list[str] = Field(default_factory=list)
    mode: str = "task"  # "task", "research", "plan", or "download"
    auto_approve: bool = False  # never pause for gated ops; approve everything
    token_budget: int | None = None  # hard token cap for this run (None → global)


class ScheduleRequest(BaseModel):
    name: str = ""
    goal: str = Field(min_length=1)
    type: str  # "recurring" | "once" | "gmail_watch"
    auto_approve: bool = False
    at: str | None = None  # recurring: "HH:MM"
    days: list[int] | None = None  # recurring: weekday mask (0=Mon)
    every_minutes: int | None = None  # recurring alternative
    run_at: str | None = None  # once: ISO datetime
    query: str | None = None  # gmail_watch: Gmail search query
    interval_minutes: int | None = None  # gmail_watch poll interval


class SchedulePatchRequest(BaseModel):
    enabled: bool | None = None
    name: str | None = None
    goal: str | None = None
    auto_approve: bool | None = None
    at: str | None = None
    days: list[int] | None = None
    every_minutes: int | None = None
    run_at: str | None = None
    query: str | None = None
    interval_minutes: int | None = None


class PoliciesRequest(BaseModel):
    policies: dict[str, dict] = Field(default_factory=dict)


class BrainChangeRequest(BaseModel):
    provider: str
    model: str | None = None
    api_key: str | None = None  # optional, in-memory only (not written to .env)
    save_default: bool = False


class BrainCreditsRequest(BaseModel):
    provider: str
    api_key: str | None = None  # optional override (typed in picker, else stored key)


class ModelCheckRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None  # optional override (typed in picker, else stored key)


class CodingBackendRequest(BaseModel):
    backend: str
    save_default: bool = False


class MemorySettingsRequest(BaseModel):
    use_stored_memory: bool
    save_default: bool = False


class SkillRequest(BaseModel):
    """A learned skill, as edited in the 💡 Skills modal."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=8000)
    steps: list[str] = Field(default_factory=list, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=30)


class ParallelSettingsRequest(BaseModel):
    parallel_steps: bool
    max_parallel_steps: int | None = None
    save_default: bool = False


class AgentLimitsRequest(BaseModel):
    """Per-agent iteration/loop budgets edited from the 🎚 Limits modal."""

    max_iterations: int = 0  # 0 => each agent's built-in default
    max_iterations_by_name: str = ""  # e.g. "coding=24,research=16"
    loop_extensions_max: int = 2  # extra windows when the loop check says "progressing"
    loop_extensions_by_name: str = ""  # e.g. "coding=4"
    loop_check_enabled: bool = True
    # Explain-trace text caps (chars). 0 = keep everything (no truncation).
    explain_max_chars: int = 0
    explain_tool_max_chars: int = 0
    save_default: bool = False


class MCPServerRequest(BaseModel):
    params: dict[str, str] = Field(default_factory=dict)
    require_approval: bool | None = None


class MCPToggleRequest(BaseModel):
    enabled: bool


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


# ── Accounts (optional — default is a no-account "TestN" guest) ──────────────
@router.get("/ui/api/me", tags=["ui"])
async def me(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Who am I? Mints a guest identity (Test1, Test2, …) on the first visit.

    Guests are never stored server-side; creating a real account happens by
    signing in with Google (needs the operator's OAuth client file)."""
    from ai_employee.integrations.google_oauth import client_config

    return {
        "id": user.id,
        "name": user.name,
        "kind": user.kind,
        "email": user.email,
        "google_signin_available": client_config(get_settings()) is not None,
    }


@router.post("/ui/api/auth/google", tags=["ui"])
async def auth_google_begin(request: Request) -> dict:
    """Begin the sign-in-with-Google flow; returns the consent URL to open."""
    from ai_employee.integrations.google_oauth import begin_signin

    redirect_uri = str(request.base_url).rstrip("/") + "/ui/api/auth/google/callback"
    try:
        return {"auth_url": begin_signin(get_settings(), redirect_uri)}
    except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
        raise HTTPException(502, str(exc)) from exc


@router.get("/ui/api/auth/google/callback", tags=["ui"], include_in_schema=False)
async def auth_google_callback(request: Request) -> HTMLResponse:
    """Google redirect target: prove the identity, create/refresh the account,
    and set the signed session cookie in this browser."""
    from ai_employee.api.users import set_user_cookie, upsert_google_user
    from ai_employee.integrations.google_oauth import complete_signin

    q = request.query_params
    if q.get("error"):
        return _oauth_result_page(False, f"Sign-in was denied: {q.get('error')}")
    code, state = q.get("code"), q.get("state")
    if not code or not state:
        return _oauth_result_page(False, "Missing code/state in the callback URL.")
    s = get_settings()
    try:
        profile = await complete_signin(s, state, code)
        user = upsert_google_user(s, profile)
    except Exception as exc:  # noqa: BLE001 - show the failure in the popup
        return _oauth_result_page(False, str(exc))
    page = _oauth_result_page(True, f"Signed in as {user.email}. Your tasks are now saved.")
    set_user_cookie(s, page, user)
    return page


@router.post("/ui/api/auth/logout", tags=["ui"])
async def auth_logout(response: Response) -> dict:
    """Sign out: drop the session cookie (the next visit becomes a fresh guest)."""
    from ai_employee.api.users import clear_user_cookie

    clear_user_cookie(response)
    return {"ok": True}


def _require_session(tm, session_id: str, user: CurrentUser):
    """The session's runs, or 404 when it doesn't exist / belongs to someone else."""
    if not tm.session_runs(session_id) or not tm.owned_by(session_id, user.id):
        raise HTTPException(404, "unknown session")
    return tm.session_runs(session_id)


def _require_task(tm, task_id: str, user: CurrentUser):
    task = tm.get_task(task_id)
    if task is None or not tm.owned_by(task.session_id, user.id):
        raise HTTPException(404, "unknown task")
    return task


# ── JSON API ─────────────────────────────────────────────────────────────────
@router.post("/ui/api/tasks", tags=["ui"])
async def create_task(
    req: NewTaskRequest,
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Start a brand-new task (creates a new session/tab, owned by the caller)."""
    tm = get_task_manager()
    task = tm.start(
        emp,
        req.goal,
        approved_ops=set(req.approved_ops),
        mode=req.mode,
        auto_approve=req.auto_approve,
        token_budget=req.token_budget,
        owner={"id": user.id, "name": user.name, "guest": user.is_guest},
    )
    return {"task_id": task.id, "session_id": task.session_id}


@router.post("/ui/api/sessions/{session_id}/followup", tags=["ui"])
async def followup(
    session_id: str,
    req: NewTaskRequest,
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Add a follow-up run to an existing session/tab (shares memory & context)."""
    tm = get_task_manager()
    _require_session(tm, session_id, user)
    task = tm.start(
        emp,
        req.goal,
        session_id=session_id,
        approved_ops=set(req.approved_ops),
        mode=req.mode,
        auto_approve=req.auto_approve,
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
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Approve a paused task and resume it *in place* — the run continues in the
    same tab (no new run), with the approved operation now permitted."""
    tm = get_task_manager()
    _require_task(tm, task_id, user)
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


@router.post("/ui/api/brain/models/check", tags=["ui"])
async def brain_model_check(req: ModelCheckRequest) -> dict:
    """Probe the provider's live model list and report whether ``model`` is usable.

    A documented model can still be unavailable to a given key (tier limits,
    regional rollout, preview gating), so the picker asks the server instead of
    assuming its suggestion list is callable.
    """
    from ai_employee.config.model_checker import check_model

    if req.provider not in {p.value for p in LLMProvider}:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    if not (req.model or "").strip():
        raise HTTPException(400, "model is required")
    return await check_model(get_settings(), req.provider, req.model.strip(), req.api_key)


@router.post("/ui/api/brain/models", tags=["ui"])
async def brain_model_list(req: BrainCreditsRequest) -> dict:
    """The provider's live model list (for refreshing the picker on demand)."""
    from ai_employee.config.model_checker import list_models

    if req.provider not in {p.value for p in LLMProvider}:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    return await list_models(get_settings(), req.provider, req.api_key)


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


# ── MCP connectors ───────────────────────────────────────────────────────────
def _mask_param(value: str) -> str:
    return "••••" + value[-4:] if len(value) > 4 else "••••"


@router.get("/ui/api/mcp/options", tags=["ui"])
async def mcp_options() -> dict:
    """The connector catalog merged with each connector's saved state.

    Secret parameter values are masked — the raw value never leaves the server.
    """
    from ai_employee.tools.mcp import CONNECTOR_CATALOG, load_servers, runtime_available

    servers = load_servers(get_settings())
    connectors = []
    for spec in CONNECTOR_CATALOG:
        entry = servers.get(spec.id) or {}
        saved_params = entry.get("params") or {}
        params = []
        for p in spec.params:
            value = saved_params.get(p.name, "")
            params.append(
                {
                    "name": p.name,
                    "label": p.label,
                    "description": p.description,
                    "secret": p.secret,
                    "required": p.required,
                    "default": p.default,
                    "configured": bool(value),
                    "masked": (_mask_param(value) if p.secret else value) if value else None,
                }
            )
        connectors.append(
            {
                "id": spec.id,
                "label": spec.label,
                "description": spec.description,
                "transport": spec.transport,
                "runtime": spec.runtime,
                "docs": spec.docs,
                "params": params,
                "auth": spec.auth,
                "authorized": bool((entry.get("auth") or {}).get("access_token")),
                "configured": spec.id in servers,
                "enabled": bool(entry.get("enabled")),
                "require_approval": bool(entry.get("require_approval")),
                "tools": [t["name"] for t in entry.get("tools") or []],
                "error": entry.get("error"),
                "connected_at": entry.get("connected_at"),
            }
        )
    return {"connectors": connectors, "runtime": runtime_available()}


@router.put("/ui/api/mcp/servers/{server_id}", tags=["ui"])
async def add_mcp_server(
    server_id: str, req: MCPServerRequest, emp: AIEmployee = Depends(get_employee)
) -> dict:
    """Enable a connector: save its params, connect, discover tools, go live.

    Blank submitted values keep the previously stored ones, so re-connecting
    never requires re-typing a token.
    """
    from ai_employee.tools.mcp import connect_server, connector_for, load_servers, sync_registry

    s = get_settings()
    try:
        connector_for(server_id)
    except KeyError:
        raise HTTPException(404, f"unknown MCP connector: {server_id}") from None
    stored = (load_servers(s).get(server_id) or {}).get("params") or {}
    merged = {**stored, **{k: v for k, v in req.params.items() if v.strip()}}
    result = await connect_server(
        s, server_id, params=merged, require_approval=req.require_approval
    )
    result["registered_tools"] = sync_registry(s, emp.tools)
    return result


@router.post("/ui/api/mcp/servers/{server_id}/authorize", tags=["ui"])
async def authorize_mcp_server(server_id: str, request: Request) -> dict:
    """Begin the browser OAuth flow for a connector; returns the consent URL.

    The provider redirects back to ``/ui/api/mcp/oauth/callback`` once the
    user approves, which finishes the token exchange and connects the server.
    """
    from ai_employee.tools.mcp import connector_for, render_server, save_server
    from ai_employee.tools.mcp.oauth import begin_authorization

    s = get_settings()
    try:
        spec = connector_for(server_id)
    except KeyError:
        raise HTTPException(404, f"unknown MCP connector: {server_id}") from None
    if spec.auth != "oauth":
        raise HTTPException(400, f"connector '{server_id}' does not use browser authorization")
    save_server(s, server_id, enabled=True)  # ensure an entry exists to attach tokens to
    redirect_uri = str(request.base_url).rstrip("/") + "/ui/api/mcp/oauth/callback"
    try:
        url = await begin_authorization(s, server_id, render_server(spec, {}).url, redirect_uri)
    except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
        raise HTTPException(502, f"could not start authorization: {exc}") from exc
    return {"server": server_id, "auth_url": url}


@router.get("/ui/api/mcp/oauth/callback", tags=["ui"], include_in_schema=False)
async def mcp_oauth_callback(
    request: Request, emp: AIEmployee = Depends(get_employee)
) -> HTMLResponse:
    """OAuth redirect target: exchange the code, connect the server, go live."""
    from ai_employee.tools.mcp import connect_server, sync_registry
    from ai_employee.tools.mcp.oauth import complete_authorization

    q = request.query_params
    if q.get("error"):
        detail = q.get("error_description") or q.get("error") or ""
        return _oauth_result_page(False, f"Authorization was denied: {detail}")
    code, state = q.get("code"), q.get("state")
    if not code or not state:
        return _oauth_result_page(False, "Missing code/state in the callback URL.")
    s = get_settings()
    try:
        server_id = await complete_authorization(s, state, code)
        result = await connect_server(s, server_id)
        sync_registry(s, emp.tools)
    except Exception as exc:  # noqa: BLE001 - show the failure in the popup
        return _oauth_result_page(False, str(exc))
    if not result.get("ok"):
        return _oauth_result_page(False, result.get("error") or "connection failed")
    return _oauth_result_page(
        True, f"{server_id} connected — {len(result.get('tools') or [])} tools available."
    )


def _oauth_result_page(ok: bool, message: str) -> HTMLResponse:
    import html as _html

    icon, title = ("✅", "Connected") if ok else ("❌", "Authorization failed")
    return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<body style="font-family:system-ui;display:grid;place-items:center;height:90vh;margin:0">
<div style="text-align:center;max-width:32rem">
  <div style="font-size:3rem">{icon}</div>
  <h2>{title}</h2><p>{_html.escape(message)}</p>
  <p style="color:#888">You can close this window and return to the console.</p>
</div>
<script>setTimeout(() => window.close(), {2500 if ok else 15000});</script>
</body>""")


# ── Scheduled tasks & watchers (⏰) ──────────────────────────────────────────
@router.get("/ui/api/schedules", tags=["ui"])
async def list_schedules() -> dict:
    from ai_employee.observability.schedules import load_schedules

    entries = sorted(load_schedules(get_settings()).values(), key=lambda e: e["created_at"])
    return {"schedules": entries}


@router.post("/ui/api/schedules", tags=["ui"])
async def add_schedule(req: ScheduleRequest, user: CurrentUser = Depends(get_current_user)) -> dict:
    from ai_employee.observability.schedules import create_schedule

    try:
        return create_schedule(
            get_settings(),
            {**req.model_dump(), "user_id": user.id, "user_name": user.name},
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.put("/ui/api/schedules/{schedule_id}", tags=["ui"])
async def patch_schedule(schedule_id: str, req: SchedulePatchRequest) -> dict:
    from ai_employee.observability.schedules import update_schedule

    try:
        return update_schedule(get_settings(), schedule_id, req.model_dump(exclude_none=True))
    except KeyError:
        raise HTTPException(404, f"unknown schedule: {schedule_id}") from None


@router.delete("/ui/api/schedules/{schedule_id}", tags=["ui"])
async def remove_schedule(schedule_id: str) -> dict:
    from ai_employee.observability.schedules import delete_schedule

    if not delete_schedule(get_settings(), schedule_id):
        raise HTTPException(404, f"unknown schedule: {schedule_id}")
    return {"deleted": schedule_id}


@router.post("/ui/api/schedules/{schedule_id}/run_now", tags=["ui"])
async def run_schedule_now(schedule_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Fire a schedule immediately (also baselines a fresh gmail_watch)."""
    from ai_employee.observability.schedules import run_schedule

    s = get_settings()
    try:
        fired, task_id = await run_schedule(s, schedule_id, emp, get_task_manager())
    except KeyError:
        raise HTTPException(404, f"unknown schedule: {schedule_id}") from None
    return {"fired": fired, "task_id": task_id}


# ── Autonomy & safety (🛡): policies, audit trail, notifications, budget ─────
@router.get("/ui/api/autonomy", tags=["ui"])
async def autonomy_status() -> dict:
    from ai_employee.config.approval_policies import load_policies
    from ai_employee.core.notify import enabled as ntfy_enabled

    s = get_settings()
    return {
        "policies": load_policies(s),
        "approval_keys": sorted(s.approval_required),
        "ntfy": {
            "enabled": ntfy_enabled(s),
            "server": s.ntfy_server,
            "topic": (s.ntfy_topic[:4] + "…") if s.ntfy_topic else "",
        },
        "task_token_budget": s.task_token_budget,
    }


@router.put("/ui/api/autonomy/policies", tags=["ui"])
async def put_policies(req: PoliciesRequest) -> dict:
    from ai_employee.config.approval_policies import save_policies

    save_policies(get_settings(), req.policies)
    return {"ok": True, "policies": req.policies}


@router.post("/ui/api/autonomy/notify_test", tags=["ui"])
async def notify_test() -> dict:
    from ai_employee.core.notify import enabled, push

    s = get_settings()
    if not enabled(s):
        raise HTTPException(400, "NTFY_TOPIC is not set in .env")
    ok = await push(s, "AI Employee", "Test notification — you're connected.", tags="tada")
    return {"ok": ok}


@router.get("/ui/api/audit", tags=["ui"])
async def audit_trail(limit: int = Query(default=100, le=1000), key: str | None = None) -> dict:
    from ai_employee.observability.audit import read_actions

    return {"actions": read_actions(get_settings(), limit=limit, key=key)}


# ── Google accounts (Gmail & Calendar via the operator's OAuth app) ──────────
def _sync_google_tools(emp: AIEmployee) -> int:
    """Register the Google toolset on the live employee (idempotent)."""
    from ai_employee.tools.builtins.google_workspace import build_google_tools

    tools = build_google_tools(get_settings())
    if tools:
        emp.tools.register_many(tools, replace=True)
    return len(tools)


@router.get("/ui/api/google/status", tags=["ui"])
async def google_status() -> dict:
    """Is the operator's OAuth client set up, and which accounts are connected."""
    from ai_employee.integrations.google_oauth import client_config, load_accounts

    s = get_settings()
    accounts = [
        {"email": email, "connected_at": a.get("connected_at"), "scopes": a.get("scopes", "")}
        for email, a in sorted(load_accounts(s).items())
    ]
    return {
        "configured": client_config(s) is not None,
        "client_file": str(s.google_oauth_client_file),
        "accounts": accounts,
    }


@router.post("/ui/api/google/authorize", tags=["ui"])
async def google_authorize(request: Request) -> dict:
    """Begin one user's Google consent flow; returns the URL to open."""
    from ai_employee.integrations.google_oauth import begin_authorization

    redirect_uri = str(request.base_url).rstrip("/") + "/ui/api/google/oauth/callback"
    try:
        return {"auth_url": begin_authorization(get_settings(), redirect_uri)}
    except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
        raise HTTPException(502, str(exc)) from exc


@router.get("/ui/api/google/oauth/callback", tags=["ui"], include_in_schema=False)
async def google_oauth_callback(
    request: Request, emp: AIEmployee = Depends(get_employee)
) -> HTMLResponse:
    """Google redirect target: exchange the code and store the user's grant."""
    from ai_employee.integrations.google_oauth import complete_authorization

    q = request.query_params
    if q.get("error"):
        return _oauth_result_page(False, f"Authorization was denied: {q.get('error')}")
    code, state = q.get("code"), q.get("state")
    if not code or not state:
        return _oauth_result_page(False, "Missing code/state in the callback URL.")
    try:
        email = await complete_authorization(get_settings(), state, code)
        count = _sync_google_tools(emp)
    except Exception as exc:  # noqa: BLE001 - show the failure in the popup
        return _oauth_result_page(False, str(exc))
    return _oauth_result_page(True, f"{email} connected — {count} Gmail/Calendar tools are live.")


@router.delete("/ui/api/google/accounts/{email}", tags=["ui"])
async def google_remove_account(email: str) -> dict:
    """Disconnect one Google account (revokes its grant at Google)."""
    from ai_employee.integrations.google_oauth import remove_account

    if not await remove_account(get_settings(), email):
        raise HTTPException(404, f"Google account not connected: {email}")
    return {"removed": email}


@router.post("/ui/api/mcp/servers/{server_id}/refresh", tags=["ui"])
async def refresh_mcp_server(server_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Re-connect with the stored params and refresh the cached tool list."""
    from ai_employee.tools.mcp import connect_server, load_servers, sync_registry

    s = get_settings()
    if server_id not in load_servers(s):
        raise HTTPException(404, f"MCP connector not configured: {server_id}")
    result = await connect_server(s, server_id)
    result["registered_tools"] = sync_registry(s, emp.tools)
    return result


@router.post("/ui/api/mcp/servers/{server_id}/toggle", tags=["ui"])
async def toggle_mcp_server(
    server_id: str, req: MCPToggleRequest, emp: AIEmployee = Depends(get_employee)
) -> dict:
    """Enable/disable a configured connector without losing its params or tools."""
    from ai_employee.tools.mcp import load_servers, save_server, sync_registry

    s = get_settings()
    if server_id not in load_servers(s):
        raise HTTPException(404, f"MCP connector not configured: {server_id}")
    save_server(s, server_id, enabled=req.enabled)
    return {
        "server": server_id,
        "enabled": req.enabled,
        "registered_tools": sync_registry(s, emp.tools),
    }


@router.delete("/ui/api/mcp/servers/{server_id}", tags=["ui"])
async def remove_mcp_server(server_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Remove a connector (params, cached tools) and unregister its tools."""
    from ai_employee.tools.mcp import delete_server, sync_registry

    s = get_settings()
    if not delete_server(s, server_id):
        raise HTTPException(404, f"MCP connector not configured: {server_id}")
    return {"deleted": server_id, "registered_tools": sync_registry(s, emp.tools)}


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
async def list_sessions(user: CurrentUser = Depends(get_current_user)) -> dict:
    return {"sessions": get_task_manager().list_sessions(owner_id=user.id)}


@router.post("/ui/api/sessions/recover", tags=["ui"])
async def recover_sessions(user: CurrentUser = Depends(get_current_user)) -> dict:
    """Re-import runs still present in the server log but not saved on disk.

    Recovers a run lost to a restart/crash (guest runs used to be memory-only) as
    a session owned by the requesting user, so it appears in Saved tasks and its
    Explain/Logs/Result views render again.
    """
    from ai_employee.observability.recover import import_all_new
    from ai_employee.observability.store import SessionStore

    s = get_settings()
    log_path = Path(s.sessions_dir).parent / "server.log"
    if not log_path.exists():
        return {"imported": [], "count": 0, "log": str(log_path), "note": "no server log found"}
    store = SessionStore(s.sessions_dir)
    imported = import_all_new(store, log_path, owner_id=user.id, owner_name=user.name)
    tm = get_task_manager()
    for meta, snap in imported:
        tm.add_saved_session(meta, [snap])
    return {
        "imported": [
            {"session_id": m["id"], "task_id": m["run_ids"][0], "title": m["title"]}
            for m, _ in imported
        ],
        "count": len(imported),
        "log": str(log_path),
    }


@router.get("/ui/api/sessions/{session_id}", tags=["ui"])
async def get_session(session_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Full session (meta + every run's snapshot) — used to open/resume a saved task."""
    tm = get_task_manager()
    detail = tm.session_detail(session_id)
    if detail is None or not tm.owned_by(session_id, user.id):
        raise HTTPException(404, "unknown session")
    return detail


def _orgchart_from_task(task) -> dict:
    """Reduce one task's event stream to {active agents, edges, progress}.

    Mirrors the client-side ``deriveFlow`` in app.js: the event stream carries
    no explicit "agent A -> agent B" message, so activity is inferred from the
    same fields the Flow tab already relies on (agent.start / workflow.parallel_*
    / workflow.node), plus the plan's own ``depends_on`` graph for step->step
    handoffs. Every edge below corresponds to a real information flow, not a
    standing org-chart line:

    - ``plan -> agent``: the planner has dispatched a step to that agent and it
      is currently running.
    - ``agent -> agent``: step B depends on step A (different agents) and B is
      now running — A's output is literally in B's prompt (see
      ``Orchestrator._format_prior``), so this is a real handoff, not a guess.
    - ``agent -> ceo``: that agent's step result is part of the digest the CEO
      is synthesizing right now.
    """
    running_steps: dict[int, str] = {}  # step -> agent, still in flight
    done_steps = 0
    planned_steps = 0
    synth = False
    ceo_active = False
    for e in task.get("events", []):
        f = e.get("fields") or {}
        ev = e.get("event")
        if ev in ("planner.plan_ready", "workflow.planned") and f.get("steps") is not None:
            planned_steps = max(planned_steps, int(f.get("steps") or 0))
        elif (
            ev == "workflow.parallel_start" or (ev == "agent.start" and f.get("node") == "execute")
        ) and f.get("step") is not None:
            running_steps[f["step"]] = f.get("agent") or "?"
        elif ev == "workflow.parallel_done" and f.get("step") is not None:
            running_steps.pop(f["step"], None)
            done_steps += 1
        elif ev == "workflow.node" and f.get("node") == "synthesize":
            synth = True
            ceo_active = True
        elif ev == "workflow.node" and f.get("node") == "reflect":
            ceo_active = False

    completed = task.get("completed_steps") or done_steps
    planned = task.get("planned_steps") or planned_steps
    progress = round(100 * completed / planned) if planned else (100 if task.get("result") else 0)

    result = task.get("result") or {}
    checkpoint = result.get("workflow_checkpoint") or {}
    plan_steps = (checkpoint.get("plan") or {}).get("steps") or []
    agent_by_step: dict[int, str] = {s["id"]: s["agent"] for s in plan_steps if s.get("agent")}
    depends_by_step: dict[int, list[int]] = {
        s["id"]: list(s.get("depends_on") or []) for s in plan_steps
    }

    active_agents = {a for a in running_steps.values() if a and a != "?"}
    edges: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    def add_edge(frm: str, to: str, kind: str) -> None:
        key = (frm, to, kind)
        if frm and to and frm != to and key not in seen_pairs:
            seen_pairs.add(key)
            edges.append({"from": frm, "to": to, "kind": kind})

    for step_id, agent in running_steps.items():
        if not agent or agent == "?":
            continue
        upstream_agents = {
            agent_by_step[dep] for dep in depends_by_step.get(step_id, []) if dep in agent_by_step
        }
        if upstream_agents:
            # This step's inputs came from other agents' finished work — a real
            # handoff, not a planner dispatch.
            for up in sorted(upstream_agents):
                add_edge(up, agent, "handoff")
        else:
            add_edge("planner", agent, "dispatch")

    if synth:
        # Everyone who has contributed a step result is reporting to the CEO
        # right now, as part of the synthesis digest.
        contributors = {r.get("agent") for r in (result.get("steps") or []) if r.get("agent")}
        for a in sorted(contributors):
            add_edge(a, "ceo", "report")

    return {
        "task_id": task.get("id"),
        "goal": task.get("goal"),
        "status": task.get("status"),
        "progress": progress,
        "active_agents": sorted(active_agents),
        "ceo_active": ceo_active,
        "edges": edges,
    }


@router.get("/ui/api/orgchart", tags=["ui"])
async def orgchart(
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Live org-chart feed: static roster + who's working right now, task by task.

    Powers the CEO-at-top team visualization — polled on the same 1s cadence as
    the per-task Flow view, since there is no push channel for agent activity.
    """
    tm = get_task_manager()
    roster = emp.agents.catalog()
    tasks = [_orgchart_from_task(t.snapshot()) for t in tm.running_tasks(owner_id=user.id)]
    return {"agents": roster, "tasks": tasks}


@router.delete("/ui/api/sessions", tags=["ui"])
async def delete_all_sessions(
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Delete every saved task (of the current user) and its recall memory."""
    tm = get_task_manager()
    ids = [s["id"] for s in tm.list_sessions(owner_id=user.id)]
    removed_memory = 0
    removed_resources = 0
    for sid in ids:
        mem = await emp.clear_session_memory(sid)
        removed_memory += int(mem.get("records_removed") or 0)
        report = emp.delete_task_resources(session_id=sid, purge=True)
        removed_resources += len(report.get("deleted", []))
        tm.delete_session(sid)
    return {
        "deleted": len(ids),
        "session_ids": ids,
        "memory_records_removed": removed_memory,
        "resources_deleted": removed_resources,
    }


@router.delete("/ui/api/sessions/{session_id}/runs/{task_id}", tags=["ui"])
async def delete_followup(
    session_id: str,
    task_id: str,
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Delete a single follow-up run. The session's initial task is kept."""
    tm = get_task_manager()
    _require_session(tm, session_id, user)
    if not tm.delete_run(session_id, task_id):
        raise HTTPException(404, "unknown session, run, or cannot delete the initial task")
    report = emp.delete_task_resources(task_id=task_id, purge=True)
    return {
        "deleted": task_id,
        "session_id": session_id,
        "resources_deleted": len(report.get("deleted", [])),
        "resources_kept": len(report.get("kept", [])),
    }


@router.delete("/ui/api/sessions/{session_id}", tags=["ui"])
async def delete_session(
    session_id: str,
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    tm = get_task_manager()
    _require_session(tm, session_id, user)
    mem = await emp.clear_session_memory(session_id)
    report = emp.delete_task_resources(session_id=session_id, purge=True)
    tm.delete_session(session_id)
    return {
        "deleted": session_id,
        "memory_records_removed": mem.get("records_removed", 0),
        "resources_deleted": len(report.get("deleted", [])),
        "resources_kept": len(report.get("kept", [])),
    }


@router.delete("/ui/api/sessions/{session_id}/memory", tags=["ui"])
async def clear_session_memory(
    session_id: str,
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Clear recall memory for one task/session. Saved run history is kept."""
    tm = get_task_manager()
    _require_session(tm, session_id, user)
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


# ── Learned skills (list / create / edit / delete) ──────────────────────────


def _skill_payload(record: MemoryRecord) -> dict:
    """Expose a skill record as editable fields.

    Metadata is the primary source, but older/imported skills may only have the
    ``SKILL: ... / STEPS:`` content format. Parse that as a fallback so every
    skill is editable from the console.
    """
    metadata = record.metadata or {}
    content = record.content or ""
    lines = content.splitlines()
    header = lines[0].strip() if lines else ""
    name = str(metadata.get("name") or "").strip()
    if not name and header.upper().startswith("SKILL:"):
        name = header.split(":", 1)[1].strip()

    body = lines[1:] if header.upper().startswith("SKILL:") else lines
    description_lines: list[str] = []
    parsed_steps: list[str] = []
    in_steps = False
    for line in body:
        if line.strip().upper() == "STEPS:":
            in_steps = True
            continue
        if in_steps:
            stripped = line.strip()
            if stripped:
                parsed_steps.append(stripped[2:].strip() if stripped.startswith("- ") else stripped)
        else:
            description_lines.append(line)

    raw_steps = metadata.get("steps")
    if isinstance(raw_steps, list):
        steps = [str(s).strip() for s in raw_steps if str(s).strip()]
    else:
        steps = parsed_steps
    raw_tags = metadata.get("tags")
    tags = (
        [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []
    )

    return {
        "id": record.id,
        "name": name or "(unnamed skill)",
        "description": "\n".join(description_lines).strip(),
        "steps": steps,
        "tags": tags,
        "content": content,
        "created_at": record.created_at.isoformat(),
    }


def _clean_skill(req: SkillRequest) -> tuple[str, str, list[str], list[str]]:
    name = req.name.strip()
    if not name:
        raise HTTPException(422, "skill name is required")
    steps = [step.strip() for step in req.steps if step.strip()]
    tags = [tag.strip() for tag in req.tags if tag.strip()]
    return name, req.description.strip(), steps, tags


@router.get("/ui/api/skills", tags=["ui"])
async def list_skills(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Every skill the employee has learned, newest first."""
    records = await emp.memory.list_skills(limit=1000)
    return {"skills": [_skill_payload(r) for r in records], "count": len(records)}


@router.post("/ui/api/skills", tags=["ui"])
async def create_skill(req: SkillRequest, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Add a skill manually (or replace the wording of an automated one)."""
    name, description, steps, tags = _clean_skill(req)
    try:
        record = await emp.memory.save_skill(
            name=name, description=description, steps=steps, tags=tags
        )
    except SkillRejectedError as exc:
        raise HTTPException(422, str(exc)) from None
    emp.persist_memory()
    return {"skill": _skill_payload(record), "created": True}


@router.put("/ui/api/skills/{skill_id}", tags=["ui"])
async def update_skill(
    skill_id: str, req: SkillRequest, emp: AIEmployee = Depends(get_employee)
) -> dict:
    """Edit a skill in place; the id (and its recall linkage) is preserved."""
    name, description, steps, tags = _clean_skill(req)
    try:
        record = await emp.memory.save_skill(
            skill_id=skill_id, name=name, description=description, steps=steps, tags=tags
        )
    except KeyError:
        raise HTTPException(404, "skill not found") from None
    except SkillRejectedError as exc:
        raise HTTPException(422, str(exc)) from None
    emp.persist_memory()
    return {"skill": _skill_payload(record), "updated": True}


@router.delete("/ui/api/skills/{skill_id}", tags=["ui"])
async def delete_skill(skill_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Forget a bad skill so future tasks stop recalling it."""
    if not await emp.memory.delete_skill(skill_id):
        raise HTTPException(404, "skill not found")
    emp.persist_memory()
    return {"deleted": skill_id}


@router.get("/ui/api/parallel/settings", tags=["ui"])
async def get_parallel_settings() -> dict:
    """Current parallel-execution mode and its saved default."""
    from ai_employee.config.ui_preferences import read_preferences

    s = get_settings()
    prefs = read_preferences(s)
    saved = prefs.get("parallel_steps")
    return {
        "parallel_steps": s.parallel_steps,
        "max_parallel_steps": s.max_parallel_steps,
        "saved_default": saved if isinstance(saved, bool) else None,
    }


@router.post("/ui/api/parallel/settings", tags=["ui"])
async def set_parallel_settings(req: ParallelSettingsRequest) -> dict:
    """Turn multi-agent parallel step execution on or off."""
    from ai_employee.config.ui_preferences import save_parallel_default

    s = get_settings()
    s.parallel_steps = req.parallel_steps
    if req.max_parallel_steps and req.max_parallel_steps > 0:
        s.max_parallel_steps = int(req.max_parallel_steps)
    if req.save_default:
        save_parallel_default(s, req.parallel_steps, s.max_parallel_steps)
    return {
        "parallel_steps": s.parallel_steps,
        "max_parallel_steps": s.max_parallel_steps,
        "saved_default": req.save_default,
    }


def _agent_limits_payload(emp: AIEmployee) -> dict:
    """Current limits + each tool-using agent's resolved budget (for the modal)."""
    from ai_employee.agents.base import ToolUsingAgent

    s = get_settings()
    agents = []
    for a in emp.agents.all():
        if not isinstance(a, ToolUsingAgent):
            continue  # planner/ceo/human_approval don't run the tool loop
        iters = s.iteration_limit_for(a.name, a.max_iterations)
        loops = s.loop_limit_for(a.name) if s.agent_loop_check_enabled else 0
        agents.append(
            {
                "name": a.name,
                "role": getattr(a, "role", ""),
                "iterations": iters,
                "loops": loops,
                "max_rounds": iters * (1 + loops),
            }
        )
    agents.sort(key=lambda r: r["name"])
    return {
        "max_iterations": s.agent_max_iterations,
        "max_iterations_by_name": s.agent_max_iterations_by_name,
        "loop_extensions_max": s.agent_loop_extensions_max,
        "loop_extensions_by_name": s.agent_loop_extensions_by_name,
        "loop_check_enabled": s.agent_loop_check_enabled,
        "explain_max_chars": s.explain_max_chars,
        "explain_tool_max_chars": s.explain_tool_max_chars,
        "agents": agents,
    }


def _normalize_agent_limits(spec: str) -> str:
    """Canonicalize "coding=16, research=12" → "coding=16,research=12"."""
    from ai_employee.config.settings import parse_agent_limits

    return ",".join(f"{k}={v}" for k, v in sorted(parse_agent_limits(spec).items()))


@router.get("/ui/api/agents/limits", tags=["ui"])
async def get_agent_limits(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Per-agent iteration & loop budgets (shown/edited in the 🎚 Limits modal)."""
    return _agent_limits_payload(emp)


@router.post("/ui/api/agents/limits", tags=["ui"])
async def set_agent_limits(
    req: AgentLimitsRequest, emp: AIEmployee = Depends(get_employee)
) -> dict:
    """Set how many tool rounds and loop extensions each agent may use.

    Per-agent overrides win over the global values, which win over each agent's
    built-in default. Applied live (the agent loop reads settings per run) and,
    with ``save_default``, persisted so it survives a restart.
    """
    from ai_employee.config.ui_preferences import save_agent_limits

    s = get_settings()
    s.agent_max_iterations = max(0, int(req.max_iterations))
    s.agent_max_iterations_by_name = _normalize_agent_limits(req.max_iterations_by_name)
    s.agent_loop_extensions_max = max(0, int(req.loop_extensions_max))
    s.agent_loop_extensions_by_name = _normalize_agent_limits(req.loop_extensions_by_name)
    s.agent_loop_check_enabled = bool(req.loop_check_enabled)
    s.explain_max_chars = max(0, int(req.explain_max_chars))
    s.explain_tool_max_chars = max(0, int(req.explain_tool_max_chars))
    if req.save_default:
        save_agent_limits(
            s,
            max_iterations=s.agent_max_iterations,
            max_iterations_by_name=s.agent_max_iterations_by_name,
            loop_extensions_max=s.agent_loop_extensions_max,
            loop_extensions_by_name=s.agent_loop_extensions_by_name,
            loop_check_enabled=s.agent_loop_check_enabled,
            explain_max_chars=s.explain_max_chars,
            explain_tool_max_chars=s.explain_tool_max_chars,
        )
    payload = _agent_limits_payload(emp)
    payload["saved_default"] = req.save_default
    return payload


@router.post("/ui/api/memory/restore", tags=["ui"])
async def restore_memories(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Reload memory from the archive saved before the last clear."""
    result = await emp.restore_memories()
    if not result.get("restored"):
        raise HTTPException(404, "no memory backup to restore")
    return result


# ── Resources: task <-> source links + isolation ─────────────────────────────
class IsolateRequest(BaseModel):
    all: bool = False  # isolate every task
    task_id: str | None = None  # or one run
    session_id: str | None = None  # or one task/session
    allow_redownload: bool = True  # re-fetch shared sources whose bytes are gone


@router.get("/ui/api/resources", tags=["ui"])
async def list_resources(emp: AIEmployee = Depends(get_employee)) -> dict:
    """Every source with the tasks linked to it (task <-> source, both ways)."""
    return emp.resources_overview()


@router.post("/ui/api/resources/isolate", tags=["ui"])
async def isolate_resources(req: IsolateRequest, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Isolate tasks so no source stays shared.

    Shared sources are copied into each task's private area; if the shared bytes
    are gone, ``allow_redownload`` re-fetches them from the recorded URL. Shared
    records left with no task links are then purged.
    """
    return await emp.isolate_resources(
        all=req.all,
        task_id=req.task_id,
        session_id=req.session_id,
        allow_redownload=req.allow_redownload,
    )


@router.delete("/ui/api/resources/{resource_id}", tags=["ui"])
async def delete_resource(resource_id: str, emp: AIEmployee = Depends(get_employee)) -> dict:
    """Delete one source by id (system actor; purges its bytes when unreferenced)."""
    mgr = emp.resource_manager()
    deleted = mgr.delete(resource_id, task=None, purge=True)
    if not deleted:
        raise HTTPException(404, "unknown resource")
    return {"deleted": resource_id}


class PurgeRequest(BaseModel):
    purge_bytes: bool = True  # also delete the files on disk
    wipe_workspace: bool = False  # also delete every other file in the workspace


@router.post("/ui/api/resources/purge", tags=["ui"])
async def purge_resources(
    req: PurgeRequest | None = None, emp: AIEmployee = Depends(get_employee)
) -> dict:
    """Delete **every** resource (and its bytes) for a fresh start.

    Removes all catalog entries, the files/objects they point at, the managed
    ``shared/`` and ``tasks/`` trees, and the legacy download cache. With
    ``wipe_workspace`` it also deletes every other file in the workspace
    (downloads/outputs that were never catalogued). Irreversible.
    """
    return emp.purge_resources(
        purge_bytes=(req.purge_bytes if req else True),
        wipe_workspace=(req.wipe_workspace if req else False),
    )


@router.post("/ui/api/tasks/{task_id}/cancel", tags=["ui"])
async def cancel_task(task_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    """Stop a running/pending task. Idempotent: cancelling an already-finished
    task just reports ``cancelled: false``."""
    tm = get_task_manager()
    _require_task(tm, task_id, user)
    return {"task_id": task_id, "cancelled": tm.cancel(task_id)}


@router.post("/ui/api/tasks/{task_id}/resume", tags=["ui"])
async def resume_task(
    task_id: str,
    emp: AIEmployee = Depends(get_employee),
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Resume an interrupted run from its last saved workflow checkpoint."""
    tm = get_task_manager()
    _require_task(tm, task_id, user)
    task = tm.resume(emp, task_id)
    if task is None:
        raise HTTPException(409, "task is not resumable")
    return {"task_id": task.id, "session_id": task.session_id}


@router.get("/ui/api/tasks/{task_id}", tags=["ui"])
async def get_task(
    task_id: str, since: int = 0, user: CurrentUser = Depends(get_current_user)
) -> dict:
    tm = get_task_manager()
    task = _require_task(tm, task_id, user)
    snap = task.snapshot(since=since)
    snap["session_title"] = tm.session_title(task.session_id)
    return snap


@router.get("/ui/api/workspace/file", tags=["ui"])
async def workspace_file(path: str = Query(..., min_length=1)) -> FileResponse:
    """Serve a file from the workspace for inline preview (video/image)."""
    ws = get_settings().workspace_dir
    try:
        target = resolve_workspace_path(ws, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(
        target,
        media_type=guess_media_type(target),
        filename=target.name,
        content_disposition_type="inline",
    )


@router.get("/ui/api/workspace/deliverables", tags=["ui"])
async def workspace_deliverables(since: float | None = Query(None)) -> dict:
    """List non-image artifacts a run produced (reports, spreadsheets, dashboards).

    ``since`` scopes the listing to files written by the current run; the
    workspace persists across runs, so an unscoped call returns everything.
    """
    return collect_deliverables(get_settings().workspace_dir, modified_since=since)


@router.get("/ui/api/workspace/media", tags=["ui"])
async def workspace_media(
    paths: list[str] = Query(default_factory=list),
    all_images: bool = Query(False),
    since: float | None = Query(None),
) -> dict:
    """List video/image previews under workspace paths.

    ``all_images=true`` returns every image in the workspace without sampling —
    used for charts, which agents generate freely but don't always name in the
    final answer. Otherwise only the given paths are inspected (sampled when >5).
    """
    flat: list[str] = []
    for p in paths:
        for part in str(p).split(","):
            part = part.strip()
            if part:
                flat.append(part)
    if all_images:
        # "." is the workspace root. The cap is high enough that every chart from a
        # normal analysis run is listed, but still bounded so a run that downloaded
        # a large image dataset degrades to a sample instead of flooding the page.
        # ``since`` scopes results to files this run created — the workspace is
        # shared, so without it older runs' charts appear in every gallery.
        # Dataset content is excluded outright: this sweep asks "what did the run
        # produce?", and downloaded training images are inputs, not results.
        return sample_media(
            get_settings().workspace_dir,
            [*flat, "."],
            sample_when_over=60,
            sample_size=60,
            modified_since=since,
            exclude_dataset_content=True,
        )
    if not flat:
        return {"videos": [], "images": [], "total": 0, "shown": 0, "sampled": False}
    return sample_media(get_settings().workspace_dir, flat)
