"""Resource acquisition tools: multi-hub search helpers, smart download, human gate."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.config import Settings
from ai_employee.core.exceptions import ApprovalRequired
from ai_employee.core.logging import get_logger
from ai_employee.observability.asset_registry import AssetRegistry
from ai_employee.observability.resource_ctx import (
    get_approved_ops,
    get_resource_decision,
    get_resource_session,
    get_resource_task_id,
)
from ai_employee.resources import ResourceManager, Visibility
from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._sandbox import ensure_workspace
from ai_employee.tools.builtins.resource_download import (
    canonical_artifact_dest,
    detect_source,
    download_github,
    download_http,
    download_huggingface,
    download_kaggle,
    download_youtube,
    prioritize_download_urls,
    sanitize_url,
)
from ai_employee.verification.resources import (
    format_waiting_for_user_message,
    is_manual_download_error,
)

log = get_logger("resources")

_HUB_TOOLS = (
    "hf_search",
    "paperswithcode_search",
    "github_search",
    "kaggle_search",
    "roboflow_search",
    "youtube_search",
    "web_search",
)


def visibility_for_path(rel_path: str) -> str:
    """Datasets/models are shared & reusable; session files are task-private."""
    parts = Path(rel_path.replace("\\", "/")).parts
    return Visibility.private.value if parts[:1] == ("sessions",) else Visibility.shared.value


def index_download(
    workspace: Path,
    settings: Settings,
    *,
    path: str | Path,
    url: str,
    source: str,
    session_id: str | None,
) -> Any:
    """Register a downloaded artifact in the resource catalog (best effort).

    Never let bookkeeping break a successful download — a failure here is logged
    and swallowed.
    """
    try:
        mgr = ResourceManager(workspace, audit_file=getattr(settings, "audit_log_file", None))
        try:
            rel = mgr_relative(workspace, Path(path))
        except Exception:  # noqa: BLE001
            rel = Path(str(path)).name
        visibility = visibility_for_path(rel)
        # Always attach a link to the downloading task: for a shared resource this
        # is a consumer edge (so several tasks can share one source), for a private
        # one it is the owner edge.
        task_key = mgr.task_identity(session_id) if session_id else ""
        return mgr.register(
            path,
            visibility=visibility,
            task=task_key or None,
            task_id=get_resource_task_id(),
            source=source,
            url=url,
            session_id=session_id or "",
            tool="download_resource",
        )
    except Exception as exc:  # noqa: BLE001 - never break a download
        log.warning("resource.index_failed", error=str(exc), path=str(path))
        return None


def mgr_relative(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(workspace).resolve()).as_posix()
    except ValueError:
        return path.name


class YoutubeSearchTool(Tool):
    name = "youtube_search"
    description = (
        "Search YouTube for videos matching a query. Returns title, url, and snippet. "
        "Use for footage, tutorials, or public video datasets hosted on YouTube."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        from ai_employee.tools.builtins.web import WebSearchTool

        self._web = WebSearchTool(settings)

    class Args(BaseModel):
        query: str
        max_results: int = Field(6, ge=1, le=15)

    async def _run(self, args: YoutubeSearchTool.Args) -> ToolResult:
        q = f"site:youtube.com {args.query}"
        res = await self._web.execute(query=q, max_results=args.max_results)
        rows = res.output if isinstance(res.output, list) else []
        yt = [r for r in rows if "youtube.com" in str(r.get("url", "")).lower()]
        return ToolResult(
            ok=True,
            output=yt,
            metadata={"source": "youtube", "via": "web_search"},
            error=res.error,
        )


def hubs_for_kind(kind: str) -> tuple[str, ...]:
    """Which hubs ``hub_search_all`` should query for a resource modality."""
    if kind == "tabular":
        return ("kaggle", "huggingface", "github", "paperswithcode")
    if kind == "video":
        return ("youtube", "huggingface", "roboflow", "github")
    if kind == "image":
        return ("roboflow", "huggingface", "kaggle", "github", "paperswithcode")
    if kind == "speech":
        return ("huggingface", "kaggle", "github", "paperswithcode")
    return ("huggingface", "paperswithcode", "github", "kaggle", "roboflow", "youtube")


class HubSearchAllTool(Tool):
    """Run the standard ML/public-data hubs with one call (reduces LLM round-trips)."""

    name = "hub_search_all"
    description = (
        "Search public resource hubs for a query. kind selects modality: tabular (CSV/Kaggle), "
        "video (YouTube/Roboflow), image (Roboflow/HF/Kaggle), speech (HF/Kaggle audio/ASR), "
        "or all. Returns a dict of hits per hub."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        query: str
        limit: int = Field(5, ge=1, le=10)
        kind: str = Field(
            "all",
            pattern="^(all|tabular|video|image|speech)$",
            description="all | tabular | video | image | speech (audio/ASR).",
        )

    async def _run(self, args: HubSearchAllTool.Args) -> ToolResult:
        from ai_employee.tools.builtins.research import (
            GithubSearchTool,
            HfSearchTool,
            KaggleSearchTool,
            PapersWithCodeSearchTool,
            RoboflowSearchTool,
        )

        catalog: dict[str, Tool] = {
            "huggingface": HfSearchTool(),
            "paperswithcode": PapersWithCodeSearchTool(),
            "github": GithubSearchTool(self._settings),
            "kaggle": KaggleSearchTool(self._settings),
            "roboflow": RoboflowSearchTool(self._settings),
            "youtube": YoutubeSearchTool(self._settings),
        }
        tools: list[tuple[str, Tool]] = [
            (hub, catalog[hub]) for hub in hubs_for_kind(args.kind) if hub in catalog
        ]
        out: dict[str, Any] = {}
        notes: list[str] = []
        for hub, tool in tools:
            try:
                if tool.name == "hf_search":
                    hf_kind = "datasets" if args.kind in {"tabular", "image", "speech"} else "both"
                    r = await tool.execute(query=args.query, kind=hf_kind, limit=args.limit)
                elif tool.name == "youtube_search":
                    r = await tool.execute(query=args.query, max_results=args.limit)
                else:
                    r = await tool.execute(query=args.query, limit=args.limit)
                out[hub] = r.output
                meta = r.metadata or {}
                if meta.get("note"):
                    notes.append(f"{hub}: {meta['note']}")
            except Exception as exc:  # noqa: BLE001
                out[hub] = []
                notes.append(f"{hub}: {exc}")
        return ToolResult(
            ok=any(out.get(h) for h in out),
            output=out,
            metadata={"hubs": list(out.keys()), "notes": notes},
            error="; ".join(notes) if notes and not any(out.get(h) for h in out) else None,
        )


class DownloadResourceTool(Tool):
    name = "download_resource"
    description = (
        "Download a resource (video, dataset, repo, file) into the workspace. "
        "Auto-detects source: YouTube → yt-dlp, Hugging Face → huggingface_hub, "
        "Kaggle → kaggle CLI, GitHub → git clone, else direct HTTP. "
        "Installs yt-dlp / huggingface_hub via pip_install when needed. "
        "Requires approval (resource_download). If credentials are missing (Kaggle, etc.) "
        "the error message states which API key to add in 🔑 Keys."
    )
    category = "research"
    approval_key = "resource_download"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ws = settings.workspace_dir

    class Args(BaseModel):
        url: str = Field(description="Direct resource URL (YouTube, HF, Kaggle, GitHub, or file).")
        destination: str = Field(
            description="Workspace path for the file or directory, e.g. ./data/mall_video.mp4"
        )
        source_hint: str = Field(
            "auto",
            description="auto | youtube | huggingface | kaggle | github | http",
        )

    async def _run(self, args: DownloadResourceTool.Args) -> ToolResult:
        ws = ensure_workspace(self._ws)
        url = sanitize_url(args.url)
        if not url:
            return ToolResult(ok=False, error="empty or invalid url")
        session_id = get_resource_session()
        try:
            dest = canonical_artifact_dest(
                ws,
                session_id,
                url=url,
                source_hint=args.source_hint,
                requested_dest=args.destination,
            )
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))

        # Bandwidth guard: if this exact URL was already fetched into the workspace
        # and the file is still there, hand back the local copy. Context tells the
        # agent to prefer local assets, but context is advisory — this is the check
        # that actually guarantees we never pull the same bytes twice.
        registry = AssetRegistry(ws)
        cached = registry.find_by_url(url)
        if cached is not None:
            local = ws / cached.path
            registry.touch(cached.path)
            log.info("download_skipped_cached", url=url, path=cached.path)
            index_download(
                ws,
                self._settings,
                path=local,
                url=url,
                source=cached.source,
                session_id=session_id,
            )
            return ToolResult(
                ok=True,
                output={
                    "path": str(local),
                    "bytes": cached.bytes,
                    "source": cached.source,
                    "cached": True,
                    "note": (
                        "Already present in the workspace — reused the existing copy "
                        "instead of downloading again."
                    ),
                },
                metadata={
                    "source": cached.source,
                    "local_path": str(local),
                    "canonical_destination": str(local),
                    "session_id": session_id,
                    "cache_hit": True,
                },
            )

        kind = args.source_hint if args.source_hint != "auto" else detect_source(url)

        async def run_shell(cmd: list[str]) -> tuple[int, str, str]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(ws),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=300)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                return 124, "", "timed out after 300s"
            return (
                proc.returncode or 0,
                out_b.decode(errors="replace"),
                err_b.decode(errors="replace"),
            )

        try:
            if kind == "youtube":
                await self._ensure_pkg("yt-dlp")
                browser = (self._settings.ytdlp_cookies_from_browser or "").strip() or None
                cookies_file = self._settings.ytdlp_cookies_file
                info = await download_youtube(
                    url,
                    dest,
                    workspace=ws,
                    run_shell=run_shell,
                    cookies_from_browser=browser,
                    cookies_file=cookies_file,
                )
            elif kind == "huggingface":
                info = await download_huggingface(
                    url,
                    dest,
                    workspace=ws,
                    token=(self._settings.huggingface_token or "").strip() or None,
                    run_shell=run_shell,
                )
            elif kind == "kaggle":
                has = bool(self._settings.kaggle_username and self._settings.kaggle_key)
                info = await download_kaggle(url, dest, run_shell=run_shell, has_creds=has)
            elif kind == "github":
                info = await download_github(url, dest, run_shell=run_shell)
            else:
                info = await download_http(url, dest)
            # Index it so any later task — in any session — finds it without a refetch.
            registry.record(
                path=info.get("path") or dest,
                source=kind,
                url=url,
                session_id=session_id or "",
            )
            index_download(
                ws,
                self._settings,
                path=info.get("path") or dest,
                url=url,
                source=kind,
                session_id=session_id,
            )
            return ToolResult(
                ok=True,
                output=info,
                metadata={
                    "source": kind,
                    "local_path": info.get("path"),
                    "canonical_destination": str(dest),
                    "session_id": session_id,
                },
            )
        except Exception as exc:  # noqa: BLE001
            hint = _credential_hint(kind, self._settings, str(exc))
            msg = str(exc)
            if hint:
                msg += f" | {hint}"
            manual = is_manual_download_error(msg)
            return ToolResult(
                ok=False,
                error=msg,
                metadata={
                    "source": kind,
                    "url": url,
                    "manual_download_required": manual,
                },
            )

    async def _ensure_pkg(self, package: str) -> None:
        from ai_employee.tools.builtins.code import PipInstallTool

        pip = PipInstallTool(self._ws)
        await pip.execute(packages=[package])


class RequestResourceInputTool(Tool):
    """Pause the task when a resource was found but cannot be downloaded automatically."""

    name = "request_resource_input"
    description = (
        "Pause and ask the human for help acquiring a resource. Use when a SUITABLE "
        "resource URL was found but automatic download failed (403, login, CAPTCHA, etc.) "
        "or when no hub returned a usable link. If candidates exist, the task enters "
        "WAITING_FOR_USER (not failed). Provide candidate URLs, which hubs were searched, "
        "and the download failure reason. The user can supply a local workspace path, "
        "a new URL, proceed without the resource, or abort."
    )
    category = "research"

    class Args(BaseModel):
        resource_description: str
        candidate_urls: list[str] = Field(default_factory=list)
        hubs_searched: list[str] = Field(default_factory=list)
        missing_credentials: list[str] = Field(default_factory=list)
        required: bool = Field(
            True,
            description="If true, warn that proceeding without the resource may block the goal.",
        )
        message: str = ""
        download_failure_reason: str = Field(
            "",
            description="Why automatic download failed (e.g. HTTP 403, login required).",
        )

    async def _run(self, args: RequestResourceInputTool.Args) -> ToolResult:
        decision = get_resource_decision()
        approved = get_approved_ops()
        if decision:
            return ToolResult(
                ok=True,
                output={
                    "human_decision": decision,
                    "note": "Resuming with user-provided resource decision.",
                },
                metadata={"resolved": True},
            )
        if approved & {"resource_acquisition", "waiting_for_user"}:
            return ToolResult(
                ok=True,
                output={
                    "human_decision": "proceed_without",
                    "note": "Resource acquisition already approved for this run.",
                },
                metadata={"resolved": True, "preapproved": True},
            )

        payload = {
            "tool": self.name,
            "args": args.model_dump(),
            "resource_description": args.resource_description,
            "candidate_urls": args.candidate_urls[:12],
            "hubs_searched": args.hubs_searched or list(_HUB_TOOLS),
            "missing_credentials": args.missing_credentials,
            "required": args.required,
            "options": [
                "provide_local_path",
                "provide_url",
                "add_api_keys",
                "proceed_without",
                "abort",
            ],
            "message": args.message,
            "download_failure_reason": args.download_failure_reason,
        }
        candidates = prioritize_download_urls(
            [u for u in (sanitize_url(x) for x in args.candidate_urls) if u]
        )
        if candidates:
            name = args.resource_description[:80] or candidates[0].split("/")[-1]
            reason = (
                args.download_failure_reason
                or args.message
                or "the host blocks automated downloads"
            )
            payload["message"] = format_waiting_for_user_message(
                name=name,
                url=candidates[0],
                reason=reason,
            )
            payload["download_failure_reason"] = reason
            raise ApprovalRequired("waiting_for_user", payload=payload)

        payload["message"] = args.message or (
            "No suitable resource could be found automatically. "
            "Provide a local path, a direct URL, or choose how to proceed."
        )
        raise ApprovalRequired("resource_acquisition", payload=payload)


def _credential_hint(kind: str, settings: Settings, error: str = "") -> str:
    if kind == "kaggle" and not (settings.kaggle_username and settings.kaggle_key):
        return "Set KAGGLE_USERNAME and KAGGLE_KEY in 🔑 Keys"
    if kind == "roboflow" and not settings.roboflow_api_key:
        return "Set ROBOFLOW_API_KEY in 🔑 Keys"
    if kind == "github" and not settings.github_token:
        return "GITHUB_TOKEN optional but improves rate limits"
    if kind == "huggingface" and not settings.huggingface_token:
        err = (error or "").lower()
        if any(x in err for x in ("gated", "authorized", "401", "private", "authenticate")):
            return "Set HF_TOKEN in 🔑 Keys for gated/private Hugging Face repos"
    return ""


class SearchLocalAssetsTool(Tool):
    """Look up already-downloaded models/datasets before hitting the network."""

    name = "search_local_assets"
    description = (
        "Search models, datasets and files ALREADY downloaded in this workspace. "
        "Call this FIRST, before hf_search / kaggle_search / github_search / "
        "download_resource — if a suitable asset is already local, use its path "
        "instead of downloading it again. Leave the query empty to list everything. "
        "Read-only and offline: no network, no approval needed."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._ws = settings.workspace_dir

    class Args(BaseModel):
        query: str = Field(
            "",
            description="What you need, e.g. 'defect detection model'. Empty lists all.",
        )
        kind: str = Field(
            "",
            description="Optional filter: model | dataset | repo | media | file",
        )

    async def _run(self, args: SearchLocalAssetsTool.Args) -> ToolResult:
        ws = ensure_workspace(self._ws)
        registry = AssetRegistry(ws)
        if args.query.strip():
            records = registry.find(args.query, limit=20)
            if args.kind:
                records = [r for r in records if r.kind == args.kind]
        else:
            records = registry.list_assets(kind=args.kind.strip())

        items = [
            {
                "identifier": r.identifier,
                "kind": r.kind,
                "path": str(ws / r.path),
                "relative_path": r.path,
                "source": r.source,
                "url": r.url,
                "bytes": r.bytes,
            }
            for r in records
        ]
        if not items:
            return ToolResult(
                ok=True,
                output={
                    "assets": [],
                    "count": 0,
                    "note": ("Nothing local matches. Searching the hubs is appropriate now."),
                },
            )
        return ToolResult(
            ok=True,
            output={
                "assets": items,
                "count": len(items),
                "note": (
                    "These are already on disk — use one of these paths instead of "
                    "downloading again unless the user asked for something new."
                ),
            },
            metadata={"count": len(items)},
        )


def build_resource_tools(settings: Settings) -> list[Tool]:
    return [
        YoutubeSearchTool(settings),
        HubSearchAllTool(settings),
        DownloadResourceTool(settings),
        SearchLocalAssetsTool(settings),
        RequestResourceInputTool(),
    ]


__all__ = [
    "YoutubeSearchTool",
    "HubSearchAllTool",
    "DownloadResourceTool",
    "SearchLocalAssetsTool",
    "RequestResourceInputTool",
    "build_resource_tools",
]
