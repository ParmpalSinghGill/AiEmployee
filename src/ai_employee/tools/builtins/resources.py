"""Resource acquisition tools: multi-hub search helpers, smart download, human gate."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.config import Settings
from ai_employee.core.exceptions import ApprovalRequired
from ai_employee.observability.resource_ctx import get_resource_decision
from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._sandbox import ensure_workspace
from ai_employee.tools.builtins.resource_download import (
    _safe_dest,
    detect_source,
    download_github,
    download_http,
    download_huggingface,
    download_kaggle,
    download_youtube,
)

_HUB_TOOLS = (
    "hf_search",
    "paperswithcode_search",
    "github_search",
    "kaggle_search",
    "roboflow_search",
    "youtube_search",
    "web_search",
)


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


class HubSearchAllTool(Tool):
    """Run the standard ML/public-data hubs with one call (reduces LLM round-trips)."""

    name = "hub_search_all"
    description = (
        "Search ALL major public resource hubs for a query: Hugging Face, Papers With Code, "
        "GitHub, Kaggle, Roboflow, and YouTube (via web). Returns a dict of hits per hub. "
        "Use when no dataset was found yet or the task needs code + data from public sources."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        query: str
        limit: int = Field(5, ge=1, le=10)

    async def _run(self, args: HubSearchAllTool.Args) -> ToolResult:
        from ai_employee.tools.builtins.research import (
            GithubSearchTool,
            HfSearchTool,
            KaggleSearchTool,
            PapersWithCodeSearchTool,
            RoboflowSearchTool,
        )

        tools: list[tuple[str, Tool]] = [
            ("huggingface", HfSearchTool()),
            ("paperswithcode", PapersWithCodeSearchTool()),
            ("github", GithubSearchTool(self._settings)),
            ("kaggle", KaggleSearchTool(self._settings)),
            ("roboflow", RoboflowSearchTool(self._settings)),
            ("youtube", YoutubeSearchTool(self._settings)),
        ]
        out: dict[str, Any] = {}
        notes: list[str] = []
        for hub, tool in tools:
            try:
                if tool.name == "hf_search":
                    r = await tool.execute(query=args.query, kind="both", limit=args.limit)
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
        try:
            dest = _safe_dest(ws, args.destination)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))

        kind = args.source_hint if args.source_hint != "auto" else detect_source(args.url)

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
                proc.kill()
                return 124, "", "timed out after 300s"
            return proc.returncode or 0, out_b.decode(errors="replace"), err_b.decode(errors="replace")

        try:
            if kind == "youtube":
                await self._ensure_pkg("yt-dlp")
                info = await download_youtube(args.url, dest, workspace=ws, run_shell=run_shell)
            elif kind == "huggingface":
                await self._ensure_pkg("huggingface_hub")
                info = await download_huggingface(args.url, dest, run_shell=run_shell)
            elif kind == "kaggle":
                has = bool(self._settings.kaggle_username and self._settings.kaggle_key)
                info = await download_kaggle(
                    args.url, dest, run_shell=run_shell, has_creds=has
                )
            elif kind == "github":
                info = await download_github(args.url, dest, run_shell=run_shell)
            else:
                info = await download_http(args.url, dest)
            return ToolResult(
                ok=True,
                output=info,
                metadata={"source": kind, "local_path": info.get("path")},
            )
        except Exception as exc:  # noqa: BLE001
            hint = _credential_hint(kind, self._settings)
            msg = str(exc)
            if hint:
                msg += f" | {hint}"
            return ToolResult(ok=False, error=msg, metadata={"source": kind, "url": args.url})

    async def _ensure_pkg(self, package: str) -> None:
        from ai_employee.tools.builtins.code import PipInstallTool

        pip = PipInstallTool(self._ws)
        await pip.execute(packages=[package])


class RequestResourceInputTool(Tool):
    """Pause the task and ask the human how to proceed when download/search failed."""

    name = "request_resource_input"
    description = (
        "Pause and ask the human for help when a required resource could not be downloaded. "
        "Provide candidate URLs, which hubs were searched, and any missing API keys. "
        "The user can: supply a local workspace path, provide a new URL, add API keys (🔑 Keys) "
        "and retry, proceed without the resource, or abort. Use when the task cannot continue "
        "without data OR when automatic download failed everywhere."
    )
    category = "research"
    approval_key = "resource_acquisition"

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

    async def _run(self, args: RequestResourceInputTool.Args) -> ToolResult:
        decision = get_resource_decision()
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
        }
        if decision:
            return ToolResult(
                ok=True,
                output={
                    "human_decision": decision,
                    "note": "Resuming with user-provided resource decision.",
                },
                metadata={"resolved": True},
            )
        raise ApprovalRequired("resource_acquisition", payload=payload)


def _credential_hint(kind: str, settings: Settings) -> str:
    if kind == "kaggle" and not (settings.kaggle_username and settings.kaggle_key):
        return "Set KAGGLE_USERNAME and KAGGLE_KEY in 🔑 Keys"
    if kind == "roboflow" and not settings.roboflow_api_key:
        return "Set ROBOFLOW_API_KEY in 🔑 Keys"
    if kind == "github" and not settings.github_token:
        return "GITHUB_TOKEN optional but improves rate limits"
    return ""


def build_resource_tools(settings: Settings) -> list[Tool]:
    return [
        YoutubeSearchTool(settings),
        HubSearchAllTool(settings),
        DownloadResourceTool(settings),
        RequestResourceInputTool(),
    ]


__all__ = [
    "YoutubeSearchTool",
    "HubSearchAllTool",
    "DownloadResourceTool",
    "RequestResourceInputTool",
    "build_resource_tools",
]
