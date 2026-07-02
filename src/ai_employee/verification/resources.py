"""Programmatic checks for resource-finding tasks (videos, datasets, files).

Agents often list search snippets and claim success without a real URL, download,
or workspace file. This module audits step output *before* the LLM verifier runs
so hallucinated resources are caught early.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_URL_RE = re.compile(r"https?://[^\s\)\]>\"'<]+", re.I)
_PATH_RE = re.compile(
    r"(?:\./)?[\w./-]+\.(?:mp4|avi|mov|mkv|webm|m4v|zip|tar|gz|csv|json|py|mp3|wav)\b",
    re.I,
)
_RESOURCE_TASK_RE = re.compile(
    r"\b(video|footage|dataset|download|obtain|find|search for|source|url|file|"
    r"resource|clip|save|saved)\b",
    re.I,
)
_FOUND_CLAIM_RE = re.compile(
    r"\b(found|obtained|downloaded|saved|identified|acquired|successfully|located)\b",
    re.I,
)
_VERIFY_TOOLS = frozenset({"fetch_page", "http_request", "read_file", "list_dir", "download_resource"})


def is_resource_task(task: str, success_criteria: str = "") -> bool:
    return bool(_RESOURCE_TASK_RE.search(f"{task} {success_criteria}"))


def _extract_urls(text: str) -> list[str]:
    return list(dict.fromkeys(_URL_RE.findall(text or "")))


def _extract_paths(text: str) -> list[str]:
    return list(dict.fromkeys(_PATH_RE.findall(text or "")))


def _files_from_tool_trace(tool_trace: list[dict[str, Any]]) -> set[str]:
    files: set[str] = set()
    for entry in tool_trace or []:
        if not entry.get("ok"):
            continue
        tool = entry.get("tool")
        args = entry.get("args") or {}
        if tool == "write_file" and args.get("path"):
            files.add(str(args["path"]))
        if tool == "download_resource" and entry.get("ok"):
            out = entry.get("output") or {}
            if isinstance(out, dict) and out.get("path"):
                files.add(str(out["path"]))
        if tool in ("shell", "python_exec"):
            files.update(_extract_paths(str(entry.get("output") or "")))
    return files


def _resolve_path(path: str, workspace: Path) -> Path | None:
    raw = Path(path)
    for candidate in (raw, workspace / raw, workspace / raw.name):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _url_was_verified(url: str, tool_trace: list[dict[str, Any]]) -> bool:
    for entry in tool_trace or []:
        if not entry.get("ok") or entry.get("tool") not in _VERIFY_TOOLS:
            continue
        args = entry.get("args") or {}
        tool_url = str(args.get("url") or args.get("path") or args.get("destination") or "")
        if entry.get("tool") == "download_resource" and entry.get("ok"):
            meta_url = str((entry.get("args") or {}).get("url") or "")
            if url.rstrip("/") in meta_url or meta_url.rstrip("/") in url:
                return True
        if url.rstrip("/") in tool_url or tool_url.rstrip("/") in url:
            return True
    return False


def audit_resources(
    *,
    task: str,
    success_criteria: str,
    output: str,
    tool_trace: list[dict[str, Any]] | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Return a structured audit for resource-finding steps."""
    if not is_resource_task(task, success_criteria):
        return {"applies": False, "issues": [], "resources": []}

    ws = (workspace or Path(".")).resolve()
    trace = tool_trace or []
    urls = _extract_urls(output)
    paths = _extract_paths(output)
    tool_files = _files_from_tool_trace(trace)
    issues: list[str] = []
    resources: list[dict[str, str]] = []

    for url in urls[:12]:
        local: str | None = None
        status = "not downloaded"
        for path in sorted(set(paths) | tool_files):
            resolved = _resolve_path(path, ws)
            if resolved:
                local = str(resolved)
                status = "downloaded"
                break
        resources.append(
            {
                "name": url.split("/")[-1][:80] or url[:80],
                "web_url": url,
                "local_path": local or "NOT DOWNLOADED",
                "download_status": status,
            }
        )

    for path in paths:
        if _resolve_path(path, ws) is None:
            issues.append(
                f"Output references local file '{path}' but it does not exist in the workspace."
            )

    claims_found = bool(_FOUND_CLAIM_RE.search(output or ""))
    has_download = any(r["download_status"] == "downloaded" for r in resources) or any(
        _resolve_path(p, ws) for p in paths
    )

    if claims_found and not urls and not has_download:
        issues.append(
            "Output claims a resource was found/obtained but provides no verifiable web URL "
            "or existing local file."
        )

    criteria = (success_criteria or "").lower()
    if urls and ("verif" in criteria or "confirm" in criteria) and not any(
        _url_was_verified(u, trace) for u in urls
    ):
        issues.append(
            "Success criteria require verification but no tool confirmed any listed URL."
        )

    task_blob = f"{task} {success_criteria}".lower()
    if (
        "download" in task_blob
        and urls
        and not has_download
        and (claims_found or "obtain" in task.lower())
    ):
        issues.append(
            "Task required obtaining/downloading a resource but no file exists in the workspace."
        )

    return {
        "applies": True,
        "issues": issues,
        "resources": resources,
        "claims_found": claims_found,
        "has_verified_download": has_download,
        "has_urls": bool(urls),
    }


def format_resource_block(resources: list[dict[str, str]], *, model: str = "") -> str:
    """Human-readable resource section for synthesis prompts."""
    if not resources:
        return "No verified resources."
    lines = []
    for r in resources:
        lines.append(
            f"- **Name:** {r.get('name', 'resource')}\n"
            f"  - **Web URL:** {r.get('web_url', 'NOT FOUND')}\n"
            f"  - **Local path:** {r.get('local_path', 'NOT DOWNLOADED')}\n"
            f"  - **Model:** {model or 'N/A'}"
        )
    return "\n".join(lines)


__all__ = ["audit_resources", "format_resource_block", "is_resource_task"]
