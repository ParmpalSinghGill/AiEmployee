"""Programmatic checks for resource-finding tasks (videos, datasets, files).

Agents often list search snippets and claim success without a real URL, download,
or workspace file. This module audits step output *before* the LLM verifier runs
so hallucinated resources are caught early.

Finding a suitable resource and downloading it are separate stages:
- ``completed`` — file exists locally
- ``waiting_for_user`` — suitable URL found, auto-download blocked; manual path needed
- ``found_not_downloaded`` — URL identified but download never attempted
- ``not_found`` / ``failed`` — nothing suitable
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai_employee.core.text_paths import clean_for_path_extraction
from ai_employee.tools.builtins.resource_download import (
    prioritize_download_urls,
    sanitize_url,
)

_URL_RE = re.compile(r"https?://[^\s\)\]>\"'`<]+", re.I)
_PATH_RE = re.compile(
    r"(?:\./)?[\w./-]+\.(?:mp4|avi|mov|mkv|webm|m4v|zip|tar|gz|csv|json|py|mp3|wav)\b",
    re.I,
)
_RESOURCE_TASK_RE = re.compile(
    r"\b(download|obtain|acquire|fetch)\b|"
    r"\b(find|search for|locate|source)\b.{0,80}\b(video|footage|dataset|url|resource|clip)\b|"
    r"\b(video|footage|dataset|resource|clip)\b.{0,80}\b(find|search|download|obtain|acquire|source)\b",
    re.I,
)
_ANALYSIS_TASK_RE = re.compile(
    r"\b(analy[sz]e|clean(?:e[ds]|ing)?|data quality|eda\b|kpi|visuali[sz]|predictive|"
    r"exploratory|insight|regression|forecast)\b",
    re.I,
)
_FOUND_CLAIM_RE = re.compile(
    r"\b(found|obtained|downloaded|saved|identified|acquired|successfully|located)\b",
    re.I,
)
_NOT_FOUND_RE = re.compile(
    r"\b(not found|no suitable|no resource|could not find|unable to find|nothing relevant)\b",
    re.I,
)
_WAITING_RE = re.compile(
    r"\b(waiting_for_user|manual download|download manually|provide.*local path)\b",
    re.I,
)
_VERIFY_TOOLS = frozenset(
    {"fetch_page", "http_request", "read_file", "list_dir", "download_resource"}
)

# Errors that mean automation cannot fetch the file — user action is required.
_MANUAL_DOWNLOAD_MARKERS = (
    "403",
    "401",
    "forbidden",
    "unauthorized",
    "cloudflare",
    "captcha",
    "login",
    "sign in",
    "confirm your age",
    "age-restricted",
    "age restricted",
    "cookies-from-browser",
    "authentication",
    "auth required",
    "robot",
    "quota",
    "rate limit",
    "timed out",
    "timeout",
    "unsupported",
    "not a direct",
    "html",
    "paywall",
    "consent",
    "blocked",
    "denied",
)


def is_resource_task(task: str, success_criteria: str = "") -> bool:
    blob = f"{task} {success_criteria}"
    # Data-cleaning / EDA steps often mention "dataset" and "save a file" but
    # they are not resource-acquisition steps — don't audit them as downloads.
    if _ANALYSIS_TASK_RE.search(blob) and not re.search(
        r"\b(download|obtain|acquire|fetch)\b", blob, re.I
    ):
        return False
    return bool(_RESOURCE_TASK_RE.search(blob))


def is_manual_download_error(error: str) -> bool:
    """True when an automatic download failed for external/policy reasons, not missing URL."""
    t = (error or "").lower()
    return any(m in t for m in _MANUAL_DOWNLOAD_MARKERS)


def _extract_urls(text: str) -> list[str]:
    raw = _URL_RE.findall(text or "")
    return list(dict.fromkeys(u for u in (sanitize_url(x) for x in raw) if u))


def _extract_paths(text: str) -> list[str]:
    # URLs and diff headers contain path-shaped substrings ("/org/repo.git",
    # "a/file.py") that are not local files; strip them before matching.
    cleaned = clean_for_path_extraction(text or "")
    return list(dict.fromkeys(_PATH_RE.findall(cleaned)))


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
            out = entry.get("output")
            if isinstance(out, str):
                try:
                    import json

                    out = json.loads(out)
                except json.JSONDecodeError:
                    out = {}
            if isinstance(out, dict) and out.get("path"):
                files.add(str(out["path"]))
        if tool in ("shell", "python_exec"):
            files.update(_extract_paths(str(entry.get("output") or "")))
    return files


def _download_paths_by_url(tool_trace: list[dict[str, Any]]) -> dict[str, str]:
    """Map each download_resource URL to its saved workspace path."""
    mapping: dict[str, str] = {}
    for entry in tool_trace or []:
        if entry.get("tool") != "download_resource" or not entry.get("ok"):
            continue
        args = entry.get("args") or {}
        url = sanitize_url(str(args.get("url") or ""))
        out = entry.get("output")
        if isinstance(out, str):
            try:
                import json

                out = json.loads(out)
            except json.JSONDecodeError:
                out = {}
        if url and isinstance(out, dict) and out.get("path"):
            mapping[url.rstrip("/")] = str(out["path"])
    return mapping


def _download_failures_from_trace(tool_trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for entry in tool_trace or []:
        if entry.get("tool") != "download_resource" or entry.get("ok"):
            continue
        err = str(entry.get("error") or entry.get("output") or "download failed")
        args = entry.get("args") or {}
        failures.append(
            {
                "url": sanitize_url(str(args.get("url") or "")),
                "error": err,
                "manual": str(is_manual_download_error(err)),
            }
        )
    return failures


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


def _classify_outcome(
    *,
    has_urls: bool,
    has_download: bool,
    claims_found: bool,
    download_failures: list[dict[str, str]],
    download_attempted: bool,
    output: str,
) -> str:
    if has_download:
        # Partial success: some URLs downloaded, others failed — not a full manual pause.
        if download_failures:
            return "partial"
        return "completed"
    if has_urls:
        if download_attempted or download_failures:
            return "waiting_for_user"
        if claims_found:
            return "found_not_downloaded"
        return "not_found"
    if claims_found and _NOT_FOUND_RE.search(output or ""):
        return "not_found"
    if claims_found:
        return "failed"
    return "not_found"


def format_waiting_for_user_message(
    *,
    name: str,
    url: str,
    reason: str,
) -> str:
    return (
        "Status:\nWAITING_FOR_USER\n\n"
        "I found a suitable resource.\n\n"
        f"Name:\n{name}\n\n"
        f"Download URL:\n{url}\n\n"
        f"Reason:\nAutomatic download failed because {reason}.\n\n"
        "Please download the file manually and provide its local path.\n\n"
        "Examples:\n\n"
        "Windows:\nC:\\Videos\\people.mp4\n\n"
        "Linux:\n/home/user/videos/people.mp4\n\n"
        "macOS:\n/Users/name/Videos/people.mp4\n\n"
        "Once the local path is provided, I will continue automatically "
        "from the next workflow step."
    )


def build_waiting_for_user_payload(
    *,
    task: str,
    resource_audit: dict[str, Any],
    tool_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a workflow pause payload when a resource was found but not downloaded."""
    resources = resource_audit.get("resources") or []
    failures = resource_audit.get("download_failures") or []
    primary = resources[0] if resources else {}
    name = str(primary.get("name") or task[:80] or "resource")
    url = sanitize_url(str(primary.get("web_url") or ""))
    candidate_urls = prioritize_download_urls(
        [sanitize_url(str(r.get("web_url") or "")) for r in resources if r.get("web_url")]
    )
    if candidate_urls:
        url = candidate_urls[0]
        name = url.split("/")[-1][:80] or name
    reason = failures[0]["error"] if failures else "the host blocks automated downloads"
    message = format_waiting_for_user_message(
        name=name, url=url or "(see candidate URLs)", reason=reason
    )
    args = {
        "resource_description": task,
        "candidate_urls": candidate_urls[:12],
        "hubs_searched": [],
        "missing_credentials": [],
        "required": True,
        "message": message,
        "download_failure_reason": reason,
    }
    return {
        "operation": "waiting_for_user",
        "tool": "request_resource_input",
        "resource_description": task,
        "candidate_urls": candidate_urls[:12],
        "download_failure_reason": reason,
        "message": message,
        "args": args,
    }


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
        return {"applies": False, "issues": [], "resources": [], "outcome": "not_applicable"}

    ws = (workspace or Path(".")).resolve()
    trace = tool_trace or []
    urls = _extract_urls(output)
    for entry in trace:
        args = entry.get("args") or {}
        for key in ("url", "resource_url"):
            u = sanitize_url(str(args.get(key) or ""))
            if u.startswith("http") and u not in urls:
                urls.append(u)
    urls = prioritize_download_urls(urls)
    paths = _extract_paths(output)
    tool_files = _files_from_tool_trace(trace)
    download_failures = _download_failures_from_trace(trace)
    download_paths = _download_paths_by_url(trace)
    download_attempted = bool(download_failures) or any(
        e.get("tool") == "download_resource" for e in trace
    )
    issues: list[str] = []
    resources: list[dict[str, str]] = []

    for url in urls[:12]:
        local: str | None = None
        status = "not downloaded"
        mapped = download_paths.get(url.rstrip("/"))
        if mapped:
            resolved = _resolve_path(mapped, ws)
            if resolved:
                local = str(resolved)
                status = "downloaded"
        if not local:
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
        if _resolve_path(path, ws) is None and not _WAITING_RE.search(output or ""):
            issues.append(
                f"Output references local file '{path}' but it does not exist in the workspace."
            )

    claims_found = bool(_FOUND_CLAIM_RE.search(output or ""))
    has_download = any(r["download_status"] == "downloaded" for r in resources) or any(
        _resolve_path(p, ws) for p in paths
    )

    outcome = _classify_outcome(
        has_urls=bool(urls),
        has_download=has_download,
        claims_found=claims_found,
        download_failures=download_failures,
        download_attempted=download_attempted,
        output=output,
    )

    if outcome == "failed":
        issues.append(
            "Output claims a resource was found/obtained but provides no verifiable web URL "
            "or existing local file."
        )
    elif outcome == "found_not_downloaded":
        issues.append(
            "A suitable resource URL was identified but automatic download was not attempted. "
            "Call download_resource, then request_resource_input if download is blocked."
        )
    elif (
        outcome == "not_found"
        and (claims_found or "download" in task.lower())
        and (_NOT_FOUND_RE.search(output or "") or not urls)
    ):
        issues.append("No suitable resource was found after searching.")

    criteria = (success_criteria or "").lower()
    if (
        urls
        and ("verif" in criteria or "confirm" in criteria)
        and not any(_url_was_verified(u, trace) for u in urls)
        and outcome not in ("waiting_for_user", "completed")
    ):
        issues.append("Success criteria require verification but no tool confirmed any listed URL.")

    return {
        "applies": True,
        "issues": issues,
        "resources": resources,
        "claims_found": claims_found,
        "has_verified_download": has_download,
        "has_urls": bool(urls),
        "outcome": outcome,
        "download_failures": download_failures,
        "download_attempted": download_attempted,
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


__all__ = [
    "audit_resources",
    "build_waiting_for_user_payload",
    "format_resource_block",
    "format_waiting_for_user_message",
    "is_manual_download_error",
    "is_resource_task",
    "prioritize_download_urls",
    "sanitize_url",
]
