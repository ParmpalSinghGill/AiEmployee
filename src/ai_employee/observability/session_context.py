"""Build follow-up context from earlier runs in the same saved session.

Follow-up goals often refer to prior work ("download that YouTube link"). Semantic
memory may not have been written (e.g. resource steps found URLs but did not
finish downloading). Persisted task snapshots always retain goals, answers, and
URLs — this module turns those into planner/agent context.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai_employee.config import get_settings

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_TRAILING_PUNCT = re.compile(r"[.,;:!?)}\]]+$")
_PATH_RE = re.compile(
    r"(?:\./)?[\w./-]+\.(?:mp4|avi|mov|mkv|webm|m4v|zip|tar|gz|csv|json|py|pt|pth|safetensors)\b",
    re.I,
)


def _clean_url(url: str) -> str:
    u = (url or "").strip().rstrip("`")
    u = _TRAILING_PUNCT.sub("", u)
    return u


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in _URL_RE.findall(text):
        u = _clean_url(raw)
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_paths(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in _PATH_RE.findall(text):
        p = raw.strip().rstrip("`")
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _urls_from_step(step: dict[str, Any]) -> list[str]:
    urls = extract_urls(step.get("output") or "")
    audit = step.get("resource_audit") or {}
    for res in audit.get("resources") or []:
        u = _clean_url(res.get("web_url") or "")
        if u.startswith("http"):
            urls.append(u)
    return urls


def urls_from_result(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    seen: set[str] = set()
    ordered: list[str] = []

    def add(candidates: list[str]) -> None:
        for u in candidates:
            u = _clean_url(u)
            if u.startswith("http") and u not in seen:
                seen.add(u)
                ordered.append(u)

    add(extract_urls(result.get("answer") or ""))
    for step in result.get("steps") or []:
        add(_urls_from_step(step))
    checkpoint = result.get("workflow_checkpoint") or {}
    for step in checkpoint.get("step_results") or []:
        add(_urls_from_step(step))
    return ordered


def _resolve_path(path: str, workspace: Path) -> Path | None:
    raw = Path(path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([workspace / raw, workspace / raw.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resource_pairs_from_result(
    result: dict[str, Any] | None,
    *,
    workspace: Path,
) -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
    existing: list[tuple[str, str | None]] = []
    missing: list[tuple[str, str | None]] = []
    if not result:
        return existing, missing
    seen_existing: set[tuple[str, str | None]] = set()
    seen_missing: set[tuple[str, str | None]] = set()

    def add_path(path: str, url: str | None = None) -> None:
        clean_url = _clean_url(url or "") or None
        resolved = _resolve_path(path, workspace)
        item = (str(resolved) if resolved else path, clean_url)
        if resolved:
            if item not in seen_existing:
                seen_existing.add(item)
                existing.append(item)
        else:
            if item not in seen_missing:
                seen_missing.add(item)
                missing.append(item)

    def scan_step(step: dict[str, Any]) -> None:
        audit = step.get("resource_audit") or {}
        for res in audit.get("resources") or []:
            local = (res.get("local_path") or "").strip()
            if local and local != "NOT DOWNLOADED":
                add_path(local, res.get("web_url"))
        for p in extract_paths(step.get("output") or ""):
            add_path(p)

    for p in extract_paths(result.get("answer") or ""):
        add_path(p)
    for step in result.get("steps") or []:
        scan_step(step)
    checkpoint = result.get("workflow_checkpoint") or {}
    for step in checkpoint.get("step_results") or []:
        scan_step(step)
    return existing, missing


def _prior_run_section(
    run: int,
    *,
    goal: str,
    result: dict[str, Any] | None,
    workspace: Path,
) -> str:
    lines = [f"### Prior run {run}", f"Goal: {goal}"]
    answer = (result or {}).get("answer") or ""
    if answer:
        clipped = answer if len(answer) <= 6000 else answer[:6000] + "\n…(answer truncated)"
        lines.append(f"Answer:\n{clipped}")
    urls = urls_from_result(result)
    if urls:
        lines.append(
            "URLs found (use these when the user refers to 'the link', 'that video', etc.):"
        )
        lines.extend(f"- {u}" for u in urls[:40])
        if len(urls) > 40:
            lines.append(f"… and {len(urls) - 40} more URL(s)")
    existing, missing = _resource_pairs_from_result(result, workspace=workspace)
    if existing:
        lines.append("Existing local resources (verified on disk now):")
        lines.extend(f"- {path}" + (f"  ← {url}" if url else "") for path, url in existing[:40])
        if len(existing) > 40:
            lines.append(f"… and {len(existing) - 40} more existing local resource(s)")
    if missing:
        lines.append(
            "Missing prior downloads (these paths do NOT exist now; verify or re-download from the URL if needed):"
        )
        lines.extend(f"- {path}" + (f"  ← {url}" if url else "") for path, url in missing[:40])
        if len(missing) > 40:
            lines.append(f"… and {len(missing) - 40} more missing local resource(s)")
    return "\n".join(lines)


def build_prior_session_context(
    prior_runs: list[Any],
    *,
    max_chars: int = 14000,
    workspace: Path | None = None,
) -> str:
    """Summarize completed earlier runs for a follow-up in the same session."""
    if not prior_runs:
        return ""
    ws = (workspace or get_settings().workspace_dir).resolve()

    sections: list[str] = []
    for i, run in enumerate(prior_runs, 1):
        if hasattr(run, "goal"):
            goal = run.goal
            result = run.result
        else:
            goal = run.get("goal", "")
            result = run.get("result")
        if not goal and not result:
            continue
        sections.append(_prior_run_section(i, goal=goal, result=result, workspace=ws))

    if not sections:
        return ""

    body = "\n\n".join(sections)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n…(prior session context truncated)"

    return (
        "CONTEXT FROM EARLIER RUNS IN THIS SESSION (same saved task tab):\n"
        "The user's current message may refer to findings below — e.g. 'that YouTube "
        "link', 'the video you found', 'download it'. Use the URLs and answers here; "
        "do not ask the user to repeat links that already appear below.\n"
        "Treat only the 'Existing local resources' list as currently available on disk. "
        "If a prior run lists a resource under 'Missing prior downloads', verify the "
        "path and re-download from the saved URL instead of assuming it still exists.\n\n"
        f"{body}"
    )
