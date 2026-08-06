"""Safely resolve and sample media files inside the workspace for the dashboard."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi", ".ogv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
_MEDIA_EXTS = _VIDEO_EXTS | _IMAGE_EXTS
_MAX_WALK_FILES = 500

# Directories that hold library assets, caches or VCS internals rather than the
# agent's own output. Without this, an installed package (matplotlib ships toolbar
# icons like home.png) floods the dashboard gallery with images nobody produced.
_SKIP_DIRS = {
    ".venv",
    "venv",
    "env",
    "site-packages",
    "node_modules",
    ".git",
    "__pycache__",
    ".cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    "mpl-data",
    ".conda",
    "dist-info",
}


def _is_library_path(path: Path, root: Path) -> bool:
    """True if ``path`` sits inside a dependency/cache directory under ``root``."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return any(part in _SKIP_DIRS for part in rel.parts[:-1])


def resolve_workspace_path(workspace: Path, path_str: str) -> Path:
    """Resolve ``path_str`` to a path inside ``workspace``.

    Accepts absolute paths under the workspace, or relative paths (optionally
    prefixed with a redundant ``data/workspace/`` segment).
    """
    ws = workspace.resolve()
    raw = (path_str or "").strip().strip("`\"'")
    if not raw or raw.upper() in {"NOT DOWNLOADED", "N/A", "NONE"}:
        raise ValueError("empty path")
    target = Path(raw)
    if not target.is_absolute():
        norm = raw.replace("\\", "/")
        while norm.startswith("./"):
            norm = norm[2:]
        # Agents often echo ``data/workspace/...`` even when workspace_dir is that root.
        for prefix in ("data/workspace/", "workspace/"):
            if norm.startswith(prefix):
                norm = norm[len(prefix) :]
                break
        parts = list(Path(norm).parts) if norm else []
        # Drop any leading segments that duplicate the workspace path suffix.
        ws_parts = list(ws.parts)
        for n in range(min(len(parts), len(ws_parts)), 0, -1):
            if parts[:n] == ws_parts[-n:]:
                parts = parts[n:]
                break
        if ".." in parts:
            raise ValueError("path must not contain '..'")
        target = ws.joinpath(*parts) if parts else ws
    target = target.resolve()
    if ws not in target.parents and target != ws:
        raise ValueError(f"path must be inside workspace ({ws})")
    return target


def workspace_relative(workspace: Path, path: Path) -> str:
    return str(path.resolve().relative_to(workspace.resolve()))


def media_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _IMAGE_EXTS:
        return "image"
    return None


def guess_media_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _collect_under(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if media_kind(root) else []
    if not root.is_dir():
        return []
    found: list[Path] = []
    # os.walk (not rglob) so dependency trees can be pruned instead of descended —
    # a single .venv can hold hundreds of images and exhaust the walk budget.
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if len(found) >= _MAX_WALK_FILES:
                found.sort(key=lambda p: str(p).lower())
                return found
            p = Path(dirpath) / fn
            if media_kind(p):
                found.append(p)
    found.sort(key=lambda p: str(p).lower())
    return found


def sample_media(
    workspace: Path,
    paths: list[str],
    *,
    sample_when_over: int = 5,
    sample_size: int = 3,
    modified_since: float | None = None,
) -> dict[str, Any]:
    """Collect video/image files under ``paths`` and sample when there are many.

    When the total media count exceeds ``sample_when_over``, only the first
    ``sample_size`` items are returned (split across videos then images by
    discovery order).

    ``modified_since`` (a POSIX timestamp) keeps only files written at or after
    that moment. The workspace is shared across runs, so without it a task's
    gallery shows every chart every previous task ever produced.
    """
    ws = workspace.resolve()
    seen: set[Path] = set()
    all_files: list[Path] = []
    for raw in paths:
        try:
            root = resolve_workspace_path(ws, raw)
        except ValueError:
            continue
        if not root.exists():
            continue
        for f in _collect_under(root):
            if f in seen:
                continue
            if modified_since is not None:
                try:
                    if f.stat().st_mtime < modified_since:
                        continue
                except OSError:
                    continue
            seen.add(f)
            all_files.append(f)

    total = len(all_files)
    sampled = total > sample_when_over
    chosen = all_files[:sample_size] if sampled else all_files

    videos: list[dict[str, str]] = []
    images: list[dict[str, str]] = []
    for f in chosen:
        rel = workspace_relative(ws, f)
        item = {
            "name": f.name,
            "path": rel,
            "url": f"/ui/api/workspace/file?path={_url_quote(rel)}",
            "mime": guess_media_type(f),
        }
        if media_kind(f) == "video":
            videos.append(item)
        else:
            images.append(item)

    return {
        "videos": videos,
        "images": images,
        "total": total,
        "shown": len(chosen),
        "sampled": sampled,
    }


def _url_quote(s: str) -> str:
    from urllib.parse import quote

    return quote(s, safe="/")


# Non-media artifacts worth surfacing as deliverables, grouped for display. Agents
# routinely produce spreadsheets, reports and dashboards that were previously
# invisible in the UI — only images had a home.
_DELIVERABLE_KINDS: dict[str, set[str]] = {
    "report": {".md", ".pdf", ".docx", ".txt", ".rst"},
    "spreadsheet": {".xlsx", ".xls", ".csv", ".tsv", ".parquet"},
    "page": {".html", ".htm"},
    "data": {".json", ".jsonl", ".yaml", ".yml", ".xml"},
    "notebook": {".ipynb"},
    "code": {".py", ".sql", ".sh", ".js", ".ts"},
    "archive": {".zip", ".tar", ".gz", ".tgz"},
}
_DELIVERABLE_EXTS = {e: k for k, exts in _DELIVERABLE_KINDS.items() for e in exts}
_MAX_DELIVERABLES = 80


def deliverable_kind(path: Path) -> str | None:
    return _DELIVERABLE_EXTS.get(path.suffix.lower())


def collect_deliverables(
    workspace: Path,
    *,
    modified_since: float | None = None,
    limit: int = _MAX_DELIVERABLES,
) -> dict[str, Any]:
    """List non-image artifacts a run produced (reports, spreadsheets, dashboards).

    Mirrors :func:`sample_media` but for files a user would download rather than
    preview inline. ``modified_since`` scopes results to one run, since the
    workspace is shared across runs.
    """
    import os

    ws = workspace.resolve()
    if not ws.is_dir():
        return {"files": [], "total": 0, "shown": 0, "truncated": False}

    found: list[tuple[float, Path]] = []
    for dirpath, dirnames, filenames in os.walk(ws):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if deliverable_kind(p) is None:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if modified_since is not None and st.st_mtime < modified_since:
                continue
            found.append((st.st_mtime, p))

    total = len(found)
    # Newest first — the run's final deliverables are what the user wants.
    found.sort(key=lambda t: (-t[0], str(t[1]).lower()))
    chosen = found[:limit]

    files: list[dict[str, Any]] = []
    for mtime, p in chosen:
        rel = workspace_relative(ws, p)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        files.append(
            {
                "name": p.name,
                "path": rel,
                "kind": deliverable_kind(p),
                "size": size,
                "modified": mtime,
                "url": f"/ui/api/workspace/file?path={_url_quote(rel)}",
            }
        )

    return {
        "files": files,
        "total": total,
        "shown": len(files),
        "truncated": total > len(files),
    }


__all__ = [
    "collect_deliverables",
    "deliverable_kind",
    "guess_media_type",
    "media_kind",
    "resolve_workspace_path",
    "sample_media",
    "workspace_relative",
]
