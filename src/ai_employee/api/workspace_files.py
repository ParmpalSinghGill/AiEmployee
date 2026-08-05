"""Safely resolve and sample media files inside the workspace for the dashboard."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi", ".ogv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
_MEDIA_EXTS = _VIDEO_EXTS | _IMAGE_EXTS
_MAX_WALK_FILES = 500


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
    for p in root.rglob("*"):
        if len(found) >= _MAX_WALK_FILES:
            break
        if p.is_file() and media_kind(p):
            found.append(p)
    found.sort(key=lambda p: str(p).lower())
    return found


def sample_media(
    workspace: Path,
    paths: list[str],
    *,
    sample_when_over: int = 5,
    sample_size: int = 3,
) -> dict[str, Any]:
    """Collect video/image files under ``paths`` and sample when there are many.

    When the total media count exceeds ``sample_when_over``, only the first
    ``sample_size`` items are returned (split across videos then images by
    discovery order).
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
            if f not in seen:
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


__all__ = [
    "guess_media_type",
    "media_kind",
    "resolve_workspace_path",
    "sample_media",
    "workspace_relative",
]
