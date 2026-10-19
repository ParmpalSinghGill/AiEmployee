"""Safe task and resource names, and workspace path confinement.

Two things this module guarantees:

* **Names are filesystem-safe.** A task id, resource name or slug that arrives
  from an LLM/user is normalized to ``[a-z0-9._-]`` — never ``..``, never an
  absolute path, never a path separator — so it can be used as a directory name.
* **Paths stay inside their root.** :func:`contained` resolves a candidate and
  refuses anything that escapes ``root`` (symlinks included), which is the
  foundation of task/workspace isolation.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

from ai_employee.resources.errors import ResourcePathError

#: Anything outside this set collapses to a single ``-``.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DOTS_RE = re.compile(r"\.{2,}")
_MAX_SLUG = 80


def safe_slug(text: str, *, default: str = "item", max_len: int = _MAX_SLUG) -> str:
    """Normalize ``text`` into a single safe path segment.

    ``"../../etc/passwd"`` → ``"etc-passwd"``; ``"ÜnÍcode Name!"`` → ``"nicode-name"``.
    """
    raw = (text or "").strip().lower()
    collapsed = _UNSAFE_RE.sub("-", raw)
    collapsed = _DOTS_RE.sub("-", collapsed)
    collapsed = collapsed.strip("-._")
    if not collapsed:
        return default
    return collapsed[:max_len].strip("-._") or default


def safe_resource_name(name: str, *, default: str = "resource") -> str:
    """Like :func:`safe_slug` but preserves a single file extension.

    ``"YOLO26n.pt"`` → ``"yolo26n.pt"``; ``"a/b/c.tar.gz"`` → ``"c-tar.gz"``.
    """
    stem = Path((name or "").strip()).name  # drop any directory components first
    if not stem:
        return default
    suffix = Path(stem).suffix
    base = safe_slug(Path(stem).stem.replace(".", "-"), default=default)
    if suffix and re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix):
        return f"{base}{suffix.lower()}"
    return base


def task_date_parts(when: datetime | date | None = None) -> tuple[str, str, str]:
    """``(YYYY, MM, DD)`` for a date, defaulting to today (UTC)."""
    d = when.date() if isinstance(when, datetime) else when
    d = d or datetime.now(UTC).date()
    return f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}"


def safe_task_identity(task_id: str | None) -> str:
    """A **stable** filesystem-safe task identity.

    Ownership is keyed on this, so it must not change with the calendar — only
    the on-disk *location* is date-based (see :func:`task_date_parts`).
    """
    return safe_slug(task_id or "", default="task")


def task_date_key(when: datetime | date | None = None) -> str:
    """``YYYY/MM/DD`` — the date-based organization segment for task storage."""
    return "/".join(task_date_parts(when))


def validate_relative(relative: str | Path) -> Path:
    """Return ``relative`` as a safe relative :class:`Path`, or raise.

    Rejects absolute paths and any ``..`` segment.
    """
    if isinstance(relative, Path):
        parts = relative.parts
    else:
        parts = Path(str(relative).replace("\\", "/")).parts
    if not parts:
        raise ResourcePathError("empty relative path")
    for part in parts:
        if part in ("..",):
            raise ResourcePathError(f"path must not contain '..': {relative!r}")
        if part in ("", "."):
            continue
        if Path(part).is_absolute() or ":" in part and Path(part).drive:
            raise ResourcePathError(f"path must be relative: {relative!r}")
    p = Path(*[p for p in parts if p not in ("", ".")])
    if p.is_absolute() or p.drive:
        raise ResourcePathError(f"path must be relative: {relative!r}")
    return p


def contained(root: Path, *parts: str | Path) -> Path:
    """Resolve ``parts`` under ``root`` and refuse any escape (incl. symlinks)."""
    root = root.resolve()
    candidate = root.joinpath(*[str(p) for p in parts])
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise ResourcePathError(f"cannot resolve path {candidate}: {exc}") from exc
    if resolved != root and root not in resolved.parents:
        raise ResourcePathError(
            f"path escapes root {root}: {candidate}",
            details={"candidate": str(candidate), "root": str(root)},
        )
    return resolved


def relative_to(root: Path, path: Path) -> str:
    """Workspace-relative POSIX form of ``path``; raises if it escapes ``root``."""
    root = root.resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ResourcePathError(
            f"path {path} is not inside {root}",
            details={"path": str(path), "root": str(root)},
        ) from exc


__all__ = [
    "safe_slug",
    "safe_resource_name",
    "safe_task_identity",
    "task_date_key",
    "task_date_parts",
    "validate_relative",
    "contained",
    "relative_to",
]
