"""Programmatic ground-truth checks for an agent's claims about filesystem paths.

The LLM :class:`Verifier` only reads an agent's prose; it cannot tell whether a
path the agent claims to have created / cloned / checked out actually exists, or
whether a quoted git commit SHA is real. This module measures those facts directly
from disk — existence, kind, directory contents, and the resolved git ``HEAD`` —
so the verifier gets ground truth instead of trusting the agent's self-report.

Mirrors :mod:`ai_employee.verification.artifacts` (which does the same for byte /
line / pixel claims). Only depends on the standard library.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Absolute Unix paths — require at least one inner slash so single-segment roots
# and device names (e.g. "/dev/sda", "/") don't flood the audit. URLs don't match:
# they start "https:" and the leading token is not a "/".
_ABS_PATH_RE = re.compile(r"/(?:[\w@.+~-]+/)+[\w@.+~-]+")

# "../"-relative paths — the class that most often resolves somewhere the agent
# didn't intend (relative to the sandbox CWD rather than the repo root).
_REL_PATH_RE = re.compile(r"(?:\.\./)+(?:[\w@.+~-]+/)*[\w@.+~-]+(?:\.\w+)?")

_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", re.I)

# Verbs/forms that *assert* a path exists. Only these turn a bare "mention" of a
# path into a claim worth flagging when the path is actually absent on disk.
_CLAIM_RE = re.compile(
    r"\b(?:exists|created|saved|wrote|written|cloned|checked out|checkout|verified|"
    r"located|resides|resolves to|is at|under)\b",
    re.I,
)

# A "does not exist"-style negation next to a path should NOT be flagged as a claim.
_NEG_RE = re.compile(r"\b(?:does not|doesn't|not)\s+exist\b", re.I)

_MAX_PATHS = 12
_MAX_ENTRIES = 12
_WINDOW = 160


def _strip(raw: str) -> str:
    return raw.rstrip(".,;:)]}\"'`")


def _resolve(raw: str, workspace: Path) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (workspace / p).resolve()


def _git_head(path: Path) -> str | None:
    """Return the resolved HEAD sha of a git repository at ``path``, or None."""
    git = path / ".git"
    try:
        if git.is_file():  # worktree / submodule pointer: ".git" is a file
            body = git.read_text(encoding="utf-8", errors="replace").strip()
            m = re.search(r"gitdir:\s*(\S+)", body)
            if m:
                git = (path / m.group(1)).resolve()
    except OSError:
        return None
    head_file = git / "HEAD"
    try:
        raw = head_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if raw.startswith("ref:"):
        ref = raw[4:].strip()
        try:
            sha = (git / ref).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            sha = _packed_ref(git, ref)
        return sha or None
    return raw or None


def _packed_ref(git: Path, ref: str) -> str | None:
    try:
        lines = (git / "packed-refs").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == ref:
            return parts[0]
    return None


def _entries(path: Path) -> list[str]:
    try:
        return sorted(p.name for p in path.iterdir())[:_MAX_ENTRIES]
    except OSError:
        return []


def _claims_near(output: str, path_str: str) -> str:
    chunks: list[str] = []
    for m in re.finditer(re.escape(path_str), output):
        s = max(0, m.start() - _WINDOW)
        e = min(len(output), m.end() + _WINDOW)
        chunks.append(output[s:e])
    return "\n".join(chunks)


def audit_filesystem(*, output: str, workspace: Path | None = None) -> dict[str, Any]:
    """Cross-check path claims in ``output`` against the real filesystem.

    Returns ``{"applies": bool, "issues": [...], "checked": [...]}`` where
    ``checked`` lists ground-truth facts per path (existence, kind, directory
    entries, git HEAD) and ``issues`` lists hard contradictions (a path claimed to
    exist that doesn't, or a quoted commit SHA that doesn't match the measured
    HEAD). ``applies=False`` means no path-like string was mentioned.
    """
    ws = (workspace or Path(".")).resolve()
    raw_paths: list[str] = []
    raw_paths += [_strip(m.group(0)) for m in _ABS_PATH_RE.finditer(output or "")]
    raw_paths += [_strip(m.group(0)) for m in _REL_PATH_RE.finditer(output or "")]

    seen: set[str] = set()
    paths: list[str] = []
    for p in raw_paths:
        if p in seen:
            continue
        seen.add(p)
        paths.append(p)
        if len(paths) >= _MAX_PATHS:
            break
    if not paths:
        return {"applies": False, "issues": [], "checked": []}

    issues: list[str] = []
    checked: list[dict[str, Any]] = []
    for p in paths:
        resolved = _resolve(p, ws)
        near = _claims_near(output, p)
        if not resolved.exists():
            checked.append({"path": p, "exists": False})
            if _CLAIM_RE.search(near) and not _NEG_RE.search(near):
                issues.append(f"Output claims '{p}' exists but it does not exist on disk.")
            continue
        if not resolved.is_dir() and not resolved.is_file():
            # Devices, sockets, fifos — nothing useful to report to the verifier.
            continue

        entry: dict[str, Any] = {
            "path": p,
            "exists": True,
            "kind": "dir" if resolved.is_dir() else "file",
        }
        if resolved.is_dir():
            names = _entries(resolved)
            entry["entries"] = names
            entry["entry_count"] = len(names)
            head = _git_head(resolved)
            if head:
                entry["git_head"] = head
                for claimed in {c.lower() for c in _SHA_RE.findall(near)}:
                    if claimed != head.lower():
                        issues.append(
                            f"Output quotes commit '{claimed}' for '{p}', but its HEAD is '{head}'."
                        )
        else:
            try:
                entry["bytes"] = resolved.stat().st_size
            except OSError:
                entry["bytes"] = None
        checked.append(entry)

    return {"applies": True, "issues": issues, "checked": checked}
