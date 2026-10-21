"""Programmatic ground-truth checks for an agent's claims about filesystem paths.

The LLM :class:`Verifier` only reads an agent's prose; it cannot tell whether a
path the agent claims to have created / cloned / checked out actually exists, or
whether a quoted git commit SHA is real. This module measures those facts directly
from disk — existence, kind, directory contents, the resolved git ``HEAD``, and —
for repositories — the working-tree status/diff — so the verifier gets ground
truth instead of trusting the agent's self-report.

Mirrors :mod:`ai_employee.verification.artifacts` (which does the same for byte /
line / pixel claims). Only depends on the standard library.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ai_employee.core.text_paths import clean_for_path_extraction, has_diff_claim

# Absolute Unix paths — require at least one inner slash so single-segment roots
# and device names (e.g. "/dev/sda", "/") don't flood the audit. URLs and git diff
# headers are stripped before matching (see ``clean_for_path_extraction``).
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

# Bounded repository discovery: small depth, few repos, short git timeouts. The
# audit runs after every step, so it must never turn into a workspace-sized walk.
_GIT = shutil.which("git")
_GIT_TIMEOUT = 5.0
_MAX_REPOS = 8
_REPO_SCAN_DEPTH = 3
_SKIP_WALK_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "site-packages",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        ".eggs",
    }
)

# Reflog actions that mean a commit/rebase genuinely landed. A fresh clone shows
# "clone: from ..." and must not be mistaken for changes saved by the agent.
_COMMIT_REFLOG_PREFIXES = (
    "commit",
    "merge",
    "rebase",
    "cherry-pick",
    "am",
    "revert",
    "pull",
)
_RECENT_COMMIT_WINDOW = 900.0

ZERO_MODIFICATIONS_ISSUE = (
    "Output presents a git diff claiming code modifications, but the git working "
    "tree on disk has zero modifications. Changes must be physically saved to disk."
)


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
        sha: str | None
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


def _git(repo: Path, *args: str) -> str:
    """Run a git query in ``repo`` and return stdout, or "" on any failure."""
    if not _GIT:
        return ""
    try:
        proc = subprocess.run(
            [_GIT, "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _is_repo_root(path: Path) -> bool:
    return (path / ".git").exists()


def _discover_repos(root: Path) -> list[Path]:
    """Find git repositories under ``root`` without walking their internals."""
    if not root.exists() or not root.is_dir():
        return []
    try:
        root = root.resolve()
    except OSError:
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if ".git" in dirnames or ".git" in filenames:
            found.append(current)
            if len(found) >= _MAX_REPOS:
                break
            # A repository's own tree is not worth scanning for nested repos.
            dirnames[:] = []
            continue
        try:
            depth = len(current.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth >= _REPO_SCAN_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SKIP_WALK_DIRS and not d.startswith(".")
        )
    return found[:_MAX_REPOS]


def _repo_for_path(path: Path) -> Path | None:
    """Resolve the repository containing ``path`` (file or directory), if any."""
    if not _GIT or not path.exists():
        return None
    probe = path if path.is_dir() else path.parent
    top = _git(probe, "rev-parse", "--show-toplevel")
    if not top:
        return None
    try:
        return Path(top).resolve()
    except OSError:
        return None


def _status_summary(status_lines: list[str]) -> str:
    if not status_lines:
        return "clean"
    untracked = sum(1 for line in status_lines if line.startswith("??"))
    tracked = len(status_lines) - untracked
    bits: list[str] = []
    if tracked:
        bits.append(f"{tracked} tracked change(s)")
    if untracked:
        bits.append(f"{untracked} untracked file(s)")
    return ", ".join(bits) or f"{len(status_lines)} change(s)"


def _stat_summary(stat: str) -> str:
    if not stat.strip():
        return "none"
    return stat.strip().splitlines()[-1].strip()[:400]


def _made_new_commit(repo: Path, *, since: float | None) -> bool:
    """True when HEAD moved by a commit-like action at/after ``since``."""
    if not _GIT:
        return False
    cutoff = since - 1.0 if since is not None else time.time() - _RECENT_COMMIT_WINDOW
    reflog = _git(repo, "reflog", "show", "-n", "12", "--format=%ct%x09%gs")
    for line in reflog.splitlines():
        ts, _, subject = line.partition("\t")
        try:
            when = float(ts)
        except ValueError:
            continue
        if when < cutoff:
            continue
        if subject.strip().lower().startswith(_COMMIT_REFLOG_PREFIXES):
            return True
    return False


def _git_facts(repo: Path, *, since: float | None = None) -> dict[str, Any]:
    """Measure working-tree state for ``repo``; {} when it is not a work tree."""
    if not _GIT or not _is_repo_root(repo):
        return {}
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        return {}
    status_lines = [line for line in _git(repo, "status", "--porcelain").splitlines() if line]
    status_summary = _status_summary(status_lines)
    tracked_changes = any(
        not line.startswith("??") and not line.startswith("!!") for line in status_lines
    )
    return {
        "git_head": _git(repo, "rev-parse", "HEAD") or _git_head(repo),
        "git_branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_status": status_summary,
        "git_diff_stat": _stat_summary(_git(repo, "diff", "--stat")),
        "git_staged_stat": _stat_summary(_git(repo, "diff", "--cached", "--stat")),
        "git_has_changes": bool(status_lines),
        "git_tracked_changes": tracked_changes,
        "git_new_commit": _made_new_commit(repo, since=since),
    }


def audit_filesystem(
    *, output: str, workspace: Path | None = None, since: float | None = None
) -> dict[str, Any]:
    """Cross-check path claims in ``output`` against the real filesystem.

    Returns ``{"applies": bool, "issues": [...], "checked": [...]}`` where
    ``checked`` lists ground-truth facts per path (existence, kind, directory
    entries, git HEAD/status/diff) and ``issues`` lists hard contradictions (a
    path claimed to exist that doesn't, a quoted commit SHA that doesn't match the
    measured HEAD, or a prose diff with no matching change on disk).
    ``applies=False`` means no path-like string or diff claim was mentioned.

    ``since`` is the wall-clock time the step started. It lets a commit made during
    this step count as "changes saved", while a pre-existing clean checkout does
    not. When omitted, a short recency window is used.
    """
    text = output or ""
    ws = (workspace or Path(".")).resolve()
    cleaned = clean_for_path_extraction(text)
    raw_paths: list[str] = []
    raw_paths += [_strip(m.group(0)) for m in _ABS_PATH_RE.finditer(cleaned)]
    raw_paths += [_strip(m.group(0)) for m in _REL_PATH_RE.finditer(cleaned)]

    seen: set[str] = set()
    paths: list[str] = []
    for p in raw_paths:
        if p in seen:
            continue
        seen.add(p)
        paths.append(p)
        if len(paths) >= _MAX_PATHS:
            break

    diff_claim = has_diff_claim(text)
    discovered = _discover_repos(ws)
    if not paths and not diff_claim:
        return {"applies": False, "issues": [], "checked": []}

    issues: list[str] = []
    checked: list[dict[str, Any]] = []
    resolved_paths: list[Path] = []
    for p in paths:
        resolved = _resolve(p, ws)
        resolved_paths.append(resolved)
        near = _claims_near(text, p)
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

    # Detect repositories in the workspace and in referenced target directories.
    repos: list[Path] = list(discovered)
    seen_repos = {str(r) for r in repos}
    for resolved in resolved_paths:
        repo = _repo_for_path(resolved)
        if repo is not None and str(repo) not in seen_repos and len(repos) < _MAX_REPOS:
            seen_repos.add(str(repo))
            repos.append(repo)

    repo_facts: list[dict[str, Any]] = []
    for repo in repos:
        facts = _git_facts(repo, since=since)
        if not facts:
            continue
        repo_facts.append(facts)
        existing = next(
            (c for c in checked if c.get("exists") and _resolve(c["path"], ws).resolve() == repo),
            None,
        )
        if existing is not None:
            existing.update(facts)
            existing.setdefault("kind", "dir")
            continue
        names = _entries(repo)
        checked.append(
            {
                "path": str(repo),
                "exists": True,
                "kind": "dir",
                "entries": names,
                "entry_count": len(names),
                **facts,
            }
        )

    if diff_claim and repo_facts:
        saved = any(
            facts.get("git_has_changes") or facts.get("git_new_commit") for facts in repo_facts
        )
        if not saved:
            issues.append(ZERO_MODIFICATIONS_ISSUE)

    return {"applies": True, "issues": issues, "checked": checked}
