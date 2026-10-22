"""Git repositories checked out in the agent workspace.

Two features share this:

* the shell/python tools put a referenced Python repo on ``PYTHONPATH`` so its
  own test runner imports the checkout, not the copy installed in the env;
* the workflow checks whether a code-change step actually changed a repo.

Scanning is shallow (workspace children and grandchildren) so it stays cheap
even when the workspace also holds large datasets.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

_GIT = shutil.which("git")
_MAX_DEPTH = 2
_MAX_REPOS = 16
_MAX_DIRS_PER_LEVEL = 200
_GIT_TIMEOUT = 20

_PY_PROJECT_FILES = ("pyproject.toml", "setup.py", "setup.cfg")


def find_repos(workspace: Path, *, max_depth: int = _MAX_DEPTH) -> list[Path]:
    """Git work trees directly under ``workspace`` (up to ``max_depth`` levels)."""
    try:
        root = workspace.resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []
    found: list[Path] = []
    frontier = [root]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for directory in frontier:
            try:
                children = sorted(
                    p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
                )
            except OSError:
                continue
            for child in children:
                if (child / ".git").exists():
                    found.append(child)
                    if len(found) >= _MAX_REPOS:
                        return found
                else:
                    nxt.append(child)
        frontier = nxt[:_MAX_DIRS_PER_LEVEL]
    return found


def _rel(repo: Path, workspace: Path | None) -> str:
    if workspace is not None:
        try:
            return repo.relative_to(workspace.resolve()).as_posix()
        except ValueError:
            pass
    return repo.name


def references_repo(text: str, repo: Path, *, workspace: Path | None = None) -> bool:
    """True when ``text`` (a command, script or step) points into ``repo``.

    Only path-like references count — ``django/tests/x.py``, ``"django"``,
    ``cd django``, ``git -C django`` or the absolute path — so a plain
    ``import django`` does not.
    """
    if not text:
        return False
    if str(repo) in text:
        return True
    rel = re.escape(_rel(repo, workspace))
    patterns = (
        rf"(?:^|[\s\"'=(,:\[])(?:\./)?{rel}/",
        rf"[\"'](?:\./)?{rel}/?[\"']",
        rf"\bcd\s+(?:\./)?{rel}/?(?=$|[\s;&|)])",
        rf"\s-C\s+(?:\./)?{rel}/?(?=$|[\s;&|)])",
    )
    return any(re.search(p, text, re.M) for p in patterns)


def referenced_repos(text: str, workspace: Path) -> list[Path]:
    """The workspace repos that ``text`` references by path."""
    return [r for r in find_repos(workspace) if references_repo(text, r, workspace=workspace)]


def is_python_project(repo: Path) -> bool:
    return any((repo / name).exists() for name in _PY_PROJECT_FILES)


def pythonpath_for(text: str, workspace: Path) -> list[str]:
    """Import roots for the Python repos ``text`` references (repo root, plus ``src/``)."""
    roots: list[str] = []
    for repo in referenced_repos(text, workspace):
        if not is_python_project(repo):
            continue
        roots.append(str(repo))
        src = repo / "src"
        if src.is_dir():
            roots.append(str(src))
    return roots


def _git(repo: Path, *args: str) -> str | None:
    if not _GIT:
        return None
    try:
        proc = subprocess.run(
            [_GIT, "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def repo_state(repo: Path) -> dict[str, object] | None:
    """A fingerprint of ``repo``'s working tree (tracked diff + untracked files).

    ``None`` when git is unavailable or the directory is not a work tree.
    """
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status is None:
        return None
    diff = _git(repo, "diff", "HEAD") or ""
    head = (_git(repo, "rev-parse", "HEAD") or "").strip()
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(status.encode())
    digest.update(diff.encode(errors="replace"))
    # Untracked files have no diff — fold in their size/mtime so edits count.
    for line in status.splitlines():
        if line.startswith("?? "):
            path = repo / line[3:].strip().strip('"')
            try:
                st = path.stat()
                digest.update(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode())
            except OSError:
                continue
    return {"fingerprint": digest.hexdigest(), "dirty": bool(status.strip())}


def snapshot_repos(repos: list[Path]) -> dict[str, dict[str, object]]:
    """``{repo_path: repo_state}`` for every repo git could read."""
    out: dict[str, dict[str, object]] = {}
    for repo in repos:
        state = repo_state(repo)
        if state is not None:
            out[str(repo)] = state
    return out


__all__ = [
    "find_repos",
    "is_python_project",
    "pythonpath_for",
    "references_repo",
    "referenced_repos",
    "repo_state",
    "snapshot_repos",
]
