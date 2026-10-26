"""Bounded look-ups of files in the agent workspace, for the audits.

Agents often name a file without its folder ("capacity_grid.csv" for
``reports/capacity_grid.csv``) or report files an earlier attempt of the same
step produced. Checking only the exact path made real files look missing, and
the verifier then called correct work "fabricated". These helpers find a file by
name, and list what changed on disk since a step started, without ever walking
more than a bounded slice of the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Directories never worth descending into (tool caches, VCS internals, envs).
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".hf_home",
        ".cache",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
    }
)

_MAX_DEPTH = 5
_MAX_DIRS = 4000


def _walk(
    root: Path,
    *,
    max_depth: int = _MAX_DEPTH,
    max_dirs: int = _MAX_DIRS,
    skip_git_repos: bool = False,
    skipped_repos: list[Path] | None = None,
):
    """``os.walk`` capped by depth and by the number of directories visited.

    ``skip_git_repos`` leaves out checked-out repositories below the root (their
    changes are audited through git instead), so a large checkout cannot use up
    the walk budget before the rest of the workspace is seen.
    """
    root = root.resolve()
    base_depth = len(root.parts)
    for visited, (dirpath, dirnames, filenames) in enumerate(os.walk(root), start=1):
        if visited > max_dirs:
            return
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        if skip_git_repos:
            keep = []
            for d in dirnames:
                if (Path(dirpath) / d / ".git").exists():
                    if skipped_repos is not None:
                        skipped_repos.append(Path(dirpath) / d)
                else:
                    keep.append(d)
            dirnames[:] = keep
        if depth >= max_depth:
            dirnames[:] = []
        yield Path(dirpath), filenames


def find_by_name(workspace: Path, name: str) -> Path | None:
    """The newest file in ``workspace`` whose name is ``name`` (bounded search).

    Only for names with an extension — a bare word like "data" is not a file
    claim worth resolving.
    """
    name = Path(name).name
    if not name or "." not in name.strip("."):
        return None
    best: Path | None = None
    best_mtime = -1.0
    for dirpath, filenames in _walk(workspace):
        if name in filenames:
            candidate = dirpath / name
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if mtime > best_mtime:
                best, best_mtime = candidate, mtime
    return best


def files_changed_since(
    workspace: Path, since: float, *, limit: int = 40
) -> list[dict[str, object]]:
    """Files created or modified in ``workspace`` at or after ``since`` (newest first).

    Ground truth for "what did this step put on disk", independent of which
    tool wrote it (a download, a shell redirect, python, a git clone).
    """
    ws = workspace.resolve()
    found: list[tuple[float, Path, int]] = []
    repos: list[Path] = []
    for dirpath, filenames in _walk(ws, skip_git_repos=True, skipped_repos=repos):
        for fname in filenames:
            path = dirpath / fname
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_mtime >= since:
                found.append((st.st_mtime, path, st.st_size))
    # A repository cloned during the step shows up once, as itself.
    for repo in repos:
        try:
            # .git/config is written at clone time; .git itself changes on every
            # `git status` (index refresh), which would list old repos as new.
            created = (repo / ".git" / "config").stat().st_mtime
        except OSError:
            continue
        if created >= since:
            found.append((created, repo, -1))
    found.sort(key=lambda t: t[0], reverse=True)
    out: list[dict[str, object]] = []
    for _mtime, path, size in found[:limit]:
        try:
            rel = path.relative_to(ws).as_posix()
        except ValueError:
            rel = str(path)
        out.append(
            {"path": rel + "/", "kind": "repo"} if size < 0 else {"path": rel, "bytes": size}
        )
    return out


__all__ = ["SKIP_DIRS", "files_changed_since", "find_by_name"]
