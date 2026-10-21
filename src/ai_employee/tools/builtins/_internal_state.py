"""Guard agent tools against reading AIEmployee's own runtime state.

The shell tool runs with the nested task workspace as its CWD, but bash can still
reach sibling runtime files (``../audit_log.jsonl``, ``../sessions/...``). File
tools are sandboxed already, yet a restricted read should fail with a clear reason
instead of the generic sandbox message. This module centralises that policy.

The protected set is derived from settings (so relocated ``data/`` directories and
test-specific filenames are covered) plus the canonical production names.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from ai_employee.core.exceptions import ToolError

# Canonical runtime-state names. Kept explicit so protection survives settings
# that point at a different file (e.g. tests use ``test_*.json`` names).
_INTERNAL_FILES = frozenset(
    {
        "audit_log.jsonl",
        "usage_ledger.jsonl",
        "auth_secret.key",
        "test_auth_secret.key",
        "secrets.json",
        "google_accounts.json",
        "google_oauth_client.json",
    }
)
_INTERNAL_DIRS = frozenset({"sessions"})

_MESSAGE = RESTRICTED_MESSAGE = (
    "Access to AIEmployee internal platform state files is restricted. "
    "Work only within the task workspace."
)

# Commands that read/search a whole directory tree. Pointing one at the data root
# would sweep the protected files, so the root itself counts as a blocked target.
_DEEP_READ_COMMANDS = frozenset(
    {
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ag",
        "ack",
        "find",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "awk",
        "sed",
        "strings",
        "xxd",
        "od",
        "sort",
        "uniq",
        "wc",
        "file",
        "stat",
        "tar",
        "zip",
        "cp",
        "rsync",
    }
)
_SHELL_OPERATORS = frozenset({">", ">>", "<", "<<", "|", "||", "&&", ";", "&"})


@dataclass(frozen=True)
class CommandRestriction:
    """Why a shell command is blocked, and the target that triggered it."""

    target: str
    message: str


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:  # pragma: no cover - resolve rarely fails on Linux
        return path.expanduser().absolute()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parent_search_dirs(workspace: Path, roots: list[Path]) -> tuple[Path, ...]:
    """Directories above the workspace that recursive searches must not enter.

    Includes the workspace's parent (``grep -r ..``) and, when the workspace is
    under a configured runtime ``data/`` root, both that root and its parent (the
    application root). This blocks ``AIEmployee/`` and ``AIEmployee/data/`` while
    leaving ``AIEmployee/data/workspace`` and everything under it allowed.
    """
    ws = _resolved(workspace)
    dirs: list[Path] = [ws.parent]
    for root in roots:
        resolved = _resolved(root)
        if _is_within(ws, resolved):
            dirs.extend([resolved, resolved.parent])
    # Never treat the filesystem root as a "parent" — that would block every
    # unrelated absolute path. The workspace's real app/data roots are enough.
    return tuple(dict.fromkeys(path for path in dirs if path != Path("/")))


def _is_parent_search_target(candidate: Path, workspace: Path, roots: list[Path]) -> bool:
    """True when a recursive read would escape the workspace into a parent tree."""
    ws = _resolved(workspace)
    if _is_within(candidate, ws):
        return False
    return any(_is_within(candidate, parent) for parent in _parent_search_dirs(ws, roots))


def _glob_parent(token: str, workspace: Path) -> Path:
    path = Path(token)
    if not path.is_absolute():
        path = workspace / path
    try:
        return path.parent.resolve()
    except OSError:  # pragma: no cover - resolve rarely fails on Linux
        return path.parent.absolute()


def _parent_search_message(workspace: Path) -> str:
    ws = _resolved(workspace)
    return (
        "Do not search parent application directories; run `find .` or `grep` "
        f"strictly inside the current working directory ({ws}). "
        "Access to AIEmployee internal platform state files is restricted."
    )


def _settings():
    try:
        from ai_employee.config import get_settings

        return get_settings()
    except Exception:  # pragma: no cover - settings import should always work
        return None


def _sensitive_entries() -> tuple[set[str], set[str], list[Path]]:
    """Return (file names, directory names, data roots) for runtime state."""
    names = set(_INTERNAL_FILES)
    dirnames = set(_INTERNAL_DIRS)
    roots: list[Path] = []
    settings = _settings()
    if settings is not None:
        for field in (
            "audit_log_file",
            "usage_ledger_file",
            "auth_secret_file",
            "secrets_file",
            "google_accounts_file",
            "google_oauth_client_file",
        ):
            value = getattr(settings, field, None)
            if value:
                p = Path(value).expanduser().resolve()
                names.add(p.name)
                roots.append(p.parent)
        sessions = getattr(settings, "sessions_dir", None)
        if sessions:
            p = Path(sessions).expanduser().resolve()
            dirnames.add(p.name)
            roots.append(p.parent)
    # Deduplicate while keeping deterministic order.
    return names, dirnames, list(dict.fromkeys(roots))


def _resolve_candidate(raw: str, workspace: Path) -> Path | None:
    token = (raw or "").strip().strip("\"'")
    if not token:
        return None
    if token.startswith("-") and "=" in token:
        token = token.split("=", 1)[1]
    if not token or token.startswith("-"):
        return None
    if token in _SHELL_OPERATORS or set(token) <= set("0123456789<>&|"):
        return None
    if token.startswith("~"):
        token = str(Path(token).expanduser())
    p = Path(token)
    if not p.is_absolute():
        p = workspace / p
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


def _is_glob(token: str) -> bool:
    return any(ch in token for ch in "*?[")


def _is_restricted(candidate: Path, names: set[str], dirnames: set[str], roots: list[Path]) -> bool:
    for root in roots:
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        # Only direct children of the runtime data root are state files; a task
        # workspace nested under that root may legitimately contain same-named files.
        if len(parts) == 1 and parts[0] in names:
            return True
        if parts and parts[0] in dirnames:
            return True
    return False


def _restricted_glob(token: str, workspace: Path, dirnames: set[str], roots: list[Path]) -> bool:
    p = Path(token)
    if not p.is_absolute():
        p = workspace / p
    try:
        parent = p.parent.resolve()
    except OSError:
        parent = p.parent.absolute()
    for root in roots:
        if parent == root:
            return True
        try:
            rel = parent.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in dirnames:
            return True
    return False


def restricted_path_reason(path: str, workspace: Path) -> str | None:
    """Return the offending path when ``path`` targets protected runtime state."""
    names, dirnames, roots = _sensitive_entries()
    candidate = _resolve_candidate(path, workspace)
    if candidate is None:
        return None
    if _is_restricted(candidate, names, dirnames, roots):
        return str(candidate)
    return None


def check_path_access(path: str, workspace: Path) -> None:
    """Raise :class:`ToolError` if ``path`` targets protected runtime state."""
    if restricted_path_reason(path, workspace):
        raise ToolError(_MESSAGE, tool="filesystem")


def _tokens(command: str) -> list[str]:
    try:
        # ``punctuation_chars`` keeps shell separators (``;``, ``&&``, ``|``) as
        # their own tokens so chained commands are inspected segment by segment.
        lexer = shlex.shlex(command or "", posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return (command or "").split()


def _first_verb(tokens: list[str]) -> str:
    for token in tokens:
        if token in {"sudo", "env", "command", "time", "nice"}:
            continue
        if "=" in token and not token.startswith("-"):
            continue  # FOO=bar environment assignment
        return Path(token).name
    return ""


def _has_deep_read(tokens: list[str]) -> bool:
    """True if any command in a ``&&`` / ``;`` / pipe chain reads a tree.

    Checking only the first verb misses ``cd /parent && grep -rn …`` — the
    recursive read is in the second segment, but its ``cd`` target is the actual
    directory that would be swept.
    """
    segment: list[str] = []
    for token in [*tokens, ";"]:
        if token in {"&&", "||", ";", "|", "&"}:
            if _first_verb(segment) in _DEEP_READ_COMMANDS:
                return True
            segment = []
        else:
            segment.append(token)
    return False


def restricted_command(command: str, workspace: Path) -> CommandRestriction | None:
    """Return why a shell command is blocked, or ``None`` when it is safe.

    Direct runtime-state files and ``sessions/`` keep the internal-state message.
    Recursive reads of a parent application/data directory get the clearer
    parent-search message instead. Targets at or under the task workspace are
    always allowed.
    """
    names, dirnames, roots = _sensitive_entries()
    tokens = _tokens(command)
    workspace = _resolved(workspace)
    deep_read = _has_deep_read(tokens)

    for token in tokens:
        if _is_glob(token):
            if _restricted_glob(token, workspace, dirnames, roots):
                return CommandRestriction(token, RESTRICTED_MESSAGE)
            if deep_read and _is_parent_search_target(
                _glob_parent(token, workspace), workspace, roots
            ):
                return CommandRestriction(token, _parent_search_message(workspace))
            continue
        candidate = _resolve_candidate(token, workspace)
        if candidate is None:
            continue
        if _is_restricted(candidate, names, dirnames, roots):
            return CommandRestriction(str(candidate), RESTRICTED_MESSAGE)
        if deep_read and _is_parent_search_target(candidate, workspace, roots):
            return CommandRestriction(str(candidate), _parent_search_message(workspace))

    # Last resort: inline code (``python -c "open('../data/secrets.json')"``) does
    # not tokenise into clean paths. Catch explicit traversal to a protected name.
    for name in names:
        pattern = rf"(?:^|[\s'\"(/])\.\./(?:[\w.@+-]+/)*{re.escape(name)}\b"
        if re.search(pattern, command or ""):
            return CommandRestriction(name, RESTRICTED_MESSAGE)
    for dirname in dirnames:
        pattern = rf"(?:^|[\s'\"(/])\.\./(?:[\w.@+-]+/)*{re.escape(dirname)}/"
        if re.search(pattern, command or ""):
            return CommandRestriction(dirname, RESTRICTED_MESSAGE)

    # Absolute references to a data root plus a protected name.
    for root in roots:
        root_str = str(root).rstrip("/")
        if not root_str or root_str not in (command or ""):
            continue
        for name in names:
            if f"{root_str}/{name}" in command:
                return CommandRestriction(name, RESTRICTED_MESSAGE)
        for dirname in dirnames:
            if f"{root_str}/{dirname}" in command:
                return CommandRestriction(dirname, RESTRICTED_MESSAGE)
    return None


def restricted_path_in_command(command: str, workspace: Path) -> str | None:
    """Backward-compatible target-only helper for callers that need just the path."""
    restriction = restricted_command(command, workspace)
    return restriction.target if restriction else None
