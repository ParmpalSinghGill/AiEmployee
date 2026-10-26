"""What a step's tools actually did — as opposed to what the agent says it did.

Used by the agent loop (nudge a code-change step that has only been reading) and
by the verifier (judge test outcomes from real command results, not prose).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai_employee.core.workspace_repos import find_repos, references_repo, snapshot_repos

# A step whose description opens with one of these verbs must change files.
_CHANGE_VERB_RE = re.compile(
    r"^\W*(?:implement|apply|modify|patch|edit|refactor|fix|change|update|rewrite|add)\b",
    re.I,
)

_WRITE_TOOLS = frozenset({"write_file", "edit_file", "apply_patch", "replace_in_file"})

# Python that writes files: open(..., "w"/"a"/"x"/"r+"), pathlib writes, copies/moves.
_PY_WRITE_RE = re.compile(
    r"open\([^)]*?[\"'](?:[wax]|r\+)[bt+]*[\"']"
    r"|\.write_(?:text|bytes)\("
    r"|\bshutil\.(?:copy\w*|move)\("
    r"|\bos\.(?:replace|rename)\(",
)

# Shell that writes files (descriptor plumbing like 2>&1 / >/dev/null excluded).
_SHELL_WRITE_RE = re.compile(
    r"\bsed\s+(?:-\w*\s+)*-i"
    r"|\bperl\s+-\w*i"
    r"|\btee\b"
    r"|\bgit\s+apply\b"
    r"|\bpatch\s+-"
    r"|(?:^|[\s;&|])(?:cp|mv)\s"
    r"|>{1,2}\s*[^\s&>]",
)
# Shell writes that can appear inside a python_exec subprocess call. Redirects and
# cp/mv are left out: ``a > b`` is a comparison in Python.
_SHELL_IN_PY_RE = re.compile(r"\bsed\s+(?:-\w*\s+)*-i|\bperl\s+-\w*i|\bgit\s+apply\b|\bpatch\s+-")
_STREAM_REDIRECT_RE = re.compile(r"(?:[12])?>&[12]\b|(?:[12])?>\s*/dev/null\b")

_TEST_CMD_RE = re.compile(
    r"\bpytest\b|runtests\.py|manage\.py\s+test\b|\bunittest\b|\bbin/test\b"
    r"|\b(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b|\bgo\s+test\b|\bcargo\s+test\b"
    r"|\btox\b|\bnox\b|\bmake\s+test\b|\bjest\b|\bmocha\b|\bphpunit\b",
)

TAIL_CHARS = 600


def expects_code_change(description: str) -> bool:
    """Does this step's own description say it must change files?"""
    first = (description or "").strip().split("\n", 1)[0]
    return bool(_CHANGE_VERB_RE.match(first))


def _call_text(args: Any) -> str:
    if not isinstance(args, dict):
        return str(args or "")
    return str(args.get("code") or args.get("command") or "")


def is_mutating_call(tool: str, args: Any) -> bool:
    """Best-effort: did this tool call write to a file?"""
    if tool in _WRITE_TOOLS:
        return True
    text = _call_text(args)
    if tool == "python_exec":
        return bool(_PY_WRITE_RE.search(text) or _SHELL_IN_PY_RE.search(text))
    if tool == "shell":
        return bool(_SHELL_WRITE_RE.search(_STREAM_REDIRECT_RE.sub("", text)))
    return False


def is_test_call(tool: str, args: Any) -> bool:
    return tool in {"shell", "python_exec"} and bool(_TEST_CMD_RE.search(_call_text(args)))


def output_tail(output: Any, error: str | None = None, *, limit: int = TAIL_CHARS) -> str:
    """The end of a tool's stdout/stderr — where test runners print their summary."""
    if isinstance(output, dict):
        parts = [str(output.get("stdout") or ""), str(output.get("stderr") or "")]
        text = "\n".join(p.strip() for p in parts if p.strip())
        code = output.get("returncode")
        if code is not None:
            text += f"\n[returncode={code}]"
    else:
        text = str(output or "")
    if error and error not in text:
        text += f"\n[error] {error}"
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


_SAVED_PATH_RE = re.compile(r"""["']path["']\s*:\s*["']([^"']+)["']""")

#: Tools whose successful call saves files without being a "write" call.
_DOWNLOAD_TOOLS = frozenset({"download_resource"})


def summarize_tool_evidence(trace: list[dict[str, Any]], *, max_runs: int = 6) -> dict[str, Any]:
    """Test-command results, successful file writes and downloads in a step.

    ``trace`` may span several attempts of the step (the first try and its
    retries), so files an earlier attempt saved still count.
    """
    runs: list[dict[str, Any]] = []
    writes = 0
    downloads: list[str] = []
    for entry in trace or []:
        tool = str(entry.get("tool") or "")
        args = entry.get("args") or {}
        if entry.get("ok") and is_mutating_call(tool, args):
            writes += 1
        if entry.get("ok") and tool in _DOWNLOAD_TOOLS:
            m = _SAVED_PATH_RE.search(str(entry.get("output") or ""))
            downloads.append(m.group(1) if m else str(args.get("destination") or "?"))
        if is_test_call(tool, args):
            command = _call_text(args).strip()
            runs.append(
                {
                    "tool": tool,
                    "command": command[:300],
                    "ok": bool(entry.get("ok")),
                    "tail": entry.get("result_tail")
                    or output_tail(entry.get("output"), entry.get("error")),
                }
            )
    return {"test_runs": runs[-max_runs:], "write_calls": writes, "downloads": downloads}


# ── Did a code-change step change anything? ──────────────────────────────────


def snapshot_for_change_check(workspace: Path) -> dict[str, dict[str, object]]:
    """Fingerprint every workspace repo before a code-change step runs."""
    return snapshot_repos(find_repos(workspace))


def no_change_issue(
    *,
    before: dict[str, dict[str, object]],
    workspace: Path,
    scope_text: str,
) -> str | None:
    """An issue when the repos this step targets are byte-for-byte unchanged.

    Only repos the step text references, and only ones that were clean when the
    step started, are checked: a pre-dirty repo may legitimately already hold
    the change, and a step that names no repo is out of scope.
    """
    ws = workspace.resolve()
    scoped = [
        Path(p)
        for p, state in before.items()
        if not state.get("dirty") and references_repo(scope_text, Path(p), workspace=ws)
    ]
    if not scoped:
        return None
    after = snapshot_repos(scoped)
    if any(
        after.get(str(r), {}).get("fingerprint") != before[str(r)]["fingerprint"] for r in scoped
    ):
        return None
    names = ", ".join(r.name for r in scoped)
    return (
        f"No file in {names} changed during this step — the change was only described, "
        "not applied. Write the edit to the files on disk (write_file, or a python_exec "
        "that rewrites the file), then show `git diff`."
    )


__all__ = [
    "expects_code_change",
    "is_mutating_call",
    "is_test_call",
    "no_change_issue",
    "output_tail",
    "snapshot_for_change_check",
    "summarize_tool_evidence",
]
