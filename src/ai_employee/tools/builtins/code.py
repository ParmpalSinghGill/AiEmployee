"""Code & shell execution tools (sandboxed to the workspace, approval-gated)."""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._internal_state import restricted_command
from ai_employee.tools.builtins._sandbox import ensure_workspace


async def _run_subprocess(cmd: list[str], cwd: Path, timeout: int, stdin: str | None = None):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE if stdin else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin.encode() if stdin else None), timeout=timeout
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return 124, "", f"timed out after {timeout}s"
    except ProcessLookupError:
        # Child already exited/reaped (common with flaky external CLIs).
        return (
            proc.returncode if proc.returncode is not None else 1,
            "",
            "subprocess exited unexpectedly (ProcessLookupError)",
        )
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


# ── Shell output caps & redirection parsing ──────────────────────────────────
# Unbounded commands (recursive ``find``, huge directory walks) can dump thousands
# of lines into the agent's context window. Cap stdout/stderr deterministically so
# the model is told to narrow its query instead of drowning in tokens.
_SHELL_MAX_LINES = 200
_SHELL_HEAD_LINES = 150
_SHELL_TAIL_LINES = 50
_SHELL_MAX_CHARS = 12000

_TRUNCATION_NOTE = (
    "\n...[output truncated: {lines} lines / {chars} chars omitted. Narrow your "
    "query using head, tail, grep, or maxdepth flags]..."
)

# Standard descriptor redirections are read-only plumbing, not file writes:
#   `2>&1`, `1>&2`, `>&2`, `2>/dev/null`, `>/dev/null`
_STREAM_REDIRECT_RE = re.compile(r"(?:[12])?>&[12]\b|(?:[12])?>\s*/dev/null\b")


def _strip_stream_redirections(command: str) -> str:
    """Remove safe descriptor redirections so they aren't mistaken for writes."""
    return _STREAM_REDIRECT_RE.sub("", command)


def _truncate_output(text: str) -> str:
    """Deterministically cap shell output (lines, then chars) with a clear note."""
    if not text:
        return text
    omitted_lines = 0
    lines = text.split("\n")
    if len(lines) > _SHELL_MAX_LINES:
        omitted_lines = len(lines) - _SHELL_MAX_LINES
        lines = lines[:_SHELL_HEAD_LINES] + lines[-_SHELL_TAIL_LINES:]
    result = "\n".join(lines)
    omitted_chars = 0
    if len(result) > _SHELL_MAX_CHARS:
        omitted_chars = len(result) - _SHELL_MAX_CHARS
        result = result[:_SHELL_MAX_CHARS]
    if omitted_lines or omitted_chars:
        result += _TRUNCATION_NOTE.format(lines=omitted_lines, chars=omitted_chars)
    return result


# ``find / `` or ``find /home`` without ``-maxdepth`` scans an unbounded tree.
_UNBOUNDED_FIND_RE = re.compile(r"\bfind\s+/(?:\s|\b(?:home|root)\b)", re.I)


def _block_unbounded_traversal(command: str) -> str | None:
    """Reject unbounded root/home recursive scans before they run."""
    if "-maxdepth" in command or "find" not in command.lower():
        return None
    if _UNBOUNDED_FIND_RE.search(command):
        return (
            "Blocked unbounded traversal: `find` on the filesystem root or home "
            "directory without `-maxdepth` would scan a huge tree. Scope the search "
            "to a specific directory and add `-maxdepth` "
            "(e.g. `find ./data -maxdepth 3 -name '*.csv'`)."
        )
    return None


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute a short Python 3 script in a subprocess inside the workspace and "
        "return its stdout/stderr. Use for calculations, data wrangling, quick checks. "
        "When reading CSVs, pass the encoding from csv_preview "
        "(or encoding='latin-1' after UnicodeDecodeError). Do not assume UTF-8. "
        "Keep scripts under ~40 lines so the tool call is not truncated."
    )
    category = "code"

    class Args(BaseModel):
        code: str = Field(description="Python source to execute.")
        timeout: int = Field(30, ge=1, le=300)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: PythonExecTool.Args) -> ToolResult:
        ws = ensure_workspace(self._ws)
        # Same interpreter as pip_install / the running API process.
        code, out, err = await _run_subprocess([sys.executable, "-c", args.code], ws, args.timeout)
        return ToolResult(
            ok=(code == 0),
            output={"stdout": out, "stderr": err, "returncode": code},
            error=None if code == 0 else (err or f"exited {code}"),
        )


class PipInstallTool(Tool):
    name = "pip_install"
    description = (
        "Install Python packages into the running environment (e.g. 'opencv-python', "
        "'numpy>=1.26'). Use this BEFORE importing a third-party library that isn't "
        "available — do not shell out to pip. Only validated PyPI-style names are "
        "accepted. If a package name is wrong it will fail; do not retry the same name."
    )
    category = "code"
    # Deliberately NOT approval-gated: the name validation below makes this far
    # narrower than the general `shell` tool, so agents can self-provision deps
    # without stalling the run on an approval prompt.
    _NAME_RE = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._\-]*(\[[A-Za-z0-9_,.\-]+\])?([<>=!~]=?[0-9A-Za-z.\-*]+)?$"
    )

    class Args(BaseModel):
        packages: list[str] = Field(
            description="PyPI package specs, e.g. ['opencv-python', 'numpy>=1.26'].",
            min_length=1,
            max_length=20,
        )
        timeout: int = Field(300, ge=1, le=600)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: PipInstallTool.Args) -> ToolResult:
        pkgs = [p.strip() for p in args.packages]
        bad = [p for p in pkgs if not self._NAME_RE.match(p)]
        if bad:
            return ToolResult(
                ok=False,
                error=f"rejected unsafe package spec(s): {bad}. Use plain PyPI names/versions only.",
            )
        ws = ensure_workspace(self._ws)
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *pkgs]
        code, out, err = await _run_subprocess(cmd, ws, args.timeout)
        return ToolResult(
            ok=(code == 0),
            output={
                "stdout": out[-4000:],
                "stderr": err[-4000:],
                "returncode": code,
                "installed": pkgs,
            },
            error=None if code == 0 else (err[-1000:].strip() or f"pip exited {code}"),
        )


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command in the workspace. Read-only commands run freely; "
        "anything that writes/installs/deletes is approval-gated."
    )
    category = "code"
    approval_key = "shell_write"

    # Commands considered safe (read-only) and exempt from approval.
    _READ_ONLY = {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "echo",
        "pwd",
        "which",
        "stat",
        "file",
        "tree",
        "du",
        "df",
        "env",
        "date",
        "whoami",
    }

    class Args(BaseModel):
        command: str = Field(description="The shell command line to execute.")
        timeout: int = Field(60, ge=1, le=600)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    def _is_read_only(self, command: str) -> bool:
        stripped = _strip_stream_redirections(command)
        first = stripped.strip().split()[0] if stripped.strip() else ""
        return first in self._READ_ONLY and not any(
            tok in stripped for tok in (">", ">>", "|", "&", ";", "rm ", "mv ", "sudo")
        )

    def requires_approval(self, arguments: dict) -> str | None:  # type: ignore[override]
        command = arguments.get("command", "")
        if isinstance(command, str) and self._is_read_only(command):
            return None
        return "shell_write"

    async def _run(self, args: ShellTool.Args) -> ToolResult:
        guard = _block_unbounded_traversal(args.command)
        if guard:
            return ToolResult(ok=False, error=guard)
        restriction = restricted_command(args.command, self._ws)
        if restriction:
            return ToolResult(
                ok=False,
                error=f"{restriction.message} Blocked target: {restriction.target}",
            )
        ws = ensure_workspace(self._ws)
        code, out, err = await _run_subprocess(["bash", "-lc", args.command], ws, args.timeout)
        out = _truncate_output(out)
        err = _truncate_output(err)
        return ToolResult(
            ok=(code == 0),
            output={"stdout": out, "stderr": err, "returncode": code},
            error=None if code == 0 else (err or f"exited {code}"),
        )
