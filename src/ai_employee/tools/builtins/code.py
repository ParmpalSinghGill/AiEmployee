"""Code & shell execution tools (sandboxed to the workspace, approval-gated)."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from ai_employee.tools.base import Tool, ToolResult
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
        proc.kill()
        return 124, "", f"timed out after {timeout}s"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute a short Python 3 script in a subprocess inside the workspace and "
        "return its stdout/stderr. Use for calculations, data wrangling, quick checks."
    )
    category = "code"

    class Args(BaseModel):
        code: str = Field(description="Python source to execute.")
        timeout: int = Field(30, ge=1, le=300)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: PythonExecTool.Args) -> ToolResult:
        ws = ensure_workspace(self._ws)
        code, out, err = await _run_subprocess(
            ["python", "-c", args.code], ws, args.timeout
        )
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
            output={"stdout": out[-4000:], "stderr": err[-4000:], "returncode": code, "installed": pkgs},
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
        "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd",
        "which", "stat", "file", "tree", "du", "df", "env", "date", "whoami",
    }

    class Args(BaseModel):
        command: str = Field(description="The shell command line to execute.")
        timeout: int = Field(60, ge=1, le=600)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    def _is_read_only(self, command: str) -> bool:
        first = command.strip().split()[0] if command.strip() else ""
        return first in self._READ_ONLY and not any(
            tok in command for tok in (">", ">>", "|", "&", ";", "rm ", "mv ", "sudo")
        )

    def requires_approval(self, arguments: dict) -> str | None:  # type: ignore[override]
        command = arguments.get("command", "")
        if isinstance(command, str) and self._is_read_only(command):
            return None
        return "shell_write"

    async def _run(self, args: ShellTool.Args) -> ToolResult:
        ws = ensure_workspace(self._ws)
        code, out, err = await _run_subprocess(
            ["bash", "-lc", args.command], ws, args.timeout
        )
        return ToolResult(
            ok=(code == 0),
            output={"stdout": out, "stderr": err, "returncode": code},
            error=None if code == 0 else (err or f"exited {code}"),
        )
