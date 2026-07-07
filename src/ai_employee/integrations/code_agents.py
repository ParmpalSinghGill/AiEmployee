"""Delegate a coding task to an external agent CLI (Claude Code / Codex / Cursor).

When ``coding_backend`` is ``claude``, ``codex``, or ``cursor``, the coding agent
hands the whole task to that CLI, which writes AND runs the code in the workspace.
We invoke it non-interactively and capture its output. A missing or
unauthenticated CLI degrades to a clear message rather than crashing the run.

Command shape is fully config-driven (``*_cli_cmd`` / ``*_cli_args`` per backend)
so the integration adapts to CLI version differences without code changes — the
task prompt is appended as the final argument.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Any

from ai_employee.config import get_settings
from ai_employee.core.logging import get_logger
from ai_employee.tools.builtins._sandbox import ensure_workspace
from ai_employee.tools.builtins.code import _run_subprocess

log = get_logger("integrations.code_agent")

_PROMPT_PREFIX = (
    "You are an autonomous coding agent working inside a project workspace. "
    "Complete the following task end to end: create the necessary files and RUN the "
    "code to verify it works, fixing errors until it runs. Then summarize what you "
    "built, how to run it, and any remaining caveats.\n\nTASK:\n"
)


def _cli_config(backend: str) -> tuple[str, str] | None:
    """Return (command, extra_args) for an external coding backend, or None."""
    s = get_settings()
    mapping: dict[str, tuple[str, str]] = {
        "claude": (s.claude_cli_cmd, s.claude_cli_args),
        "codex": (s.codex_cli_cmd, s.codex_cli_args),
        "cursor": (s.cursor_cli_cmd, s.cursor_cli_args),
    }
    return mapping.get(backend)


def cli_available(backend: str) -> bool:
    """Whether the CLI for ``backend`` is installed and on PATH."""
    cfg = _cli_config(backend)
    if not cfg:
        return False
    cmd, _ = cfg
    return bool(cmd) and shutil.which(cmd) is not None


async def run_external_code_agent(
    backend: str, task: str, *, workspace: Path | None = None
) -> dict[str, Any]:
    """Run the external coding CLI on ``task`` and return a normalized result dict."""
    s = get_settings()
    prompt = _PROMPT_PREFIX + task
    cfg = _cli_config(backend)
    if not cfg:
        return {
            "ok": False,
            "output": f"Unknown coding backend {backend!r}.",
            "error": "bad_backend",
            "returncode": None,
            "backend": backend,
        }
    base, extra = cfg

    if shutil.which(base) is None:
        msg = (
            f"The '{backend}' CLI ('{base}') is not installed or not on PATH. Install it "
            "(and sign in), or switch the coding agent back to 'brain' in the UI."
        )
        log.warning("code_agent.cli_missing", backend=backend, cmd=base)
        return {
            "ok": False,
            "output": msg,
            "error": "cli_not_found",
            "returncode": None,
            "backend": backend,
        }

    argv = [base, *shlex.split(extra or ""), prompt]
    ws = ensure_workspace(workspace or s.workspace_dir)
    log.info("code_agent.exec", backend=backend, cmd=base, cwd=str(ws))
    code, out, err = await _run_subprocess(argv, ws, s.external_agent_timeout)
    combined = (out or "").strip()
    if err and err.strip():
        combined += "\n\n[stderr]\n" + err.strip()
    log.info("code_agent.done", backend=backend, returncode=code, ok=(code == 0))
    return {
        "ok": code == 0,
        "output": combined or f"({backend} produced no output; exit {code})",
        "error": None if code == 0 else (err[-1000:].strip() or f"{backend} exited {code}"),
        "returncode": code,
        "backend": backend,
    }
