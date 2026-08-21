"""Sandboxed filesystem tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._sandbox import resolve_in_workspace


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace and return its contents."
    category = "filesystem"

    class Args(BaseModel):
        path: str = Field(description="Path relative to the workspace root.")
        max_bytes: int = Field(200_000, description="Truncate output beyond this many bytes.")

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: ReadFileTool.Args) -> ToolResult:
        p = resolve_in_workspace(self._ws, args.path, must_exist=True)
        data = p.read_text(encoding="utf-8", errors="replace")[: args.max_bytes]
        return ToolResult(output=data, metadata={"bytes": len(data)})


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a text file in the workspace."
    category = "filesystem"
    approval_key = "shell_write"  # mutating the filesystem is approval-gated

    class Args(BaseModel):
        path: str = Field(description="Path relative to the workspace root.")
        content: str
        append: bool = False

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: WriteFileTool.Args) -> ToolResult:
        p = resolve_in_workspace(self._ws, args.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with p.open(mode, encoding="utf-8") as fh:
            fh.write(args.content)
        return ToolResult(output=f"wrote {len(args.content)} chars to {args.path}")


class ListDirTool(Tool):
    name = "list_dir"
    description = (
        "List files and subdirectories of a workspace directory. Set recursive=true "
        "to see the whole tree in ONE call instead of walking level by level. "
        "For counting, measuring or parsing many files (image counts, dimensions, "
        "class totals) do NOT keep listing — use python_exec once and compute it."
    )
    category = "filesystem"

    class Args(BaseModel):
        path: str = Field(".", description="Directory relative to the workspace root.")
        recursive: bool = Field(
            False,
            description="Walk subdirectories too. Use this instead of repeated calls.",
        )
        max_entries: int = Field(
            400,
            description="Cap on returned entries; a truncated listing says so.",
        )

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: ListDirTool.Args) -> ToolResult:
        p = resolve_in_workspace(self._ws, args.path, must_exist=True)
        limit = max(1, int(args.max_entries))

        if not args.recursive:
            entries = [
                {
                    "name": e.name,
                    "type": "dir" if e.is_dir() else "file",
                    "size": e.stat().st_size if e.is_file() else None,
                }
                for e in sorted(p.iterdir())
            ]
            truncated = len(entries) > limit
            return ToolResult(
                output=entries[:limit],
                metadata={"count": len(entries), "truncated": truncated},
            )

        # Recursive: one call replaces the level-by-level walking that used to
        # trip the duplicate-call guard on deep dataset trees.
        import os

        rows: list[dict[str, Any]] = []
        total = 0
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames.sort()
            rel_dir = Path(dirpath).relative_to(p)
            for fn in sorted(filenames):
                total += 1
                if len(rows) < limit:
                    fp = Path(dirpath) / fn
                    try:
                        size = fp.stat().st_size
                    except OSError:
                        size = None
                    rows.append(
                        {
                            "name": str(rel_dir / fn) if str(rel_dir) != "." else fn,
                            "type": "file",
                            "size": size,
                        }
                    )
        note = (
            f"Truncated to {limit} of {total} files. Use python_exec to count or "
            "aggregate rather than listing again."
            if total > limit
            else None
        )
        return ToolResult(
            output=rows,
            metadata={"count": total, "truncated": total > limit, "note": note},
        )


class MakeDirTool(Tool):
    name = "make_dir"
    description = "Create a directory (and parents) in the workspace."
    category = "filesystem"
    approval_key = "shell_write"

    class Args(BaseModel):
        path: str

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: MakeDirTool.Args) -> ToolResult:
        p = resolve_in_workspace(self._ws, args.path)
        p.mkdir(parents=True, exist_ok=True)
        return ToolResult(output=f"created {args.path}")
