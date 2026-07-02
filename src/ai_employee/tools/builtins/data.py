"""Lightweight data tools: CSV preview and JSON querying."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._sandbox import resolve_in_workspace


class CsvPreviewTool(Tool):
    name = "csv_preview"
    description = "Preview a CSV file: column names, row count, and the first N rows."
    category = "data"

    class Args(BaseModel):
        path: str
        rows: int = Field(5, ge=1, le=100)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: CsvPreviewTool.Args) -> ToolResult:
        import csv

        p = resolve_in_workspace(self._ws, args.path, must_exist=True)
        with p.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            all_rows = list(reader)
        if not all_rows:
            return ToolResult(output={"columns": [], "row_count": 0, "rows": []})
        header, *body = all_rows
        return ToolResult(
            output={
                "columns": header,
                "row_count": len(body),
                "rows": [dict(zip(header, r, strict=False)) for r in body[: args.rows]],
            }
        )


class JsonQueryTool(Tool):
    name = "json_query"
    description = (
        "Navigate a JSON document with a dotted/bracketed path "
        "(e.g. 'data.items[0].name') and return the value at that path."
    )
    category = "data"

    class Args(BaseModel):
        data: dict[str, Any] | list[Any]
        path: str = Field(description="Dotted path, e.g. 'a.b[2].c'.")

    async def _run(self, args: JsonQueryTool.Args) -> ToolResult:
        import re

        cur: Any = args.data
        tokens = re.findall(r"[^.\[\]]+|\[\d+\]", args.path)
        for tok in tokens:
            if tok.startswith("[") and tok.endswith("]"):
                idx = int(tok[1:-1])
                cur = cur[idx]
            else:
                cur = cur[tok]
        return ToolResult(output=cur)
