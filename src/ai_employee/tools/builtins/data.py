"""Lightweight data tools: CSV preview and JSON querying."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.tools.base import Tool, ToolResult
from ai_employee.tools.builtins._internal_state import check_path_access
from ai_employee.tools.builtins._sandbox import resolve_in_workspace

# utf-8-sig covers BOM and plain UTF-8. cp1252/latin-1 cover Superstore-style
# files that embed 0xA0 (nbsp) and other Windows bytes pandas rejects as UTF-8.
_TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def detect_text_encoding(path: Path, sample_bytes: int = 65_536) -> str:
    """Return an encoding that can decode ``path`` (latin-1 always can)."""
    raw = path.read_bytes()[:sample_bytes]
    if not raw:
        return "utf-8"
    for enc in _TEXT_ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


class CsvPreviewTool(Tool):
    name = "csv_preview"
    description = (
        "Preview a CSV file: column names, row count, first N rows, and the file "
        "`encoding` to pass to pandas (`pd.read_csv(path, encoding=...)`). "
        "Never assume UTF-8 — many public CSVs are cp1252/latin-1."
    )
    category = "data"

    class Args(BaseModel):
        path: str
        rows: int = Field(5, ge=1, le=100)

    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    async def _run(self, args: CsvPreviewTool.Args) -> ToolResult:
        import csv

        check_path_access(args.path, self._ws)
        p = resolve_in_workspace(self._ws, args.path, must_exist=True)
        encoding = detect_text_encoding(p)
        with p.open(newline="", encoding=encoding) as fh:
            reader = csv.reader(fh)
            all_rows = list(reader)
        if not all_rows:
            return ToolResult(
                output={"columns": [], "row_count": 0, "rows": [], "encoding": encoding}
            )
        header, *body = all_rows
        return ToolResult(
            output={
                "columns": header,
                "row_count": len(body),
                "rows": [dict(zip(header, r, strict=False)) for r in body[: args.rows]],
                "encoding": encoding,
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
