"""Document loaders for many source types.

Each loader degrades gracefully if its optional dependency is missing, returning a
clear error rather than crashing ingestion. ``load_source`` dispatches on the
source kind (path extension, URL, or directory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadedDocument:
    source: str
    text: str
    content_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
    title: str | None = None


def load_source(source: str) -> list[LoadedDocument]:
    """Load one or many documents from a path, URL, or directory."""
    if source.startswith(("http://", "https://")):
        return [_load_url(source)]
    path = Path(source).expanduser()
    if path.is_dir():
        return _load_directory(path)
    if path.is_file():
        return [_load_file(path)]
    # Treat as raw text.
    return [LoadedDocument(source="<inline>", text=source, content_type="text")]


# ── Files ────────────────────────────────────────────────────────────────────
def _load_file(path: Path) -> LoadedDocument:
    ext = path.suffix.lower()
    loader = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".pptx": _load_pptx,
        ".csv": _load_csv,
        ".xlsx": _load_xlsx,
        ".xls": _load_xlsx,
        ".html": _load_html_file,
        ".htm": _load_html_file,
    }.get(ext, _load_text_file)
    return loader(path)


def _load_text_file(path: Path) -> LoadedDocument:
    ct = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "text"
    if path.suffix.lower() in {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp"}:
        ct = "code"
    return LoadedDocument(
        source=str(path),
        text=path.read_text(encoding="utf-8", errors="replace"),
        content_type=ct,
        title=path.name,
    )


def _load_pdf(path: Path) -> LoadedDocument:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        return LoadedDocument(source=str(path), text=text, content_type="pdf", title=path.name)
    except Exception as exc:  # noqa: BLE001
        return LoadedDocument(source=str(path), text="", metadata={"error": str(exc)})


def _load_docx(path: Path) -> LoadedDocument:
    try:
        import docx

        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return LoadedDocument(source=str(path), text=text, content_type="docx", title=path.name)
    except Exception as exc:  # noqa: BLE001
        return LoadedDocument(source=str(path), text="", metadata={"error": str(exc)})


def _load_pptx(path: Path) -> LoadedDocument:
    try:
        from pptx import Presentation

        prs = Presentation(str(path))
        parts: list[str] = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    parts.append(shape.text_frame.text)
        return LoadedDocument(
            source=str(path), text="\n".join(parts), content_type="pptx", title=path.name
        )
    except Exception as exc:  # noqa: BLE001
        return LoadedDocument(source=str(path), text="", metadata={"error": str(exc)})


def _load_csv(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return LoadedDocument(source=str(path), text=text, content_type="csv", title=path.name)


def _load_xlsx(path: Path) -> LoadedDocument:
    try:
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        parts: list[str] = []
        for ws in wb.worksheets:
            parts.append(f"# Sheet: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                parts.append(", ".join("" if c is None else str(c) for c in row))
        return LoadedDocument(
            source=str(path), text="\n".join(parts), content_type="xlsx", title=path.name
        )
    except Exception as exc:  # noqa: BLE001
        return LoadedDocument(source=str(path), text="", metadata={"error": str(exc)})


def _load_html_file(path: Path) -> LoadedDocument:
    from ai_employee.tools.builtins.web import _html_to_text

    text = _html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    return LoadedDocument(source=str(path), text=text, content_type="html", title=path.name)


# ── URL ──────────────────────────────────────────────────────────────────────
def _load_url(url: str) -> LoadedDocument:
    import httpx

    from ai_employee.tools.builtins.web import _html_to_text

    try:
        resp = httpx.get(
            url, follow_redirects=True, timeout=30, headers={"User-Agent": "AIEmployee/0.1"}
        )
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        text = _html_to_text(resp.text) if "html" in ct else resp.text
        return LoadedDocument(source=url, text=text, content_type="web", title=url)
    except Exception as exc:  # noqa: BLE001
        return LoadedDocument(source=url, text="", metadata={"error": str(exc)})


# ── Directory / codebase ─────────────────────────────────────────────────────
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}
_TEXT_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".sql",
    ".sh",
}


def _load_directory(root: Path, *, max_files: int = 500) -> list[LoadedDocument]:
    docs: list[LoadedDocument] = []
    for p in sorted(root.rglob("*")):
        if len(docs) >= max_files:
            break
        if p.is_dir() or any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in _TEXT_EXTS or p.suffix.lower() in {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
        }:
            doc = _load_file(p)
            if doc.text.strip():
                docs.append(doc)
    return docs
