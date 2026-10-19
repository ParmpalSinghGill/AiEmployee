"""Programmatic ground-truth checks for an agent's claims about files it produced.

Agents narrate evidence in prose — "404 bytes", "23 lines", "1391 x 753 PNG" — and
that narration is sometimes arithmetically wrong or stale (e.g. quoting an
intermediate file from earlier in the same tool call). The LLM :class:`Verifier`
only ever sees that prose (see ``Verifier.verify``'s ``output`` argument); it has
no way to open the actual file, so it can't tell a genuine defect from the agent
just being sloppy describing a fine result.

This module closes that gap the same way :mod:`ai_employee.verification.resources`
does for downloads: run cheap, deterministic, no-LLM checks *before* the verifier,
and hand it concrete facts instead of trusting the agent's arithmetic. It only
depends on the standard library so it can run outside the code-execution sandbox.
"""

from __future__ import annotations

import csv
import re
import struct
from datetime import date, datetime
from pathlib import Path
from typing import Any

# File path followed by a claimed byte/line count or pixel dimensions, e.g.:
#   "aapl_last_30d.csv` | 404 bytes, 23 lines"
#   "aapl_close_chart.png` ... 99,669 bytes"
#   "~1391 px wide" / "1391 x 753" / "1391x753"
_FILE_RE = re.compile(r"[\w./-]+\.(?:csv|png|jpe?g|json|txt|md)\b", re.I)
_BYTES_RE = re.compile(r"([\d,]+)\s*bytes\b", re.I)
_LINES_RE = re.compile(r"([\d,]+)\s*(?:lines|rows)\b", re.I)
_DIMENSIONS_RE = re.compile(r"(\d{2,5})\s*(?:x|×|by)\s*(\d{2,5})\b", re.I)

# Only check claims made "close to" a filename mention, so an unrelated number
# elsewhere in a long answer isn't misread as a claim about that file.
_WINDOW = 200

# "last 1 year", "past 6 months", "last 30 days" — the common way a task phrases
# a relative lookback window. Plural/singular tolerant.
_RELATIVE_RANGE_RE = re.compile(
    r"\b(?:last|past)\s+(\d+)\s*(day|week|month|year)s?\b", re.I
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

# A bare 4-digit year followed by "-" then 2-digit month/day — cheap ISO date sniff,
# tried against each field of a CSV row's first few cells to find the date column.
_DATE_PATTERNS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d-%m-%Y")

# A fetch that comes up materially short of what was asked is a real shortfall —
# but "1 year" of trading-day data will never hit 365 calendar days, so this must
# have headroom for weekends/holidays/market-hours gaps, not just measurement noise.
_MIN_COVERAGE_RATIO = 0.5


def requested_days_from_task(text: str) -> int | None:
    """Best-effort parse of a relative lookback window mentioned in task text.

    Returns the number of calendar days requested (e.g. "last 1 year" -> 365), or
    None if the task doesn't state a relative range this way.
    """
    m = _RELATIVE_RANGE_RE.search(text or "")
    if not m:
        return None
    count = int(m.group(1))
    unit = m.group(2).lower()
    return count * _UNIT_DAYS[unit]


def _parse_date_cell(cell: str) -> date | None:
    cell = (cell or "").strip()
    if not cell:
        return None
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(cell[: len(fmt) + 2], fmt).date()
        except ValueError:
            continue
    return None


def _csv_date_span(path: Path) -> tuple[date, date] | None:
    """Sniff a date-like column in a CSV and return (earliest, latest) dates found.

    Deterministic and cheap: reads the header to find a plausible date/datetime
    column by name, then scans that column only. Returns None if no such column
    is found or no cell in it parses as a date — callers should treat that as
    "can't determine," not "empty range."
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return None
            date_col = next(
                (
                    i
                    for i, name in enumerate(header)
                    if re.search(r"date|time|timestamp", name, re.I)
                ),
                None,
            )
            if date_col is None:
                return None
            earliest: date | None = None
            latest: date | None = None
            for row in reader:
                if date_col >= len(row):
                    continue
                parsed = _parse_date_cell(row[date_col])
                if parsed is None:
                    continue
                if earliest is None or parsed < earliest:
                    earliest = parsed
                if latest is None or parsed > latest:
                    latest = parsed
            if earliest is None or latest is None:
                return None
            return earliest, latest
    except OSError:
        return None


def _resolve_path(path: str, workspace: Path) -> Path | None:
    raw = Path(path)
    for candidate in (raw, workspace / raw, workspace / raw.name):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read width/height straight from the PNG header (IHDR chunk) — no Pillow needed."""
    try:
        with path.open("rb") as f:
            header = f.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _claims_near(output: str, filename: str) -> str:
    """Slice of the output text around each mention of ``filename``, concatenated."""
    chunks = []
    for m in re.finditer(re.escape(filename), output):
        start = max(0, m.start() - _WINDOW)
        end = min(len(output), m.end() + _WINDOW)
        chunks.append(output[start:end])
    return "\n".join(chunks)


def audit_artifacts(
    *,
    output: str,
    workspace: Path | None = None,
    task: str = "",
) -> dict[str, Any]:
    """Cross-check file-related claims in ``output`` against the real files on disk.

    ``task`` is the step description an agent was given; when it states a relative
    lookback window ("last 1 year"), a saved CSV's actual date span is measured and
    compared against it — this catches a fetch that silently under-delivers (e.g. a
    provider limiting intraday history to 7 days when a year was requested) without
    relying on the agent to notice and self-report the shortfall.

    Returns ``{"applies": bool, "issues": [...], "checked": [...]}``. ``issues`` are
    concrete, independently-verified discrepancies (or missing files) worth failing
    a step over; an empty list with ``applies=True`` means every claim checked out.
    """
    ws = (workspace or Path(".")).resolve()
    filenames = list(dict.fromkeys(_FILE_RE.findall(output or "")))
    if not filenames:
        return {"applies": False, "issues": [], "checked": []}

    requested_days = requested_days_from_task(task)
    issues: list[str] = []
    checked: list[dict[str, Any]] = []

    for name in filenames:
        resolved = _resolve_path(name, ws)
        claim_text = _claims_near(output, name)
        if resolved is None:
            # Only flag as missing if the text actually claims the file exists/was
            # saved — plenty of mentions are just "will save to X" or a suggestion.
            if re.search(r"\b(saved|wrote|written|created|exists|bytes|lines)\b", claim_text, re.I):
                issues.append(f"Output claims '{name}' was produced but it does not exist on disk.")
            continue

        actual_bytes = resolved.stat().st_size
        entry: dict[str, Any] = {"file": name, "actual_bytes": actual_bytes}

        byte_claims = {_int(m) for m in _BYTES_RE.findall(claim_text)}
        # Drop claims equal to the real value before comparing so we only flag the
        # ones that actually mismatch (a file can be re-mentioned correctly later).
        for claimed in byte_claims - {actual_bytes}:
            issues.append(
                f"Output claims '{name}' is {claimed:,} bytes, but the file is actually "
                f"{actual_bytes:,} bytes."
            )

        if resolved.suffix.lower() in (".csv", ".txt", ".json", ".md"):
            try:
                actual_lines = sum(1 for _ in resolved.open("r", encoding="utf-8", errors="replace"))
            except OSError:
                actual_lines = None
            if actual_lines is not None:
                entry["actual_lines"] = actual_lines
                line_claims = {_int(m) for m in _LINES_RE.findall(claim_text)}
                for claimed in line_claims - {actual_lines}:
                    issues.append(
                        f"Output claims '{name}' has {claimed:,} lines/rows, but the file "
                        f"actually has {actual_lines:,}."
                    )

        if resolved.suffix.lower() == ".csv" and requested_days is not None:
            span = _csv_date_span(resolved)
            if span is not None:
                earliest, latest = span
                actual_days = (latest - earliest).days
                entry["actual_date_range"] = [earliest.isoformat(), latest.isoformat()]
                entry["actual_span_days"] = actual_days
                if actual_days < requested_days * _MIN_COVERAGE_RATIO:
                    issues.append(
                        f"Task asked for roughly {requested_days} days of history, but "
                        f"'{name}' only spans {actual_days} days ({earliest} to {latest}). "
                        "This looks like a provider limit (e.g. intraday data lookback) "
                        "that was not reported as a shortfall."
                    )

        if resolved.suffix.lower() == ".png":
            dims = _png_dimensions(resolved)
            if dims is not None:
                entry["actual_dimensions"] = dims
                for w, h in _DIMENSIONS_RE.findall(claim_text):
                    claimed_dims = (int(w), int(h))
                    if claimed_dims != dims and claimed_dims != dims[::-1]:
                        issues.append(
                            f"Output claims '{name}' is {claimed_dims[0]}x{claimed_dims[1]}, "
                            f"but the file is actually {dims[0]}x{dims[1]}."
                        )

        checked.append(entry)

    return {"applies": True, "issues": issues, "checked": checked}
