"""Persistent token-usage ledger.

Appends one JSON line per LLM call — ``{ts, date, provider, model, prompt,
completion, total, purpose}`` — so cumulative usage *per brain* can be reported
and filtered by date. Append is O(1); aggregation reads and groups on demand.

Wired to the token choke point via :func:`ai_employee.core.usage.register_usage_sink`,
so every metered call lands here without any call site knowing.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("usage.ledger")

_lock = threading.Lock()


def make_sink(path: str | Path):
    """Build a ``record_usage`` sink that appends to the ledger at ``path``."""
    p = Path(path)

    def sink(provider: str, model: str, usage: Any, purpose: str) -> None:
        total = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
        if total <= 0:
            return
        now = datetime.now(UTC)
        rec = {
            "ts": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "provider": provider,
            "model": model,
            "purpose": purpose,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": total,
        }
        line = json.dumps(rec)
        with _lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(line + "\n")

    return sink


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _blank() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def _add(bucket: dict[str, int], rec: dict[str, Any]) -> None:
    bucket["prompt_tokens"] += rec.get("prompt_tokens", 0)
    bucket["completion_tokens"] += rec.get("completion_tokens", 0)
    bucket["total_tokens"] += rec.get("total_tokens", 0)
    bucket["calls"] += 1


def query(
    path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Aggregate usage, optionally restricted to the inclusive ``[start, end]``
    date range (``YYYY-MM-DD``). Returns overall totals plus breakdowns by brain
    (provider) and by date."""
    rows = _read(Path(path))
    total = _blank()
    by_provider: dict[str, dict[str, int]] = {}
    by_date: dict[str, dict[str, int]] = {}
    dates_seen: set[str] = set()

    for rec in rows:
        d = rec.get("date", "")
        if start and d < start:
            continue
        if end and d > end:
            continue
        dates_seen.add(d)
        _add(total, rec)
        _add(by_provider.setdefault(rec.get("provider", "unknown"), _blank()), rec)
        _add(by_date.setdefault(d, _blank()), rec)

    return {
        "range": {"start": start, "end": end},
        "total": total,
        "by_provider": dict(sorted(by_provider.items(), key=lambda kv: -kv[1]["total_tokens"])),
        "by_date": dict(sorted(by_date.items())),
        "days": len(dates_seen),
    }


__all__ = ["make_sink", "query"]
