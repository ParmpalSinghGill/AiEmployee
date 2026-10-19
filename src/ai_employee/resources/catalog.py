"""Atomic, lock-protected JSON catalog of resources.

The catalog is the index, never the truth: a resource's bytes are re-verified
against the filesystem on read, so a file deleted outside the app is reported
missing rather than confidently handed to an agent. Writes go through a temp
file + ``os.replace`` under a :class:`~ai_employee.resources.locking.FileLock`,
so a crash or a second process can never leave a truncated or mixed ledger.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from ai_employee.core.logging import get_logger
from ai_employee.resources.locking import FileLock, lock_path_for
from ai_employee.resources.models import ResourceRecord

log = get_logger("resources.catalog")

SCHEMA_VERSION = 3

T = TypeVar("T")


class Catalog:
    """Load/save the resource records keyed by resource id."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = lock_path_for(self.path)

    # ── read ─────────────────────────────────────────────────────────────────
    def load(self) -> dict[str, ResourceRecord]:
        with FileLock(self.lock_path):
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, ResourceRecord]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("catalog.unreadable", path=str(self.path), error=str(exc))
            return {}
        raw = data.get("resources") if isinstance(data, dict) else data
        rows = raw.values() if isinstance(raw, dict) else (raw or [])
        records: dict[str, ResourceRecord] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                rec = ResourceRecord.from_dict(row)
            except Exception as exc:  # noqa: BLE001 - one bad row must not hide the rest
                log.warning("catalog.bad_row", error=str(exc))
                continue
            records[rec.id] = rec
        return records

    # ── write ────────────────────────────────────────────────────────────────
    def save(self, records: dict[str, ResourceRecord]) -> None:
        with FileLock(self.lock_path):
            self._save_unlocked(records)

    def _save_unlocked(self, records: dict[str, ResourceRecord]) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "resources": {rid: rec.to_dict() for rid, rec in records.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=self.path.parent,
                prefix=".catalog-",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as fh:
                tmp_name = fh.name
                json.dump(payload, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except OSError as exc:
            log.warning("catalog.write_failed", error=str(exc))
            if tmp_name:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
            raise

    # ── read-modify-write ────────────────────────────────────────────────────
    def mutate(self, fn: Callable[[dict[str, ResourceRecord]], T]) -> T:
        """Run ``fn(records)`` under the lock and persist any changes.

        ``fn`` mutates the mapping in place and may return a value.
        """
        with FileLock(self.lock_path):
            records = self._load_unlocked()
            result = fn(records)
            self._save_unlocked(records)
            return result


__all__ = ["Catalog", "SCHEMA_VERSION"]
