"""Disk persistence for sessions, tasks and memory — the "save & resume" store.

Layout (under ``sessions_dir``, default ``data/sessions/``):

    data/sessions/
      <session_id>/
        session.json          # {id, title, created_at, run_ids}
        runs/<task_id>.json   # full task snapshot: goal, status, events, result
      _memory.jsonl           # exported memory records (for resume context)

Everything is plain JSON so a session is human-readable and portable: the goal,
the agent/tool event history, and the result ("what it built") are all there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("store")


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Sessions & tasks ─────────────────────────────────────────────────────
    def _sdir(self, session_id: str) -> Path:
        d = self.root / session_id
        (d / "runs").mkdir(parents=True, exist_ok=True)
        return d

    def save_session(self, meta: dict[str, Any]) -> None:
        d = self._sdir(meta["id"])
        _write_json(d / "session.json", meta)

    def task_path(self, session_id: str, task_id: str) -> Path:
        return self.root / session_id / "runs" / f"{task_id}.json"

    def rel_task_path(self, session_id: str, task_id: str) -> str:
        p = self.task_path(session_id, task_id)
        try:
            return str(p.relative_to(Path.cwd()))
        except ValueError:
            return str(p)

    def save_task(self, session_id: str, snapshot: dict[str, Any]) -> None:
        d = self._sdir(session_id)
        _write_json(self.task_path(session_id, snapshot["id"]), snapshot)

    def delete_task(self, session_id: str, task_id: str) -> None:
        path = self.task_path(session_id, task_id)
        if path.exists():
            path.unlink()

    def delete_session(self, session_id: str) -> None:
        import shutil

        d = self.root / session_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    def load_all(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Return (session metas, {task_id: snapshot}) from disk."""
        sessions: list[dict[str, Any]] = []
        tasks: dict[str, dict[str, Any]] = {}
        if not self.root.exists():
            return sessions, tasks
        for sdir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            meta_file = sdir / "session.json"
            if not meta_file.exists():
                continue
            try:
                meta = _read_json(meta_file)
            except Exception as exc:  # noqa: BLE001
                log.warning("store.session_load_failed", dir=str(sdir), error=str(exc))
                continue
            sessions.append(meta)
            for run_file in sorted((sdir / "runs").glob("*.json")):
                try:
                    snap = _read_json(run_file)
                    tasks[snap["id"]] = snap
                except Exception as exc:  # noqa: BLE001
                    log.warning("store.run_load_failed", file=str(run_file), error=str(exc))
        log.info("store.loaded", sessions=len(sessions), tasks=len(tasks))
        return sessions, tasks

    # ── Memory ───────────────────────────────────────────────────────────────
    def memory_path(self) -> Path:
        return self.root / "_memory.jsonl"

    def memory_backup_path(self) -> Path:
        return self.root / "_memory.jsonl.bak"

    def save_memory(self, records: list[dict[str, Any]]) -> None:
        self._write_memory_file(self.memory_path(), records)

    def load_memory(self) -> list[dict[str, Any]]:
        return self._load_memory_file(self.memory_path())

    def archive_memory(self, records: list[dict[str, Any]]) -> None:
        """Snapshot the current memory export before a wipe (for optional restore)."""
        if not records:
            return
        self._write_memory_file(self.memory_backup_path(), records)

    def load_memory_backup(self) -> list[dict[str, Any]]:
        return self._load_memory_file(self.memory_backup_path())

    def _write_memory_file(self, path: Path, records: list[dict[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, default=str) + "\n")
        tmp.replace(path)

    def _load_memory_file(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
