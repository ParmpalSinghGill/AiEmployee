"""Cross-platform advisory file locking for catalog/resource mutations.

Concurrency protection has two layers:

1. A process-wide :class:`threading.RLock` per lock path, so two threads in one
   process serialize (and a thread may safely re-enter).
2. An OS advisory lock on a lock file (``flock`` on POSIX, ``msvcrt.locking`` on
   Windows), so a *second process* — a CLI run and the server, say — cannot
   corrupt the catalog mid write.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


_registry_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}


def _thread_lock(key: str) -> threading.RLock:
    with _registry_guard:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _thread_locks[key] = lock
        return lock


class FileLockTimeout(TimeoutError):
    """Raised when a lock could not be acquired within its timeout."""


class FileLock:
    """Exclusive, re-entrant-within-thread lock backed by a lock file."""

    def __init__(self, path: str | Path, *, timeout: float = 15.0, poll: float = 0.05) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self._fh: object | None = None
        self._tlock: threading.RLock | None = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tlock = _thread_lock(str(self.path.resolve()))
        if not self._tlock.acquire(timeout=self.timeout):
            raise FileLockTimeout(f"timed out acquiring {self.path}")
        try:
            self._acquire_os()
        except BaseException:
            self._tlock.release()
            self._tlock = None
            raise
        return self

    def _acquire_os(self) -> None:
        deadline = time.monotonic() + self.timeout
        self._fh = open(self.path, "a+b")  # noqa: SIM115 - kept open for the lock's lifetime
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    self._release_os()
                    raise FileLockTimeout(f"timed out acquiring {self.path}") from None
                time.sleep(self.poll)

    def _release_os(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        with contextlib.suppress(OSError):
            fh.close()

    def __exit__(self, *exc: object) -> None:
        self._release_os()
        if self._tlock is not None:
            with contextlib.suppress(RuntimeError):
                self._tlock.release()
            self._tlock = None


def lock_path_for(target: str | Path) -> Path:
    """A sibling lock file path for a target file (``x.json`` → ``.x.json.lock``)."""
    p = Path(target)
    return p.with_name(f".{p.name}.lock")


__all__ = ["FileLock", "FileLockTimeout", "lock_path_for"]
