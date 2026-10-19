"""Content identity for resources.

A resource's identity is the **hash of its bytes**, not its name or URL — that is
what makes deduplication safe: two tasks that download the same model produce the
same hash and share one copy.

* A **file** is hashed by streaming its bytes (``kind="content"``).
* A **directory** is hashed from a canonical manifest of
  ``(relative-path, size, file-hash)`` entries (``kind="content"``). Hashing every
  byte of a huge dataset is slow, so beyond a budget the manifest falls back to
  ``(relative-path, size)`` only and reports ``kind="manifest"`` — still a stable
  identity for dedup, and the weaker guarantee is recorded, never hidden.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024  # 1 MiB
_ALGO = "sha256"


def _hash_file_bytes(path: Path, *, limit: int | None = None) -> tuple[str, int]:
    h = hashlib.new(_ALGO)
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            if limit is not None and size >= limit:
                break
    return h.hexdigest(), size


def hash_file(path: str | Path) -> tuple[str, int]:
    """``(sha256-hex, size-bytes)`` of a single file."""
    p = Path(path)
    return _hash_file_bytes(p)


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            files.append(Path(dirpath) / fn)
    return files


def hash_path(
    path: str | Path,
    *,
    max_hashed_bytes: int = 512 * 1024 * 1024,
    max_hashed_files: int = 20_000,
) -> dict[str, Any]:
    """Content identity for a file or directory.

    Returns ``{"hash": "sha256:<hex>", "kind": "content"|"manifest", "size": int,
    "files": int}``.
    """
    p = Path(path)
    if p.is_file():
        digest, size = hash_file(p)
        return {
            "hash": f"{_ALGO}:{digest}",
            "kind": "content",
            "size": size,
            "files": 1,
        }

    if not p.is_dir():
        raise FileNotFoundError(f"cannot hash missing path: {p}")

    files = _iter_files(p)
    h = hashlib.new(_ALGO)
    total = 0
    content_hashed = True
    bytes_hashed = 0
    hashed_files = 0
    for f in files:
        rel = f.relative_to(p).as_posix()
        try:
            st = f.stat()
        except OSError:
            continue
        total += st.st_size
        if content_hashed and (hashed_files < max_hashed_files and bytes_hashed < max_hashed_bytes):
            try:
                fh_digest, _ = _hash_file_bytes(f, limit=max(0, max_hashed_bytes - bytes_hashed))
            except OSError:
                fh_digest = "unreadable"
            bytes_hashed += st.st_size
            hashed_files += 1
            h.update(f"{rel}\0{st.st_size}\0{fh_digest}\n".encode())
        else:
            content_hashed = False
            h.update(f"{rel}\0{st.st_size}\n".encode())
    return {
        "hash": f"{_ALGO}:{h.hexdigest()}",
        "kind": "content" if content_hashed else "manifest",
        "size": total,
        "files": len(files),
    }


def normalize_hash(value: str) -> str:
    """Canonical ``algo:hex`` form for a hash the model or a user supplied."""
    v = (value or "").strip().lower()
    if not v:
        return ""
    if ":" not in v:
        return f"{_ALGO}:{v}"
    algo, _, digest = v.partition(":")
    return f"{algo or _ALGO}:{digest}"


__all__ = ["hash_file", "hash_path", "normalize_hash"]
