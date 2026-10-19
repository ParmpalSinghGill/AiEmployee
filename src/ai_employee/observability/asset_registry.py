"""A persistent index of models and datasets already on disk.

Prior-run context (``session_context``) only reaches back through *one saved
session*, so a brand-new task starts blind: it re-searches the hubs and
re-downloads a model the workspace already holds. This module records every
successful acquisition in a workspace-level ledger and renders it as context for
*any* task, so the employee checks local inventory before going to the internet.

The ledger is the index, never the source of truth — paths are re-verified
against the filesystem on every read, so a file deleted outside the app is
reported as missing rather than confidently handed to an agent.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("asset_registry")

_LEDGER_NAME = "assets.json"
_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AssetRecord:
    """One acquired resource: where it came from and where it landed."""

    path: str  # workspace-relative
    kind: str  # model | dataset | repo | media | file
    source: str  # huggingface | kaggle | github | youtube | http | local
    url: str = ""
    identifier: str = ""  # e.g. "bert-base-uncased", "coco/2017"
    session_id: str = ""
    bytes: int = 0
    acquired_at: str = field(default_factory=_now_iso)
    last_used_at: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AssetRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


# Where ``canonical_artifact_dest`` routes each bucket, mapped to what it holds.
_BUCKET_KIND = {"models": "model", "data": "dataset"}
_MODEL_FILE_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".h5", ".ckpt"}
_MEDIA_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".wav", ".mp3", ".flac"}
_DATA_EXTS = {".csv", ".tsv", ".parquet", ".jsonl", ".arrow", ".zip", ".tar", ".gz"}


def classify_asset(rel_path: str, source: str) -> str:
    """Best-effort kind for an acquired path, from its bucket then its extension."""
    parts = Path(rel_path).parts
    if parts and parts[0].lower() in _BUCKET_KIND:
        return _BUCKET_KIND[parts[0].lower()]
    if source == "github":
        return "repo"
    if source == "youtube":
        return "media"
    ext = Path(rel_path).suffix.lower()
    if ext in _MODEL_FILE_EXTS:
        return "model"
    if ext in _MEDIA_EXTS:
        return "media"
    if ext in _DATA_EXTS:
        return "dataset"
    return "file"


def derive_identifier(url: str, rel_path: str) -> str:
    """A stable human name — the hub repo id when there is one, else the filename."""
    u = (url or "").strip()
    for host, keep in (
        ("huggingface.co", 2),
        ("kaggle.com/datasets", 2),
        ("github.com", 2),
    ):
        if host in u:
            tail = u.split(host, 1)[1].strip("/")
            segs = [s for s in tail.split("/") if s and s not in {"datasets", "models"}]
            if segs:
                return "/".join(segs[:keep])
    name = Path(rel_path).name
    return name or rel_path


def _dir_size(path: Path, *, budget: int = 2000) -> int:
    """Total bytes under ``path``. Bounded so a huge corpus can't stall a write."""
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    seen = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            seen += 1
            if seen > budget:
                return total
            try:
                total += (Path(dirpath) / fn).stat().st_size
            except OSError:
                continue
    return total


class AssetRegistry:
    """Workspace-level ledger of acquired assets, shared across every session."""

    def __init__(self, workspace: Path) -> None:
        self._ws = Path(workspace).resolve()
        self._ledger = self._ws / _LEDGER_NAME

    @property
    def workspace(self) -> Path:
        """The canonical workspace directory assets are stored relative to."""
        return self._ws

    # ── persistence ──────────────────────────────────────────────────────────
    def _load_raw(self) -> list[dict[str, Any]]:
        if not self._ledger.exists():
            return []
        try:
            data = json.loads(self._ledger.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("asset_ledger_unreadable", error=str(exc))
            return []
        if isinstance(data, dict):
            return list(data.get("assets") or [])
        return list(data) if isinstance(data, list) else []

    def _write_raw(self, assets: list[dict[str, Any]]) -> None:
        payload = {"version": _SCHEMA_VERSION, "assets": assets}
        try:
            self._ws.mkdir(parents=True, exist_ok=True)
            # Atomic replace: a crash mid-write must not truncate the ledger.
            with tempfile.NamedTemporaryFile(
                "w", dir=self._ws, prefix=".assets-", suffix=".tmp", delete=False
            ) as fh:
                json.dump(payload, fh, indent=2)
                tmp = Path(fh.name)
            tmp.replace(self._ledger)
        except OSError as exc:
            log.warning("asset_ledger_unwritable", error=str(exc))

    def _relative(self, path: str | Path) -> str | None:
        """Workspace-relative form of ``path``, or None if it escapes the workspace.

        A relative input may be relative to the *workspace* ("models/x.pt") or to the
        process cwd ("data/workspace/models/x.pt") — download helpers produce both.
        Resolving against cwd first and only then joining the workspace avoids
        doubling the prefix ("workspace/workspace/models/x.pt"), which silently
        dropped assets from the ledger.
        """
        p = Path(path)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self._ws / p)  # workspace-relative (the common case)
            candidates.append(Path.cwd() / p)  # cwd-relative, e.g. data/workspace/...
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                rel = resolved.relative_to(self._ws)
            except (ValueError, OSError):
                continue
            if resolved.exists():
                return str(rel)
        # Nothing exists yet (caller may create it): fall back to the first form that
        # is at least inside the workspace, so the path check stays authoritative.
        for candidate in candidates:
            try:
                return str(candidate.resolve().relative_to(self._ws))
            except (ValueError, OSError):
                continue
        return None

    # ── writes ───────────────────────────────────────────────────────────────
    def record(
        self,
        *,
        path: str | Path,
        source: str,
        url: str = "",
        session_id: str = "",
        kind: str = "",
        note: str = "",
    ) -> AssetRecord | None:
        """Index a newly acquired asset. Returns None if the path isn't usable."""
        rel = self._relative(path)
        if not rel:
            log.warning("asset_outside_workspace", path=str(path))
            return None
        abs_path = self._ws / rel
        if not abs_path.exists():
            return None

        rec = AssetRecord(
            path=rel,
            kind=kind or classify_asset(rel, source),
            source=source,
            url=(url or "").strip(),
            identifier=derive_identifier(url, rel),
            session_id=session_id,
            bytes=_dir_size(abs_path),
            note=note,
        )
        assets = [a for a in self._load_raw() if a.get("path") != rel]
        assets.append(rec.to_dict())
        self._write_raw(assets)
        log.info("asset_recorded", path=rel, kind=rec.kind, source=source)
        return rec

    def touch(self, path: str | Path) -> None:
        """Mark an asset as reused, so inventory can be ordered by recency."""
        rel = self._relative(path)
        if not rel:
            return
        assets = self._load_raw()
        changed = False
        for a in assets:
            if a.get("path") == rel:
                a["last_used_at"] = _now_iso()
                changed = True
        if changed:
            self._write_raw(assets)

    def forget_missing(self) -> int:
        """Drop ledger entries whose files are gone. Returns how many were removed."""
        assets = self._load_raw()
        kept = [a for a in assets if (self._ws / str(a.get("path", ""))).exists()]
        removed = len(assets) - len(kept)
        if removed:
            self._write_raw(kept)
        return removed

    # ── reads ────────────────────────────────────────────────────────────────
    def list_assets(self, *, kind: str = "", verify: bool = True) -> list[AssetRecord]:
        """Every indexed asset, newest first, verified against the filesystem."""
        out: list[AssetRecord] = []
        for raw in self._load_raw():
            try:
                rec = AssetRecord.from_dict(raw)
            except TypeError:
                continue
            if kind and rec.kind != kind:
                continue
            if verify and not (self._ws / rec.path).exists():
                continue
            out.append(rec)
        out.sort(key=lambda r: r.last_used_at or r.acquired_at, reverse=True)
        return out

    def find_by_url(self, url: str) -> AssetRecord | None:
        """The asset previously fetched from ``url``, if its file is still present.

        Matching is on the normalized URL, so ``…/repo`` and ``…/repo/`` (or an
        http/https flip) still count as the same resource rather than re-fetching.
        """
        target = _normalize_url(url)
        if not target:
            return None
        for rec in self.list_assets():
            if rec.url and _normalize_url(rec.url) == target:
                return rec
        return None

    def find(self, query: str, *, limit: int = 5) -> list[AssetRecord]:
        """Assets whose identifier, path, or URL match the query's terms."""
        terms = [t for t in _tokenize(query) if len(t) > 2]
        if not terms:
            return []
        scored: list[tuple[int, AssetRecord]] = []
        for rec in self.list_assets():
            haystack = f"{rec.identifier} {rec.path} {rec.url} {rec.note}".lower()
            score = sum(1 for t in terms if t in haystack)
            if score:
                scored.append((score, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _score, rec in scored[:limit]]


def _normalize_url(url: str) -> str:
    """Lowercased, scheme- and trailing-slash-insensitive form for cache matching."""
    u = (url or "").strip().strip("`").lower()
    if not u:
        return ""
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix) :]
            break
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def _tokenize(text: str) -> list[str]:
    return [
        t for t in "".join(c if c.isalnum() else " " for c in (text or "").lower()).split() if t
    ]


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    if n < 1024**3:
        return f"{n / 1048576:.1f} MB"
    return f"{n / 1073741824:.1f} GB"


def build_local_inventory_context(
    registry: AssetRegistry,
    *,
    max_items: int = 40,
) -> str:
    """Render the local inventory as planner/agent context.

    Deliberately instructional, not just a list: without an explicit "prefer these
    over searching" rule the model treats the inventory as trivia and searches the
    hubs anyway.
    """
    assets = registry.list_assets()
    if not assets:
        return ""

    by_kind: dict[str, list[AssetRecord]] = {}
    for rec in assets[:max_items]:
        by_kind.setdefault(rec.kind, []).append(rec)

    lines: list[str] = []
    for kind in ("model", "dataset", "repo", "media", "file"):
        group = by_kind.get(kind)
        if not group:
            continue
        lines.append(f"{kind.upper()}S already on disk:")
        for rec in group:
            size = _fmt_bytes(rec.bytes)
            # Resolve each ledger path to its canonical absolute form so agents can
            # use it directly — the shell tool runs inside the nested workspace, so a
            # relative ``data/...`` path is ambiguous between the repo root and cwd.
            abs_path = (registry.workspace / rec.path).resolve()
            bits = [f"- {rec.identifier}", f"path: {abs_path}"]
            if size:
                bits.append(size)
            if rec.url:
                bits.append(f"from {rec.url}")
            lines.append("  ".join(bits))

    hidden = len(assets) - min(len(assets), max_items)
    if hidden > 0:
        lines.append(
            f"… and {hidden} more local asset(s) — use search_local_assets to look them up."
        )

    return (
        "LOCAL ASSET INVENTORY (already downloaded in this workspace, available to "
        "every task):\n"
        "Check this list BEFORE searching Hugging Face, Kaggle, arXiv, GitHub or the "
        "web for a model or dataset. If something here already fits the goal, use it "
        "at the path shown and do NOT download it again — downloads are slow, "
        "approval-gated, and waste disk. Search the internet only when nothing here "
        "fits, or when the user explicitly asks for something new, newer, or "
        "different. If you use one of these, say which path you used and why it fit.\n\n"
        + "\n".join(lines)
    )
