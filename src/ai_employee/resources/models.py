"""Data structures for the resource catalog.

Deliberately keeps the five concepts separate:

``task identity``       — :class:`Provenance.task_id` / ``owner_task`` (date-based key)
``resource identity``   — :attr:`ResourceRecord.hash` (content hash) + stable ``id``
``resource ownership``  — :attr:`ResourceRecord.owner_task` / ``owner_user``
``resource visibility`` — :class:`Visibility` (private | shared)
``physical storage``    — :attr:`ResourceRecord.storage` (object store vs logical path)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Visibility(str, Enum):
    """Who can see a resource.

    ``private`` — only the owning task (``owner_task``) may read it.
    ``shared``  — any task in the workspace may read it (reusable assets).
    """

    private = "private"
    shared = "shared"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Provenance:
    """Where a resource came from and which actor produced it."""

    source: str = "local"  # huggingface | kaggle | github | youtube | http | local | migration
    url: str = ""
    tool: str = ""  # e.g. download_resource
    agent: str = ""  # e.g. scout
    task_id: str = ""  # date-based task key, not the free-text goal
    session_id: str = ""
    user_id: str | None = None
    ts: str = field(default_factory=now_iso)
    note: str = ""
    dedup_of: str = ""  # id of the resource whose bytes this one reuses, if any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Provenance:
        raw = raw or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ResourceLink:
    """A bidirectional edge between one task and one resource.

    A record keeps a link per task that owns or uses it. This is what lets the
    catalog answer **both** directions — "which tasks reference this source?" and
    "which sources does this task reference?" — so it knows when the *same* source
    is shared by *multiple* tasks (and must survive one of them being deleted).

    * ``task_id``  — the concrete run id (``TaskRecord.id``) when known.
    * ``task_key`` — the stable ownership identity (safe session slug).
    * ``session_id`` — groups the runs of one session (initial + follow-ups).
    * ``role``     — ``owner`` (created it) or ``consumer`` (reuses it).
    """

    task_id: str = ""
    task_key: str = ""
    session_id: str = ""
    user_id: str | None = None
    role: str = "owner"  # owner | consumer
    ts: str = field(default_factory=now_iso)

    @property
    def identity(self) -> str:
        return self.task_id or self.task_key or self.session_id

    def matches(self, ref: str) -> bool:
        """True if ``ref`` is any of this link's task id / key / session id."""
        ref = (ref or "").strip()
        return bool(ref) and ref in {self.task_id, self.task_key, self.session_id}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ResourceLink:
        raw = raw or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ResourceVersion:
    """One revision of a resource's content."""

    version: int
    hash: str
    hash_kind: str  # content | manifest
    size: int
    logical_path: str  # workspace-relative materialisation for this version
    created_at: str = field(default_factory=now_iso)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResourceVersion:
        data = dict(raw)
        data["provenance"] = Provenance.from_dict(data.get("provenance"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ResourceRecord:
    """The catalog entry for one logical resource."""

    id: str
    name: str
    kind: str  # model | dataset | repo | media | file
    visibility: str = Visibility.private.value
    owner_task: str | None = None
    owner_user: str | None = None
    hash: str = ""
    hash_kind: str = "content"
    size: int = 0
    #: ``{"mode": "path", "path": "<rel>"}`` or
    #: ``{"mode": "object", "object": "<aa>/<hash>", "path": "<rel>"}``.
    storage: dict[str, Any] = field(default_factory=dict)
    source: str = "local"
    url: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_used_at: str = ""
    versions: list[ResourceVersion] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    #: Bidirectional task edges — one per task that owns/uses this resource.
    links: list[ResourceLink] = field(default_factory=list)

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def path(self) -> str:
        """Workspace-relative logical path of the current version."""
        return str(self.storage.get("path") or "")

    @property
    def version(self) -> int:
        return self.versions[-1].version if self.versions else 0

    @property
    def is_shared(self) -> bool:
        return self.visibility == Visibility.shared.value

    def link_identities(self) -> set[str]:
        """Every task id/key/session id linked to this resource."""
        out: set[str] = set()
        for link in self.links:
            for value in (link.task_id, link.task_key, link.session_id):
                if value:
                    out.add(value)
        return out

    # ── serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        d["versions"] = [v.to_dict() for v in self.versions]
        d["links"] = [link.to_dict() for link in self.links]
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResourceRecord:
        data = dict(raw)
        data["versions"] = [ResourceVersion.from_dict(v) for v in data.get("versions") or []]
        data["provenance"] = Provenance.from_dict(data.get("provenance"))
        data["links"] = [ResourceLink.from_dict(x) for x in data.get("links") or []]
        if not isinstance(data.get("storage"), dict):
            data["storage"] = {}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


__all__ = [
    "Visibility",
    "Provenance",
    "ResourceLink",
    "ResourceVersion",
    "ResourceRecord",
    "now_iso",
]
