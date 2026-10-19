"""Isolation: turn every shared source into a task-private copy.

The resource manager does the filesystem work (a *real* copy, then re-own); this
module adds the one thing the manager deliberately avoids — **network I/O** — so
that a shared source whose bytes are missing can be **re-downloaded from the
internet** into the task's private area, fulfilling the "no shared resource after
isolating" guarantee even when the shared copy was pruned.

Everything here is best-effort and never raises on a single failure: one source
that can't be fetched stays in ``needs_redownload`` with its reason, and the rest
still isolate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger
from ai_employee.resources.models import Visibility
from ai_employee.resources.store import ResourceManager

log = get_logger("resources.isolation")

#: ``fetch(url, dest_abs) -> True`` on success.
Fetcher = Callable[[str, Path], Awaitable[bool]]


def default_fetcher(settings: Any) -> Fetcher:
    """Fetch a URL into a destination using the framework's download helpers."""

    async def fetch(url: str, dest: Path) -> bool:
        from ai_employee.tools.builtins.resource_download import (
            detect_source,
            download_http,
            download_huggingface,
        )

        dest.parent.mkdir(parents=True, exist_ok=True)
        kind = detect_source(url)
        if kind == "huggingface":
            await download_huggingface(
                url,
                dest,
                workspace=Path(settings.workspace_dir),
                token=(getattr(settings, "huggingface_token", "") or None),
            )
        else:
            await download_http(url, dest)
        return dest.exists()  # noqa: ASYNC240 - cheap path check, no blocking I/O

    return fetch


async def _restore(
    manager: ResourceManager,
    item: dict[str, Any],
    fetch: Fetcher,
) -> dict[str, Any] | None:
    """Re-download one missing shared source into the task's private area.

    Returns the isolation entry on success, or ``None`` (leaving the caller to
    keep the item in ``needs_redownload``).
    """
    dest = manager.workspace / item["dest"]
    dest.parent.mkdir(
        parents=True, exist_ok=True
    )  # noqa: ASYNC240 - cheap path math, no blocking I/O
    try:
        ok = await fetch(item["url"], dest)
    except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
        item["error"] = f"{type(exc).__name__}: {exc}"
        return None
    if not (ok and dest.exists()):
        item["error"] = item.get("error", "fetch produced no file")
        return None
    priv = manager.put(
        dest,
        name=item["name"],
        kind=item["kind"],
        visibility=Visibility.private.value,
        task=item["task_key"],
        task_id=item.get("task_id", ""),
        session_id=item.get("session_id", ""),
        source=item.get("source", "http"),
        url=item["url"],
        note=f"re-downloaded to isolate from shared {item['shared']}",
        dedupe=False,
    )
    manager.unlink(item["shared"], item["task_key"])
    if item.get("task_id"):
        manager.unlink(item["shared"], item["task_id"])
    return {"shared": item["shared"], "private": priv.id, "path": priv.path, "redownloaded": True}


async def isolate_task(
    manager: ResourceManager,
    task_ref: str,
    *,
    settings: Any | None = None,
    fetcher: Fetcher | None = None,
    allow_redownload: bool = True,
    purge_orphans: bool = True,
) -> dict[str, Any]:
    """Isolate one task's shared sources (copy, then re-download what's missing)."""
    result = manager.isolate_task(task_ref)
    fetch = fetcher or (default_fetcher(settings) if settings is not None else None)
    recovered: list[dict[str, Any]] = []
    still: list[dict[str, Any]] = []
    for item in list(result["needs_redownload"]):
        if not (allow_redownload and fetch is not None):
            still.append({**item, "reason": "redownload disabled"})
            continue
        entry = await _restore(manager, dict(item), fetch)
        if entry is None:
            still.append(item)
            continue
        recovered.append(entry)
        if not manager.links_for(item["shared"]):
            result["orphaned"].append(item["shared"])
    result["isolated"] += recovered
    result["needs_redownload"] = still
    result["orphaned"] = sorted(set(result["orphaned"]))
    if purge_orphans and result["orphaned"]:
        result["purged"] = manager.purge_records(result["orphaned"])
    log.info(
        "resources.isolate_task",
        task=result.get("task"),
        isolated=len(result["isolated"]),
        redownload=len(recovered),
        pending=len(still),
    )
    return result


async def isolate_all(
    manager: ResourceManager,
    *,
    settings: Any | None = None,
    fetcher: Fetcher | None = None,
    allow_redownload: bool = True,
    purge_orphans: bool = True,
) -> dict[str, Any]:
    """Isolate every task, re-downloading shared sources that must become private."""
    combined = manager.isolate_all(copy=True, purge_orphans=False)
    fetch = fetcher or (default_fetcher(settings) if settings is not None else None)
    recovered: list[dict[str, Any]] = []
    still: list[dict[str, Any]] = []
    orphans = set(combined.get("orphaned", []))
    for item in list(combined["needs_redownload"]):
        if not (allow_redownload and fetch is not None):
            still.append({**item, "reason": "redownload disabled"})
            continue
        entry = await _restore(manager, dict(item), fetch)
        if entry is None:
            still.append(item)
            continue
        recovered.append(entry)
        if not manager.links_for(item["shared"]):
            orphans.add(item["shared"])
    combined["isolated"] += recovered
    combined["needs_redownload"] = still
    combined["orphaned"] = sorted(orphans)
    if purge_orphans and combined["orphaned"]:
        combined["purged"] = manager.purge_records(combined["orphaned"])
    log.info(
        "resources.isolate_all",
        tasks=len(combined.get("tasks", [])),
        isolated=len(combined["isolated"]),
        redownload=len(recovered),
        pending=len(still),
        purged=len(combined.get("purged", [])),
    )
    return combined


__all__ = ["Fetcher", "default_fetcher", "isolate_task", "isolate_all"]
