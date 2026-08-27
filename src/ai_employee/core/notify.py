"""Push notifications to the operator's phone via ntfy.

Set ``NTFY_TOPIC`` in ``.env`` (pick a long, unguessable topic name), install
the ntfy app (Android/iOS) or open https://ntfy.sh/<topic> in a browser, and
subscribe to the same topic — done, no account needed. The task runner then
pushes when a run pauses for approval, finishes, or fails, so the employee can
work with the console closed and only ping a human for a signature.

Fire-and-forget: failures are logged and swallowed — a notification outage must
never affect a run. No network is touched when no topic is configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("notify")


def enabled(settings: Any) -> bool:
    return bool(str(getattr(settings, "ntfy_topic", "") or "").strip())


async def push(
    settings: Any,
    title: str,
    message: str,
    *,
    priority: str = "default",  # min|low|default|high|urgent
    tags: str = "",
    click_url: str = "",
) -> bool:
    """Send one push. Returns True when accepted by the ntfy server."""
    if not enabled(settings):
        return False
    import httpx

    url = f"{str(settings.ntfy_server).rstrip('/')}/{settings.ntfy_topic.strip()}"
    headers = {"Title": title.encode("ascii", "ignore").decode(), "Priority": priority}
    if tags:
        headers["Tags"] = tags
    if click_url:
        headers["Click"] = click_url
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message.encode(), headers=headers)
        ok = resp.status_code < 300
        if not ok:
            log.warning("notify.rejected", status=resp.status_code)
        return ok
    except Exception as exc:  # noqa: BLE001 - never let a push break a run
        log.warning("notify.failed", error=str(exc))
        return False


def push_soon(settings: Any, title: str, message: str, **kwargs: Any) -> None:
    """Schedule a push on the running loop without awaiting it."""
    if not enabled(settings):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(push(settings, title, message, **kwargs))
    task.add_done_callback(lambda t: t.exception())  # retrieve, never raise


__all__ = ["enabled", "push", "push_soon"]
