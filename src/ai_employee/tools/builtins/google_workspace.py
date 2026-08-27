"""Gmail & Google Calendar tools — act on any user-connected Google account.

These call Google's REST APIs directly with tokens from
:mod:`ai_employee.integrations.google_oauth` (the per-user consent flow).
Every tool takes an optional ``account`` (email); with a single connected
account it is picked automatically. Sending mail reuses the ``email_send``
approval gate; creating events is gated by ``calendar_write``.

The toolset is only registered when the operator's OAuth client file exists
(see ``build_google_tools``), so default/offline installs see no dead tools.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any

from pydantic import BaseModel, Field

from ai_employee.config import Settings
from ai_employee.tools.base import Tool, ToolResult

_GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
_CALENDAR = "https://www.googleapis.com/calendar/v3"


async def _api(
    settings: Settings,
    account: str,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    import httpx

    from ai_employee.integrations.google_oauth import ensure_access_token, resolve_account

    email = resolve_account(settings, account)
    token = await ensure_access_token(settings, email)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Google API {resp.status_code}: {resp.text[:400]}")
    return resp.json() if resp.content else {}


def _header(payload: dict[str, Any], name: str) -> str:
    for h in payload.get("headers") or []:
        if h.get("name", "").lower() == name.lower():
            return str(h.get("value", ""))
    return ""


def _body_text(payload: dict[str, Any]) -> str:
    """Depth-first search for the first text/plain part (fallback text/html)."""

    def decode(part: dict[str, Any]) -> str:
        data = (part.get("body") or {}).get("data", "")
        if not data:
            return ""
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
            "utf-8", errors="replace"
        )

    stack, html = [payload], ""
    while stack:
        part = stack.pop(0)
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            text = decode(part)
            if text:
                return text
        elif mime == "text/html" and not html:
            html = decode(part)
        stack.extend(part.get("parts") or [])
    return html


class GmailSearchTool(Tool):
    name = "gmail_search"
    description = (
        "Search a connected Gmail mailbox. 'query' uses Gmail search syntax "
        "(e.g. 'from:alice@x.com is:unread newer_than:7d'). Returns id, from, "
        "subject, date, and snippet per message."
    )
    category = "google"

    class Args(BaseModel):
        query: str = Field(description="Gmail search query")
        max_results: int = Field(default=10, ge=1, le=50)
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        listing = await _api(
            self._settings,
            args.account,
            "GET",
            f"{_GMAIL}/messages",
            params={"q": args.query, "maxResults": args.max_results},
        )
        ids = [m["id"] for m in listing.get("messages") or []]
        lines = []
        for mid in ids:
            msg = await _api(
                self._settings,
                args.account,
                "GET",
                f"{_GMAIL}/messages/{mid}",
                params={
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date"],
                },
            )
            payload = msg.get("payload") or {}
            lines.append(
                f"[{mid}] {_header(payload, 'Date')} | {_header(payload, 'From')} | "
                f"{_header(payload, 'Subject')}\n    {msg.get('snippet', '')}"
            )
        output = "\n".join(lines) or "No messages matched."
        return ToolResult(ok=True, output=output, metadata={"count": len(ids)})


class GmailReadTool(Tool):
    name = "gmail_read"
    description = "Read one Gmail message in full (headers + body) by its id from gmail_search."
    category = "google"

    class Args(BaseModel):
        message_id: str
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        msg = await _api(
            self._settings,
            args.account,
            "GET",
            f"{_GMAIL}/messages/{args.message_id}",
            params={"format": "full"},
        )
        payload = msg.get("payload") or {}
        body = _body_text(payload) or msg.get("snippet", "")
        output = (
            f"From: {_header(payload, 'From')}\nTo: {_header(payload, 'To')}\n"
            f"Date: {_header(payload, 'Date')}\nSubject: {_header(payload, 'Subject')}\n\n"
            f"{body[:8000]}"
        )
        return ToolResult(ok=True, output=output, metadata={"message_id": args.message_id})


class GmailSendTool(Tool):
    name = "gmail_send"
    description = "Send an email from a connected Gmail account."
    category = "google"
    approval_key = "email_send"  # same human gate as the SMTP tool

    class Args(BaseModel):
        to: str = Field(description="Recipient address(es), comma-separated")
        subject: str
        body: str = Field(description="Plain-text body")
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        mime = EmailMessage()
        mime["To"] = args.to
        mime["Subject"] = args.subject
        mime.set_content(args.body)
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
        sent = await _api(
            self._settings,
            args.account,
            "POST",
            f"{_GMAIL}/messages/send",
            json_body={"raw": raw},
        )
        return ToolResult(
            ok=True,
            output=f"Email sent to {args.to} (id {sent.get('id', '?')}).",
            metadata={"message_id": sent.get("id")},
        )


class CalendarListEventsTool(Tool):
    name = "calendar_list_events"
    description = (
        "List upcoming events on a connected Google Calendar. Times are RFC3339 "
        "(e.g. 2026-08-27T00:00:00Z); omitted bounds default to 'from now'."
    )
    category = "google"

    class Args(BaseModel):
        time_min: str = Field(default="", description="RFC3339 lower bound (default: now)")
        time_max: str = Field(default="", description="RFC3339 upper bound (optional)")
        max_results: int = Field(default=20, ge=1, le=100)
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        from datetime import UTC, datetime

        params: dict[str, Any] = {
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": args.max_results,
            "timeMin": args.time_min or datetime.now(UTC).isoformat(),
        }
        if args.time_max:
            params["timeMax"] = args.time_max
        data = await _api(
            self._settings, args.account, "GET", f"{_CALENDAR}/calendars/primary/events",
            params=params,
        )
        lines = []
        for ev in data.get("items") or []:
            start = (ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date")
            lines.append(f"[{ev.get('id')}] {start} — {ev.get('summary', '(no title)')}")
        return ToolResult(
            ok=True,
            output="\n".join(lines) or "No events in this range.",
            metadata={"count": len(lines)},
        )


class CalendarCreateEventTool(Tool):
    name = "calendar_create_event"
    description = "Create an event on a connected Google Calendar."
    category = "google"
    approval_key = "calendar_write"

    class Args(BaseModel):
        summary: str
        start: str = Field(description="RFC3339 start, e.g. 2026-08-28T15:00:00+05:30")
        end: str = Field(description="RFC3339 end")
        description: str = ""
        attendees: list[str] = Field(default_factory=list, description="Attendee emails")
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        body: dict[str, Any] = {
            "summary": args.summary,
            "description": args.description,
            "start": {"dateTime": args.start},
            "end": {"dateTime": args.end},
        }
        if args.attendees:
            body["attendees"] = [{"email": a} for a in args.attendees]
        ev = await _api(
            self._settings, args.account, "POST", f"{_CALENDAR}/calendars/primary/events",
            json_body=body,
        )
        return ToolResult(
            ok=True,
            output=f"Event created: {ev.get('htmlLink') or ev.get('id')}",
            metadata={"event_id": ev.get("id")},
        )


def build_google_tools(settings: Settings) -> list[Tool]:
    """The Google toolset — empty unless the operator's OAuth client is set up."""
    from ai_employee.integrations.google_oauth import client_config

    if client_config(settings) is None:
        return []
    return [
        GmailSearchTool(settings),
        GmailReadTool(settings),
        GmailSendTool(settings),
        CalendarListEventsTool(settings),
        CalendarCreateEventTool(settings),
    ]


__all__ = [
    "CalendarCreateEventTool",
    "CalendarListEventsTool",
    "GmailReadTool",
    "GmailSearchTool",
    "GmailSendTool",
    "build_google_tools",
]
