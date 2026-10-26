"""Gmail, Google Calendar & Drive tools — act on any user-connected Google account.

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
from ai_employee.core.exceptions import ToolError
from ai_employee.tools.base import Tool, ToolResult

_GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
_CALENDAR = "https://www.googleapis.com/calendar/v3"
_DRIVE = "https://www.googleapis.com/drive/v3"


async def _api(
    settings: Settings,
    account: str,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    raw: bool = False,
) -> Any:
    import httpx

    from ai_employee.integrations.google_oauth import ensure_access_token, resolve_account

    email = resolve_account(settings, account)
    token = await ensure_access_token(settings, email)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code >= 400:
        text = resp.text
        message = _google_error_message(text)
        lowered = text.lower()
        if resp.status_code == 403 and "insufficient" in lowered:
            raise ToolError(
                f"the Google account '{email}' has not granted access to this service — "
                "open 🔌 MCP and click Connect on the Gmail / Google Calendar / Google "
                "Drive card to grant it"
            )
        if resp.status_code == 403 and (
            "service_disabled" in lowered
            or "accessnotconfigured" in lowered
            or "has not been used" in lowered
        ):
            raise ToolError(
                "this Google API is not enabled in your Google Cloud project — open the "
                "link in Google's message (or APIs & Services → Library in Google Cloud "
                f"console), click Enable, wait a minute, and retry. Google said: {message}"
            )
        raise RuntimeError(f"Google API {resp.status_code}: {message}")
    if raw:
        return resp
    return resp.json() if resp.content else {}


def _google_error_message(text: str) -> str:
    """Google's human-readable error (``error.message``), else the raw body."""
    import json

    try:
        message = (json.loads(text).get("error") or {}).get("message")
    except (ValueError, AttributeError):
        message = None
    return str(message or text)[:500]


#: Cheapest read call per Google-backed MCP card — proves the API is enabled
#: in the operator's Cloud project and the account's grant works.
SERVICE_PROBES: dict[str, str] = {
    "gmail": f"{_GMAIL}/profile",
    "google_calendar": f"{_CALENDAR}/users/me/calendarList?maxResults=1",
    "gdrive": f"{_DRIVE}/about?fields=user",
}


async def probe_service(settings: Settings, service_id: str, account: str = "") -> str | None:
    """None when the service answers for ``account``, else why it doesn't."""
    url = SERVICE_PROBES.get(service_id)
    if url is None:
        return None
    try:
        await _api(settings, account, "GET", url)
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        return str(exc)
    return None


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
            self._settings,
            args.account,
            "GET",
            f"{_CALENDAR}/calendars/primary/events",
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
            self._settings,
            args.account,
            "POST",
            f"{_CALENDAR}/calendars/primary/events",
            json_body=body,
        )
        return ToolResult(
            ok=True,
            output=f"Event created: {ev.get('htmlLink') or ev.get('id')}",
            metadata={"event_id": ev.get("id")},
        )


# ── Google Drive (read-only) ─────────────────────────────────────────────────
#: Google-native files have no bytes of their own — they are exported instead.
_DRIVE_EXPORTS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.drawing": "image/svg+xml",
}
_DRIVE_TEXT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
    "application/csv",
)
_DRIVE_SEARCH_SYNTAX = (" contains ", "mimeType", "=", " in parents", "modifiedTime", " and ")


def _drive_query(query: str) -> str:
    """Plain words → a full-text search; Drive query syntax passes through."""
    q = query.strip()
    if not q:
        return "trashed = false"
    if any(token in q for token in _DRIVE_SEARCH_SYNTAX):
        return q if "trashed" in q else f"({q}) and trashed = false"
    escaped = q.replace("\\", "\\\\").replace("'", "\\'")
    return f"fullText contains '{escaped}' and trashed = false"


class DriveSearchTool(Tool):
    name = "drive_search"
    description = (
        "Search the connected Google Drive. 'query' is plain words (full-text "
        "search) or Drive query syntax (e.g. \"name contains 'budget' and "
        "mimeType = 'application/vnd.google-apps.spreadsheet'\"). Returns id, "
        "name, type, modified time and link per file — read one with drive_read."
    )
    category = "google"

    class Args(BaseModel):
        query: str = Field(default="", description="Words to find, or Drive query syntax")
        max_results: int = Field(default=10, ge=1, le=50)
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        params: dict[str, Any] = {
            "q": _drive_query(args.query),
            "pageSize": args.max_results,
            "fields": "files(id,name,mimeType,modifiedTime,size,webViewLink)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if not args.query.strip():  # Drive rejects orderBy on full-text searches
            params["orderBy"] = "modifiedTime desc"
        listing = await _api(self._settings, args.account, "GET", f"{_DRIVE}/files", params=params)
        files = listing.get("files") or []
        lines = [
            f"[{f.get('id')}] {f.get('name')} | {f.get('mimeType')} | "
            f"modified {f.get('modifiedTime', '?')}\n    {f.get('webViewLink', '')}"
            for f in files
        ]
        output = "\n".join(lines) or "No files matched."
        return ToolResult(ok=True, output=output, metadata={"count": len(files)})


class DriveReadTool(Tool):
    name = "drive_read"
    description = (
        "Read one Google Drive file's text by its id from drive_search. Docs and "
        "Slides come back as plain text, Sheets as CSV (first sheet), and text "
        "files as-is. Binary files (PDF, images, …) are described, not read."
    )
    category = "google"

    class Args(BaseModel):
        file_id: str
        max_chars: int = Field(default=20000, ge=500, le=100000)
        account: str = Field(default="", description="Connected account email (optional)")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: Args) -> ToolResult:  # type: ignore[override]
        meta = await _api(
            self._settings,
            args.account,
            "GET",
            f"{_DRIVE}/files/{args.file_id}",
            params={"fields": "id,name,mimeType,size,webViewLink", "supportsAllDrives": "true"},
        )
        mime = str(meta.get("mimeType") or "")
        name = meta.get("name") or args.file_id
        header = f"{name} ({mime})\n{meta.get('webViewLink', '')}\n\n"
        export = _DRIVE_EXPORTS.get(mime)
        if export:
            resp = await _api(
                self._settings,
                args.account,
                "GET",
                f"{_DRIVE}/files/{args.file_id}/export",
                params={"mimeType": export},
                raw=True,
            )
        elif mime.startswith(_DRIVE_TEXT_TYPES):
            if int(meta.get("size") or 0) > 5_000_000:
                return ToolResult(
                    ok=False,
                    output=header + "File is larger than 5 MB — too big to read here.",
                    error="file too large",
                )
            resp = await _api(
                self._settings,
                args.account,
                "GET",
                f"{_DRIVE}/files/{args.file_id}",
                params={"alt": "media", "supportsAllDrives": "true"},
                raw=True,
            )
        else:
            return ToolResult(
                ok=False,
                output=(
                    header + f"This is a binary file ({mime or 'unknown type'}); its text "
                    "can't be read through Drive. Open the link instead."
                ),
                error="binary file",
                metadata={"file_id": args.file_id, "mime_type": mime},
            )
        text = resp.text
        truncated = len(text) > args.max_chars
        body = text[: args.max_chars] + ("\n…(truncated)" if truncated else "")
        return ToolResult(
            ok=True,
            output=header + body,
            metadata={"file_id": args.file_id, "mime_type": mime, "truncated": truncated},
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
        DriveSearchTool(settings),
        DriveReadTool(settings),
    ]


__all__ = [
    "CalendarCreateEventTool",
    "CalendarListEventsTool",
    "DriveReadTool",
    "DriveSearchTool",
    "GmailReadTool",
    "GmailSearchTool",
    "GmailSendTool",
    "build_google_tools",
]
