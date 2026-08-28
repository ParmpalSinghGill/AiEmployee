"""Optional user accounts for the console.

The console works with NO account by default: a visitor without a session
cookie is minted an ephemeral **guest** identity ("Test1", "Test2", …) that is
never written to disk — it lives only in a signed browser cookie, so a server
restart (or clearing cookies) forgets the guest entirely. Guests can run tasks,
but their sessions are kept in memory only and are not persisted.

Real accounts are created by signing in with Google (via the operator's
registered OAuth client — see :mod:`ai_employee.integrations.google_oauth`).
Signed-in users are stored in ``settings.users_file`` and their sessions
persist and survive restarts, scoped to their account.

The cookie is a compact HMAC-signed JSON payload (no JWT dependency); the
signing secret is generated on first use and kept at ``settings.auth_secret_file``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import itertools
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request, Response

from ai_employee.core.logging import get_logger

log = get_logger("auth")

COOKIE_NAME = "ai_employee_user"

# Monotonic display names for guests. Process-local by design: guests are not
# remembered anywhere, so after a restart numbering starts again at Test1.
# Scoping uses the random guest id, not the display name, so a stale "Test1"
# cookie from before a restart never sees a new Test1's tasks.
_guest_counter = itertools.count(1)


@dataclass(frozen=True)
class CurrentUser:
    """The identity attached to a request — a guest or a stored account."""

    id: str  # "guest:<hex>" or "usr_<hex>"
    name: str  # display name ("Test1", or the account's name/email)
    kind: str  # "guest" | "account"
    email: str = ""

    @property
    def is_guest(self) -> bool:
        return self.kind == "guest"


# ── Cookie signing ───────────────────────────────────────────────────────────
def _secret(settings: Any) -> bytes:
    path = Path(settings.auth_secret_file)
    if path.exists():
        data = path.read_bytes().strip()
        if data:
            return data
    key = secrets.token_hex(32).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    with contextlib.suppress(OSError):  # the secret signs identities — owner-only
        path.chmod(0o600)
    log.info("auth.secret_created", path=str(path))
    return key


def _sign(secret: bytes, payload: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def encode_cookie(settings: Any, user: CurrentUser) -> str:
    payload = json.dumps(
        {
            "id": user.id,
            "name": user.name,
            "kind": user.kind,
            "email": user.email,
            "exp": time.time() + settings.auth_session_days * 86400,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    body = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    return f"{body}.{_sign(_secret(settings), payload)}"


def decode_cookie(settings: Any, value: str) -> CurrentUser | None:
    """Verify + decode a session cookie; None for anything invalid or expired."""
    try:
        body, sig = value.split(".", 1)
        payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        if not hmac.compare_digest(sig, _sign(_secret(settings), payload)):
            return None
        data = json.loads(payload)
        if float(data.get("exp") or 0) < time.time():
            return None
        kind = str(data.get("kind") or "")
        if kind not in {"guest", "account"}:
            return None
        # Accounts must still exist in the store (deleting a user logs them out).
        if kind == "account" and str(data.get("id")) not in load_users(settings):
            return None
        return CurrentUser(
            id=str(data["id"]),
            name=str(data.get("name") or "user"),
            kind=kind,
            email=str(data.get("email") or ""),
        )
    except Exception:  # noqa: BLE001 - any malformed cookie is just "not signed in"
        return None


# ── User store (signed-in accounts only — guests are never stored) ───────────
def load_users(settings: Any) -> dict[str, dict[str, Any]]:
    p = Path(settings.users_file)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        users = data.get("users") if isinstance(data, dict) else None
        return users if isinstance(users, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("auth.users_load_failed", error=str(exc))
        return {}


def _save_users(settings: Any, users: dict[str, dict[str, Any]]) -> None:
    p = Path(settings.users_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"users": users}, indent=2))


def upsert_google_user(settings: Any, profile: dict[str, str]) -> CurrentUser:
    """Create (or refresh) the stored account for a Google sign-in profile."""
    email = profile["email"].strip().lower()
    users = load_users(settings)
    uid = next((k for k, u in users.items() if u.get("email") == email), None)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if uid is None:
        uid = "usr_" + secrets.token_hex(6)
        users[uid] = {"email": email, "created_at": now}
        log.info("auth.user_created", email=email)
    users[uid].update(
        {
            "name": profile.get("name") or email,
            "picture": profile.get("picture") or "",
            "last_login_at": now,
        }
    )
    _save_users(settings, users)
    return CurrentUser(id=uid, name=users[uid]["name"], kind="account", email=email)


# ── Request identity ─────────────────────────────────────────────────────────
def mint_guest() -> CurrentUser:
    return CurrentUser(
        id="guest:" + secrets.token_hex(8),
        name=f"Test{next(_guest_counter)}",
        kind="guest",
    )


def set_user_cookie(settings: Any, response: Response, user: CurrentUser) -> None:
    response.set_cookie(
        COOKIE_NAME,
        encode_cookie(settings, user),
        max_age=settings.auth_session_days * 86400,
        httponly=True,
        samesite="lax",
    )


def clear_user_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME)


def get_current_user(request: Request, response: Response) -> CurrentUser:
    """FastAPI dependency: the request's identity, minting a guest when absent.

    The guest cookie is set on the way out, so the same browser keeps the same
    Test identity (and sees its own running tasks) across page reloads.
    """
    from ai_employee.config import get_settings

    s = get_settings()
    raw = request.cookies.get(COOKIE_NAME)
    if raw:
        user = decode_cookie(s, raw)
        if user is not None:
            return user
    guest = mint_guest()
    set_user_cookie(s, response, guest)
    log.info("auth.guest_minted", name=guest.name)
    return guest


__all__ = [
    "COOKIE_NAME",
    "CurrentUser",
    "clear_user_cookie",
    "decode_cookie",
    "encode_cookie",
    "get_current_user",
    "load_users",
    "mint_guest",
    "set_user_cookie",
    "upsert_google_user",
]
