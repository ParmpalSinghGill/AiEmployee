"""User accounts: guest-by-default identities, Google sign-in, session scoping."""

from __future__ import annotations

import json
from pathlib import Path

from ai_employee.api import users as users_mod
from ai_employee.api.users import (
    CurrentUser,
    decode_cookie,
    encode_cookie,
    load_users,
    mint_guest,
    upsert_google_user,
)


# ── Cookie signing ───────────────────────────────────────────────────────────
def test_cookie_roundtrip(settings):
    user = CurrentUser(id="usr_abc", name="Jo", kind="account", email="jo@x.com")
    value = encode_cookie(settings, user)
    decoded = decode_cookie(settings, value)
    # Accounts must exist in the store to stay valid — this one doesn't.
    assert decoded is None
    guest = mint_guest()
    assert decode_cookie(settings, encode_cookie(settings, guest)) == guest


def test_cookie_tamper_rejected(settings):
    value = encode_cookie(settings, mint_guest())
    body, sig = value.split(".", 1)
    assert decode_cookie(settings, f"{body}.{'0' * len(sig)}") is None
    assert decode_cookie(settings, "garbage") is None
    assert decode_cookie(settings, f"{body}x.{sig}") is None


def test_guests_are_test_named_and_unique():
    a, b = mint_guest(), mint_guest()
    assert a.name.startswith("Test") and b.name.startswith("Test")
    assert a.id != b.id
    assert a.is_guest


# ── User store (accounts only — guests never touch disk) ─────────────────────
def test_upsert_google_user_creates_and_updates(settings):
    u1 = upsert_google_user(settings, {"email": "A@X.com", "name": "Ann", "picture": ""})
    assert u1.kind == "account" and u1.email == "a@x.com"
    stored = load_users(settings)
    assert stored[u1.id]["email"] == "a@x.com"
    # Same email again → same account id, refreshed profile.
    u2 = upsert_google_user(settings, {"email": "a@x.com", "name": "Ann B", "picture": ""})
    assert u2.id == u1.id
    assert load_users(settings)[u1.id]["name"] == "Ann B"
    # And the cookie for a stored account round-trips now.
    assert decode_cookie(settings, encode_cookie(settings, u2)) == u2


def test_guest_never_written_to_users_file(settings):
    mint_guest()
    assert load_users(settings) == {}
    assert not Path(settings.users_file).exists()


# ── API: /me mints a guest, cookie persists it ───────────────────────────────
async def test_me_mints_guest_and_is_stable(client):
    me = (await client.get("/ui/api/me")).json()
    assert me["kind"] == "guest"
    assert me["name"].startswith("Test")
    again = (await client.get("/ui/api/me")).json()
    assert again["id"] == me["id"] and again["name"] == me["name"]


async def test_logout_resets_to_new_guest(client):
    first = (await client.get("/ui/api/me")).json()
    r = await client.post("/ui/api/auth/logout")
    assert r.status_code == 200
    fresh = (await client.get("/ui/api/me")).json()
    assert fresh["kind"] == "guest"
    assert fresh["id"] != first["id"]


async def test_google_signin_unconfigured_is_502(client):
    r = await client.post("/ui/api/auth/google")
    assert r.status_code == 502
    assert "client_secret.json" in r.json()["detail"]


# ── Session scoping ──────────────────────────────────────────────────────────
async def _wait_done(client, task_id: str) -> None:
    import asyncio

    for _ in range(200):
        snap = (await client.get(f"/ui/api/tasks/{task_id}")).json()
        if snap["status"] in {"completed", "failed", "awaiting_approval", "waiting_for_user"}:
            return
        await asyncio.sleep(0.05)


async def test_guest_sessions_scoped_and_not_persisted(client, settings):
    from httpx import ASGITransport, AsyncClient

    r = (await client.post("/ui/api/tasks", json={"goal": "guest scoping probe"})).json()
    await _wait_done(client, r["task_id"])

    mine = (await client.get("/ui/api/sessions")).json()["sessions"]
    entry = next(s for s in mine if s["id"] == r["session_id"])
    assert entry["saved"] is False  # guest runs are memory-only

    # Nothing was written to the session store for the guest run.
    assert not (Path(settings.sessions_dir) / r["session_id"]).exists()

    # A different browser (no shared cookie) is a different guest — it must not
    # see or open the first guest's session.
    transport = client._transport
    assert isinstance(transport, ASGITransport)
    async with AsyncClient(transport=transport, base_url="http://test") as other:
        listed = (await other.get("/ui/api/sessions")).json()["sessions"]
        assert all(s["id"] != r["session_id"] for s in listed)
        assert (await other.get(f"/ui/api/sessions/{r['session_id']}")).status_code == 404
        assert (await other.get(f"/ui/api/tasks/{r['task_id']}")).status_code == 404
        assert (
            await other.post(f"/ui/api/sessions/{r['session_id']}/followup", json={"goal": "x"})
        ).status_code == 404


async def test_account_sessions_persisted(client, settings):
    # Sign the client in by planting a stored-account cookie (no network).
    user = upsert_google_user(settings, {"email": "own@x.com", "name": "Owner", "picture": ""})
    client.cookies.set(users_mod.COOKIE_NAME, encode_cookie(settings, user))
    me = (await client.get("/ui/api/me")).json()
    assert me == {
        "id": user.id,
        "name": "Owner",
        "kind": "account",
        "email": "own@x.com",
        "google_signin_available": False,
    }

    r = (await client.post("/ui/api/tasks", json={"goal": "account persistence probe"})).json()
    await _wait_done(client, r["task_id"])
    meta_file = Path(settings.sessions_dir) / r["session_id"] / "session.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["user_id"] == user.id and meta["user_name"] == "Owner"

    listed = (await client.get("/ui/api/sessions")).json()["sessions"]
    entry = next(s for s in listed if s["id"] == r["session_id"])
    assert entry["saved"] is True and entry["user_name"] == "Owner"

    # Clean up so the persisted session doesn't leak into other tests.
    await client.delete(f"/ui/api/sessions/{r['session_id']}")


async def test_unowned_sessions_hidden_from_everyone(client, settings, employee):
    """ChatGPT-style scoping: pre-accounts (unowned) sessions appear for nobody —
    guests start with an empty history, accounts see only their own tasks."""
    from ai_employee.observability import get_task_manager

    tm = get_task_manager()
    legacy = tm.start(employee, "legacy unowned goal")  # no owner recorded
    try:
        # Guest browser: not listed, not openable.
        listed = (await client.get("/ui/api/sessions")).json()["sessions"]
        assert all(s["id"] != legacy.session_id for s in listed)
        assert (await client.get(f"/ui/api/sessions/{legacy.session_id}")).status_code == 404
        assert (await client.get(f"/ui/api/tasks/{legacy.id}")).status_code == 404

        # Signed-in account: same — only its own sessions show.
        user = upsert_google_user(settings, {"email": "acc@x.com", "name": "Acc", "picture": ""})
        client.cookies.set(users_mod.COOKIE_NAME, encode_cookie(settings, user))
        listed = (await client.get("/ui/api/sessions")).json()["sessions"]
        assert all(s["id"] != legacy.session_id for s in listed)
        assert (await client.get(f"/ui/api/sessions/{legacy.session_id}")).status_code == 404
    finally:
        client.cookies.delete(users_mod.COOKIE_NAME)
        tm.delete_session(legacy.session_id)


async def test_delete_all_only_touches_own_sessions(client, settings):
    from httpx import ASGITransport, AsyncClient

    mine = (await client.post("/ui/api/tasks", json={"goal": "mine"})).json()
    await _wait_done(client, mine["task_id"])

    transport = client._transport
    assert isinstance(transport, ASGITransport)
    async with AsyncClient(transport=transport, base_url="http://test") as other:
        theirs = (await other.post("/ui/api/tasks", json={"goal": "theirs"})).json()
        await _wait_done(other, theirs["task_id"])
        await client.delete("/ui/api/sessions")
        listed = (await other.get("/ui/api/sessions")).json()["sessions"]
        assert any(s["id"] == theirs["session_id"] for s in listed)
        assert all(s["id"] != mine["session_id"] for s in listed)
