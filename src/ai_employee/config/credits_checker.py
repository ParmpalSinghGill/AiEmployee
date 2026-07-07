"""Live credit / balance checks for managed LLM keys.

``test_key`` (in :mod:`key_tester`) only proves a key *authenticates* — a key can
pass that check yet have zero usable credit. This module answers the next
question: *does this key have money behind it?*

Not every provider exposes a balance API. Where one exists (e.g. DeepSeek) we
return the real number; where it doesn't, we say so plainly and point at the
provider's billing dashboard rather than guessing. ``check_credits`` never
raises: it always returns ``{ok, supported, message, detail, balance}``.
"""

from __future__ import annotations

from typing import Any

import httpx

_UA = {"User-Agent": "AIEmployee-Research/0.1"}


def _result(
    *,
    ok: bool,
    supported: bool,
    message: str,
    detail: str = "",
    balance: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "supported": supported,
        "message": message,
        "detail": detail,
        "balance": balance,
    }


def _unsupported(provider: str, dashboard: str) -> dict[str, Any]:
    return _result(
        ok=True,
        supported=False,
        message=f"{provider} has no balance API — check credits in the dashboard.",
        detail=dashboard,
    )


def _auth_failure(status: int) -> dict[str, Any] | None:
    if status in (401, 403):
        return _result(
            ok=False,
            supported=True,
            message="Key rejected (unauthorized). Check the key value.",
            detail=f"HTTP {status}",
        )
    if status == 429:
        return _result(
            ok=False, supported=True, message="Rate limited — try again shortly.", detail="HTTP 429"
        )
    if status >= 500:
        return _result(
            ok=False,
            supported=True,
            message="Provider error (not your key). Try again later.",
            detail=f"HTTP {status}",
        )
    return None


# ── Providers with a real balance endpoint ───────────────────────────────────
async def _deepseek_balance(base_url: str, key: str) -> dict[str, Any]:
    """DeepSeek exposes GET /user/balance → is_available + per-currency balances."""
    url = base_url.rstrip("/") + "/user/balance"
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        r = await client.get(url, headers={**_UA, "Authorization": f"Bearer {key}"})
    bad = _auth_failure(r.status_code)
    if bad:
        return bad
    try:
        data = r.json()
    except Exception:  # noqa: BLE001 - non-JSON is a provider-side problem
        return _result(
            ok=False,
            supported=True,
            message="Unexpected response from DeepSeek.",
            detail=r.text[:200],
        )
    infos = data.get("balance_infos") or []
    parts = [f"{i.get('total_balance', '?')} {i.get('currency', '')}".strip() for i in infos]
    balance = ", ".join(p for p in parts if p) or "unknown"
    available = bool(data.get("is_available"))
    if available:
        return _result(ok=True, supported=True, message=f"Balance: {balance}.", balance=balance)
    return _result(
        ok=False,
        supported=True,
        message=f"No usable credit (balance: {balance}). Top up to run this brain.",
        balance=balance,
    )


# name → (provider label, async checker) for keys with a real balance API.
_BALANCE_CHECKERS = {
    "deepseek_api_key": lambda s, v: _deepseek_balance(s.deepseek_base_url, v),
}

# name → dashboard URL for providers without a programmatic balance endpoint.
_DASHBOARDS = {
    "openai_api_key": ("OpenAI", "https://platform.openai.com/settings/organization/billing"),
    "google_api_key": ("Google Gemini", "https://aistudio.google.com/app/plan_information"),
    "glm_api_key": ("Zhipu GLM", "https://open.bigmodel.cn/usercenter/proapikeys"),
    "opencode_api_key": ("OpenCode Zen", "https://opencode.ai/auth"),
}

# LLM provider id → managed key name (for brain-level credit checks).
_PROVIDER_KEY = {
    "deepseek": "deepseek_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "glm": "glm_api_key",
    "opencode": "opencode_api_key",
}


async def check_credits(settings: Any, name: str, value: str | None = None) -> dict[str, Any]:
    """Check remaining credit/balance for a managed LLM key.

    ``value`` (if given) is used instead of the stored key, so the UI can check
    before saving. Returns ``supported=False`` for providers with no balance API.
    """
    val = (value or "").strip() or (getattr(settings, name, "") or "")
    if not val:
        return _result(
            ok=False, supported=True, message="No key to check — save or type a value first."
        )

    checker = _BALANCE_CHECKERS.get(name)
    if checker is not None:
        try:
            return await checker(settings, val)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            return _result(
                ok=False,
                supported=True,
                message="Could not reach the provider — network/proxy issue, not the key.",
                detail=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - report, never raise
            return _result(
                ok=False,
                supported=True,
                message=f"Check failed: {type(exc).__name__}",
                detail=str(exc),
            )

    if name in _DASHBOARDS:
        provider, dashboard = _DASHBOARDS[name]
        return _unsupported(provider, dashboard)

    return _result(ok=True, supported=False, message="Credit check isn't available for this key.")


async def check_brain_credits(
    settings: Any, provider: str, api_key: str | None = None
) -> dict[str, Any]:
    """Check remaining credit/balance for a brain provider (hosted APIs only).

    ``mock`` and ``local`` skip the check. For other providers, delegates to
    :func:`check_credits` using the provider's managed key (or ``api_key`` override).
    """
    if provider in ("mock", "local"):
        return _result(
            ok=True,
            supported=False,
            message="Local and mock brains have no remote credit balance.",
        )
    key_name = _PROVIDER_KEY.get(provider)
    if not key_name:
        return _result(
            ok=True, supported=False, message=f"Credit check isn't available for {provider}."
        )
    return await check_credits(settings, key_name, api_key)


__all__ = ["check_credits", "check_brain_credits"]
