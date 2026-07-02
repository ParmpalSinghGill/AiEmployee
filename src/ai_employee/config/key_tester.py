"""Live validity checks for managed API keys.

Each tester makes one cheap authenticated request and classifies the outcome so
the UI can tell the user *why* a key failed — rejected (wrong key) vs. a network
or proxy problem they need to fix — rather than a generic error.

``test_key`` never raises: it always returns ``{ok, message, detail}``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ai_employee.config.secrets_store import spec_for

_UA = {"User-Agent": "AIEmployee-Research/0.1"}


def _ok(message: str, detail: str = "") -> dict[str, Any]:
    return {"ok": True, "message": message, "detail": detail}


def _fail(message: str, detail: str = "") -> dict[str, Any]:
    return {"ok": False, "message": message, "detail": detail}


def _classify_status(status: int) -> dict[str, Any] | None:
    """Map a generic auth status to a failure; return None if it looks fine."""
    if status in (401, 403):
        return _fail("Key rejected (unauthorized). Check the key value.", f"HTTP {status}")
    if status == 429:
        return _fail("Rate limited — key may be valid but throttled.", "HTTP 429")
    if status >= 500:
        return _fail("Provider error (not your key). Try again later.", f"HTTP {status}")
    return None


async def _get(url: str, *, headers: dict[str, str] | None = None, auth=None) -> httpx.Response:
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        return await client.get(url, headers={**_UA, **(headers or {})}, auth=auth)


async def _test_roboflow(s: Any, key: str) -> dict[str, Any]:
    r = await _get(f"https://api.roboflow.com/universe/search?q=test&api_key={key}")
    bad = _classify_status(r.status_code)
    return bad or _ok("Roboflow key works.", f"HTTP {r.status_code}")


async def _test_kaggle(s: Any, user: str, key: str) -> dict[str, Any]:
    if not (user and key):
        return _fail("Kaggle needs BOTH a username and an API key.")
    r = await _get("https://www.kaggle.com/api/v1/datasets/list?search=test", auth=(user, key))
    bad = _classify_status(r.status_code)
    return bad or _ok("Kaggle credentials work.", f"HTTP {r.status_code}")


async def _test_github(s: Any, token: str) -> dict[str, Any]:
    r = await _get("https://api.github.com/user", headers={"Authorization": f"Bearer {token}"})
    bad = _classify_status(r.status_code)
    return bad or _ok("GitHub token works.", f"HTTP {r.status_code}")


async def _test_tavily(s: Any, key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": "test", "max_results": 1},
        )
    bad = _classify_status(r.status_code)
    return bad or _ok("Tavily key works.", f"HTTP {r.status_code}")


async def _test_openai_compatible(base_url: str, key: str, label: str) -> dict[str, Any]:
    r = await _get(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
    bad = _classify_status(r.status_code)
    return bad or _ok(f"{label} key works.", f"HTTP {r.status_code}")


async def _test_gemini(s: Any, key: str) -> dict[str, Any]:
    r = await _get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
    bad = _classify_status(r.status_code)
    if r.status_code == 400:
        return _fail("Key rejected (bad request — usually an invalid key).", "HTTP 400")
    return bad or _ok("Gemini key works.", f"HTTP {r.status_code}")


async def test_key(settings: Any, name: str, value: str | None = None) -> dict[str, Any]:
    """Test a managed key. ``value`` (if given) is tested instead of the stored one,
    so the UI can validate before saving."""
    spec_for(name)  # validates the name (raises KeyError for unknown keys)
    val = (value or "").strip() or (getattr(settings, name, "") or "")

    if name == "kaggle_username":
        user = val
        keyval = getattr(settings, "kaggle_key", "")
    elif name == "kaggle_key":
        user = getattr(settings, "kaggle_username", "")
        keyval = val
    else:
        if not val:
            return _fail("No key value to test — type a value or save one first.")

    try:
        if name == "roboflow_api_key":
            return await _test_roboflow(settings, val)
        if name in ("kaggle_username", "kaggle_key"):
            return await _test_kaggle(settings, user, keyval)
        if name == "github_token":
            return await _test_github(settings, val)
        if name == "tavily_api_key":
            return await _test_tavily(settings, val)
        if name == "deepseek_api_key":
            return await _test_openai_compatible(settings.deepseek_base_url, val, "DeepSeek")
        if name == "openai_api_key":
            return await _test_openai_compatible(settings.openai_base_url, val, "OpenAI")
        if name == "glm_api_key":
            return await _test_openai_compatible(settings.glm_base_url, val, "GLM")
        if name == "opencode_api_key":
            return await _test_openai_compatible(settings.opencode_base_url, val, "OpenCode Zen")
        if name == "google_api_key":
            return await _test_gemini(settings, val)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
        return _fail(
            "Could not reach the provider — network/proxy issue, not the key. "
            "Configure a proxy or check connectivity.",
            f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - any other failure is reported, not raised
        return _fail(f"Test failed: {type(exc).__name__}", str(exc))

    return _fail(f"No tester available for {name!r}.")  # unreachable for managed keys


__all__ = ["test_key"]
