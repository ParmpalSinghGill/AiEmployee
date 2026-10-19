"""Live model checks against a provider's server.

Two different questions matter, and they are *not* the same:

1. **Is the model listed?** A provider's docs/catalog can list a model that a given
   API key is not entitled to call (tier limits, regional rollout, preview gating,
   project-level model enablement). Listing it proves nothing.
2. **Will the API actually serve a request for it?** Only a real call answers this.
   The common failure is a model that appears in ``/models`` yet returns 403/404/429
   the moment you generate with it — which looks like "the agent produced no
   response" in the UI.

So :func:`check_model` does both, in order, and reports each outcome separately.
Everything is best-effort and never raises.
"""

from __future__ import annotations

from typing import Any

import httpx

_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TIMEOUT = 8.0
#: A generation probe is a real round-trip to the model; allow more time than a list.
_PROBE_TIMEOUT = 25.0
_MAX_REPORTED = 200
_PROBE_PROMPT = "Reply with the single word: ok"

_OPENAI_COMPAT_BASE = {
    "deepseek": "deepseek_base_url",
    "openai": "openai_base_url",
    "glm": "glm_base_url",
    "opencode": "opencode_base_url",
    "local": "local_base_url",
}
_KEY_ATTR = {
    "deepseek": "deepseek_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "glm": "glm_api_key",
    "opencode": "opencode_api_key",
    "local": "local_api_key",
}


def normalize_model(name: str) -> str:
    """Canonical form for comparison: drop Gemini's ``models/`` prefix, lowercase."""
    n = (name or "").strip()
    if n.lower().startswith("models/"):
        n = n[len("models/") :]
    return n.lower()


def match_model(model: str, available: list[str]) -> str | None:
    """Return the provider's own spelling of ``model`` if present, else ``None``."""
    want = normalize_model(model)
    for candidate in available:
        if normalize_model(candidate) == want:
            return candidate
    return None


def _resolve_key(settings: Any, provider: str, api_key: str | None) -> str:
    if api_key:
        return api_key.strip()
    attr = _KEY_ATTR.get(provider)
    return (getattr(settings, attr, "") or "").strip() if attr else ""


def _result(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ok": False,
        "supported": True,
        "provider": "",
        "model": "",
        "listed": None,
        "callable": None,
        "available": None,  # back-compat alias of ``callable``
        "matched": None,
        "models": [],
        "count": 0,
        "source": "",
        "http_status": None,
        "message": "",
        "detail": "",
        "error": None,
    }
    base.update(kw)
    return base


def _snippet(resp: httpx.Response, limit: int = 400) -> str:
    """Best-effort extraction of the provider's own error text."""
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return (resp.text or "").strip()[:limit]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            parts = [str(err.get(k)) for k in ("status", "code", "message") if err.get(k)]
            if parts:
                return " | ".join(parts)[:limit]
        if isinstance(err, str):
            return err[:limit]
        for key in ("message", "detail", "msg"):
            if body.get(key):
                return str(body[key])[:limit]
    return str(body)[:limit]


def _classify(status: int, detail: str) -> tuple[str, str]:
    """Map an HTTP failure to (machine code, human explanation)."""
    low = (detail or "").lower()
    if status == 401:
        return "auth", "The API key was rejected (unauthorized)."
    if status == 403:
        return "forbidden", (
            "Access denied for this model — the key/project is not entitled to it "
            "(tier, region, or the model is not enabled for this project)."
        )
    if status == 404:
        return "model_unavailable", (
            "The provider does not serve this model on this endpoint (unknown or "
            "not enabled for this API version)."
        )
    if status == 429:
        return "quota", (
            "Rate limited or out of quota — the model exists but this key cannot "
            "spend on it right now."
        )
    if status >= 500:
        return "provider_error", "The provider failed on its side. Try again later."
    if status == 400:
        if any(
            s in low
            for s in (
                "not supported",
                "unsupported",
                "not found",
                "does not exist",
                "invalid model",
                "unknown model",
                "not available",
                "no such model",
            )
        ):
            return "model_unavailable", "The provider rejected this model as unavailable."
        return "bad_request", "The provider rejected the request."
    return "http_error", f"The provider returned HTTP {status}."


async def list_models(
    settings: Any, provider: str, api_key: str | None = None
) -> dict[str, Any]:
    """Ask ``provider`` which models the configured key can see."""
    if provider == "mock":
        return _result(
            supported=False,
            provider=provider,
            message="The mock provider is offline and has no model list.",
        )
    if provider not in _OPENAI_COMPAT_BASE and provider != "gemini":
        return _result(
            supported=False, provider=provider, message=f"Unknown provider: {provider}"
        )

    key = _resolve_key(settings, provider, api_key)
    if not key and provider != "local":
        return _result(
            provider=provider,
            message="No API key configured for this provider.",
            error="missing_key",
        )

    headers = {"User-Agent": "AIEmployee-ModelCheck/0.1"}
    if provider == "gemini":
        url = f"{_GEMINI_MODELS_URL}?key={key}"
        source = "gemini-native"
    else:
        base = getattr(settings, _OPENAI_COMPAT_BASE[provider], "") or ""
        if not base:
            return _result(
                supported=False,
                provider=provider,
                message=f"No base URL configured for {provider}.",
            )
        url = base.rstrip("/") + "/models"
        headers["Authorization"] = f"Bearer {key}"
        source = "openai-compatible"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
    except Exception as exc:  # noqa: BLE001 - network problems are reported, not raised
        return _result(
            provider=provider,
            source=source,
            message=f"Could not reach the provider: {type(exc).__name__}",
            error="unreachable",
        )

    if not resp.is_success:
        code, msg = _classify(resp.status_code, _snippet(resp))
        return _result(
            provider=provider,
            source=source,
            http_status=resp.status_code,
            message=msg,
            detail=_snippet(resp),
            error=code,
        )

    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return _result(
            provider=provider,
            source=source,
            message="Provider returned non-JSON.",
            error="bad_response",
        )

    ids: list[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):  # OpenAI-compatible
            ids = [
                str(m.get("id"))
                for m in payload["data"]
                if isinstance(m, dict) and m.get("id")
            ]
        elif isinstance(payload.get("models"), list):  # Gemini native
            ids = [
                str(m.get("name"))
                for m in payload["models"]
                if isinstance(m, dict) and m.get("name")
            ]
    if not ids:
        return _result(
            provider=provider,
            source=source,
            message="The provider returned an empty model list.",
            error="empty",
        )

    ids = sorted(ids)
    return _result(
        ok=True,
        provider=provider,
        source=source,
        models=ids[:_MAX_REPORTED],
        count=len(ids),
        message=f"{len(ids)} model(s) available.",
    )


def _openai_payload(model: str, *, token_field: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _PROBE_PROMPT}],
        token_field: 16,
        "temperature": 0,
    }


async def probe_model(
    settings: Any, provider: str, model: str, api_key: str | None = None
) -> dict[str, Any]:
    """Make one minimal generation call and report whether the API serves ``model``.

    This is the only reliable test of entitlement: a model can be listed and still
    return 403/404/429 when you actually generate with it.
    """
    if provider == "mock":
        return _result(
            supported=False,
            provider=provider,
            model=model,
            message="The mock provider is offline — nothing to probe.",
        )
    if provider not in _OPENAI_COMPAT_BASE and provider != "gemini":
        return _result(
            supported=False, provider=provider, model=model, message=f"Unknown provider: {provider}"
        )

    key = _resolve_key(settings, provider, api_key)
    if not key and provider != "local":
        return _result(
            provider=provider,
            model=model,
            message="No API key configured for this provider.",
            error="missing_key",
        )

    bare = normalize_model(model)
    headers = {"User-Agent": "AIEmployee-ModelProbe/0.1"}

    async def _send(payload: dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            if provider == "gemini":
                url = _GEMINI_GENERATE_URL.format(model=bare) + f"?key={key}"
                return await client.post(url, headers=headers, json=payload)
            base = getattr(settings, _OPENAI_COMPAT_BASE[provider], "") or ""
            return await client.post(
                base.rstrip("/") + "/chat/completions",
                headers={**headers, "Authorization": f"Bearer {key}"},
                json=payload,
            )

    if provider == "gemini":
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": _PROBE_PROMPT}]}],
            "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
        }
        source = "gemini-native"
    else:
        payload = _openai_payload(model, token_field="max_tokens")
        source = "openai-compatible"

    try:
        resp = await _send(payload)
        # Reasoning models (e.g. o-series) reject ``max_tokens`` and require
        # ``max_completion_tokens`` — retry once so a valid model isn't misreported.
        if (
            provider != "gemini"
            and resp.status_code == 400
            and "max_tokens" in _snippet(resp).lower()
        ):
            resp = await _send(_openai_payload(model, token_field="max_completion_tokens"))
    except Exception as exc:  # noqa: BLE001
        return _result(
            provider=provider,
            model=model,
            source=source,
            message=f"Could not reach the provider: {type(exc).__name__}",
            error="unreachable",
        )

    if resp.is_success:
        return _result(
            ok=True,
            provider=provider,
            model=model,
            source=source,
            callable=True,
            available=True,
            http_status=resp.status_code,
            message=f"'{model}' answered a test request — this key can use it.",
        )

    detail = _snippet(resp)
    code, msg = _classify(resp.status_code, detail)
    return _result(
        provider=provider,
        model=model,
        source=source,
        callable=False,
        available=False,
        http_status=resp.status_code,
        message=msg,
        detail=detail,
        error=code,
    )


async def check_model(
    settings: Any,
    provider: str,
    model: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Check *both* whether ``model`` is listed and whether the API will serve it.

    ``listed`` answers "does it appear in the provider's model list?", ``callable``
    answers "did a real generation request succeed?". They can disagree — that
    disagreement is the whole point of this check.
    """
    listed = await list_models(settings, provider, api_key)
    out = dict(listed)
    out["model"] = model

    if listed.get("ok"):
        found = match_model(model, listed["models"])
        out["listed"] = found is not None
        out["matched"] = found
        if found is None:
            out["message"] = (
                f"'{model}' is NOT in the list of {listed['count']} model(s) this key can see."
            )

    if not listed.get("supported", True):
        # Provider has no model list / no probing (mock, unknown).
        out["callable"] = None
        out["available"] = None
        return out

    probe = await probe_model(settings, provider, model, api_key)
    out["callable"] = probe.get("callable")
    out["available"] = probe.get("callable")
    out["probe"] = {
        "ok": probe.get("ok"),
        "http_status": probe.get("http_status"),
        "error": probe.get("error"),
        "message": probe.get("message"),
        "detail": probe.get("detail"),
    }

    if probe.get("callable") is True:
        if out.get("listed") is False:
            out["message"] = (
                f"'{model}' is not in the model list, but the API served a test request."
            )
        elif out.get("listed") is None:
            out["message"] = f"'{model}' answered a test request — this key can use it."
        else:
            out["message"] = f"'{model}' is listed and answered a test request."
        out["ok"] = True
        out["error"] = None
        out["detail"] = ""
    else:
        out["ok"] = False
        out["error"] = probe.get("error")
        out["detail"] = probe.get("detail", "")
        out["http_status"] = probe.get("http_status")
        listed_note = (
            " (it IS in the model list, but the API refuses to serve it)"
            if out.get("listed")
            else ""
        )
        out["message"] = probe.get("message", "The model could not be used.") + listed_note
    return out
