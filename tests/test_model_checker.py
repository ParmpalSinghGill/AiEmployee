"""Live model checks for the brain picker.

Two distinct questions, and the tests assert they are answered separately:

1. Is the model *listed* by the provider?  (necessary, not sufficient)
2. Will the API *actually serve* a request for it?  (the real entitlement test)

A model can appear in ``/models`` and still fail on generate — that is the exact
failure this feature exists to catch, so it gets explicit coverage.
"""

from __future__ import annotations

import httpx
import pytest

from ai_employee.config import model_checker
from ai_employee.config.model_checker import check_model, list_models, match_model, probe_model

GEMINI_MODELS = {
    "models": [
        {"name": "models/gemini-2.5-pro"},
        {"name": "models/gemini-3.8-flash"},
        {"name": "models/gemini-embedding-001"},
    ]
}
OPENAI_MODELS = {"data": [{"id": "gpt-4o"}, {"id": "o3-mini"}]}


def _patch(monkeypatch, handler):
    """Route httpx.AsyncClient through a MockTransport."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(model_checker.httpx, "AsyncClient", factory)


def _routes(*, models=None, generate=None):
    """Handler dispatching on URL: model listing vs. generation."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if ":generateContent" in url or url.endswith("/chat/completions"):
            assert generate is not None, f"unexpected generation call to {url}"
            return generate(request)
        return models(request) if models else httpx.Response(200, json=GEMINI_MODELS)

    return handler


# ── matching ────────────────────────────────────────────────────────────────
def test_normalize_and_match_strip_gemini_prefix() -> None:
    assert match_model("gemini-3.8-flash", ["models/gemini-3.8-flash"]) == "models/gemini-3.8-flash"
    assert match_model("models/gemini-3.8-flash", ["gemini-3.8-flash"]) == "gemini-3.8-flash"
    assert match_model("GEMINI-3.8-FLASH", ["gemini-3.8-flash"]) == "gemini-3.8-flash"
    assert match_model("gemini-9.9-nope", ["gemini-3.8-flash"]) is None


# ── listing ─────────────────────────────────────────────────────────────────
async def test_gemini_listing(monkeypatch, settings) -> None:
    settings.google_api_key = "test-key"

    def models(request):
        assert "generativelanguage.googleapis.com" in str(request.url)
        assert "key=test-key" in str(request.url)
        return httpx.Response(200, json=GEMINI_MODELS)

    _patch(monkeypatch, _routes(models=models))
    res = await list_models(settings, "gemini")
    assert res["ok"] is True
    assert res["count"] == 3


async def test_openai_compatible_listing(monkeypatch, settings) -> None:
    settings.openai_api_key = "sk-test"

    def models(request):
        assert str(request.url).endswith("/models")
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json=OPENAI_MODELS)

    _patch(monkeypatch, _routes(models=models))
    res = await list_models(settings, "openai")
    assert res["ok"] is True
    assert res["count"] == 2


# ── the real test: is it callable? ──────────────────────────────────────────
async def test_listed_and_callable_passes(monkeypatch, settings) -> None:
    """The good path: listed AND a real generate request succeeds."""
    settings.google_api_key = "k"
    _patch(
        monkeypatch,
        _routes(generate=lambda request: httpx.Response(200, json={"candidates": []})),
    )
    res = await check_model(settings, "gemini", "gemini-3.8-flash")
    assert res["listed"] is True
    assert res["callable"] is True
    assert res["available"] is True
    assert res["ok"] is True
    assert res["error"] is None


async def test_listed_but_not_callable_is_the_key_case(monkeypatch, settings) -> None:
    """The bug this feature exists for: the model IS listed, but the API refuses it.

    Previously the checker reported success here, so the UI showed an available
    model that then produced no response when a task actually ran.
    """
    settings.google_api_key = "k"
    _patch(
        monkeypatch,
        _routes(
            generate=lambda request: httpx.Response(
                403,
                json={"error": {"status": "PERMISSION_DENIED", "message": "not entitled"}},
            )
        ),
    )
    res = await check_model(settings, "gemini", "gemini-3.8-flash")
    assert res["listed"] is True, "the model appears in the provider's list"
    assert res["callable"] is False, "but a real request is refused"
    assert res["ok"] is False
    assert res["error"] == "forbidden"
    assert "in the model list" in res["message"]


async def test_not_listed_but_callable_is_reported_honestly(monkeypatch, settings) -> None:
    """Some providers omit models from the list yet still serve them."""
    settings.google_api_key = "k"
    _patch(
        monkeypatch,
        _routes(generate=lambda request: httpx.Response(200, json={"candidates": []})),
    )
    res = await check_model(settings, "gemini", "gemini-9.9-hidden")
    assert res["listed"] is False
    assert res["callable"] is True
    assert res["ok"] is True
    assert "not in the model list" in res["message"]


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (401, "auth"),
        (403, "forbidden"),
        (404, "model_unavailable"),
        (429, "quota"),
        (500, "provider_error"),
    ],
)
async def test_probe_error_classification(monkeypatch, settings, status, expected_code) -> None:
    settings.google_api_key = "k"
    _patch(monkeypatch, _routes(generate=lambda request: httpx.Response(status, json={})))
    res = await probe_model(settings, "gemini", "gemini-3.8-flash")
    assert res["callable"] is False
    assert res["error"] == expected_code
    assert res["http_status"] == status


async def test_probe_detects_unsupported_model_in_400(monkeypatch, settings) -> None:
    settings.google_api_key = "k"
    _patch(
        monkeypatch,
        _routes(
            generate=lambda request: httpx.Response(
                400, json={"error": {"message": "model not supported"}}
            )
        ),
    )
    res = await probe_model(settings, "gemini", "gemini-3.8-flash")
    assert res["error"] == "model_unavailable"


async def test_openai_probe_retries_with_max_completion_tokens(monkeypatch, settings) -> None:
    """o-series models reject ``max_tokens``; the probe must not misreport them."""
    settings.openai_api_key = "sk-test"
    calls: list[dict] = []

    def generate(request):
        import json

        body = json.loads(request.content)
        calls.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: 'max_tokens' is not supported"}},
            )
        return httpx.Response(200, json={"choices": []})

    _patch(monkeypatch, _routes(generate=generate))
    res = await probe_model(settings, "openai", "o3-mini")
    assert res["callable"] is True
    assert len(calls) == 2
    assert "max_completion_tokens" in calls[1]


async def test_openai_probe_sends_the_model_and_prompt(monkeypatch, settings) -> None:
    settings.deepseek_api_key = "sk-d"
    seen: dict = {}

    def generate(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    _patch(monkeypatch, _routes(generate=generate))
    res = await probe_model(settings, "deepseek", "deepseek-chat")
    assert res["callable"] is True
    assert seen["model"] == "deepseek-chat"
    assert seen["messages"][0]["role"] == "user"
    assert seen["max_tokens"] == 16


# ── failure modes that must not raise ───────────────────────────────────────
async def test_missing_key_is_reported(monkeypatch, settings) -> None:
    settings.google_api_key = ""
    res = await check_model(settings, "gemini", "gemini-3.8-flash")
    assert res["ok"] is False
    assert res["error"] == "missing_key"
    assert res["callable"] is None


async def test_rejected_key_on_listing(monkeypatch, settings) -> None:
    settings.google_api_key = "bad"
    _patch(monkeypatch, lambda request: httpx.Response(403, json={}))
    res = await check_model(settings, "gemini", "gemini-3.8-flash")
    assert res["ok"] is False
    assert res["error"] == "forbidden"


async def test_unreachable_provider(monkeypatch, settings) -> None:
    settings.google_api_key = "k"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    _patch(monkeypatch, handler)
    res = await check_model(settings, "gemini", "gemini-3.8-flash")
    assert res["ok"] is False
    assert res["error"] == "unreachable"


async def test_local_provider_needs_no_key(monkeypatch, settings) -> None:
    settings.local_api_key = ""
    settings.local_base_url = "http://localhost:11434/v1"
    _patch(monkeypatch, lambda request: httpx.Response(200, json={"data": [{"id": "qwen3:8b"}]}))
    res = await list_models(settings, "local")
    assert res["ok"] is True
    assert res["models"] == ["qwen3:8b"]


async def test_mock_provider_is_unsupported() -> None:
    from ai_employee.config import get_settings

    res = await list_models(get_settings(), "mock")
    assert res["supported"] is False
    probed = await probe_model(get_settings(), "mock", "mock-1")
    assert probed["supported"] is False
    assert probed["callable"] is None


async def test_unknown_provider_unsupported() -> None:
    from ai_employee.config import get_settings

    res = await list_models(get_settings(), "nope")
    assert res["supported"] is False


@pytest.mark.parametrize("payload", [{"data": []}, {"models": []}, {}])
async def test_empty_model_list(monkeypatch, settings, payload) -> None:
    settings.google_api_key = "k"
    _patch(monkeypatch, lambda request: httpx.Response(200, json=payload))
    res = await list_models(settings, "gemini")
    assert res["ok"] is False
    assert res["error"] == "empty"
