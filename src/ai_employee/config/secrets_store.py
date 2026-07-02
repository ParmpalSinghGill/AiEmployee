"""Runtime-managed API keys.

A small persistence layer so keys (Roboflow, Kaggle, GitHub, Tavily, LLM
providers) can be added/removed from the UI instead of editing ``.env``. Keys are
stored in a JSON file (``settings.secrets_file``) and overlaid onto the live
:class:`~ai_employee.config.settings.Settings` object at startup and on change.

Precedence: ``.env`` / environment is the base layer; the JSON store overlays on
top. Removing a managed key drops the overlay (an env-provided value, if any,
reappears on restart). Values are masked when listed — the full secret is never
returned to the client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("secrets")


@dataclass(frozen=True)
class KeySpec:
    name: str  # public id (also the settings attribute name)
    label: str
    category: str  # "search" | "llm"
    description: str
    requires_rebuild: bool = False  # LLM keys need the brain rebuilt to take effect
    secret: bool = True  # mask when listing (usernames are not secret)


MANAGED_KEYS: tuple[KeySpec, ...] = (
    KeySpec("roboflow_api_key", "Roboflow API key", "search",
            "Roboflow Universe — computer-vision datasets & models."),
    KeySpec("kaggle_username", "Kaggle username", "search",
            "Kaggle account username (paired with the Kaggle API key).", secret=False),
    KeySpec("kaggle_key", "Kaggle API key", "search", "Kaggle API token for dataset search."),
    KeySpec("github_token", "GitHub token", "search",
            "Raises the GitHub repo-search rate limit."),
    KeySpec("tavily_api_key", "Tavily API key", "search",
            "Higher-quality web search than the keyless fallback."),
    KeySpec("deepseek_api_key", "DeepSeek API key", "llm",
            "DeepSeek LLM provider.", requires_rebuild=True),
    KeySpec("openai_api_key", "OpenAI API key", "llm",
            "OpenAI LLM provider.", requires_rebuild=True),
    KeySpec("google_api_key", "Google (Gemini) API key", "llm",
            "Google Gemini LLM provider.", requires_rebuild=True),
    KeySpec("glm_api_key", "GLM API key", "llm",
            "Zhipu GLM LLM provider.", requires_rebuild=True),
    KeySpec("opencode_api_key", "OpenCode Zen API key", "llm",
            "OpenCode Zen gateway (Claude/GPT/DeepSeek/GLM/… via one key).",
            requires_rebuild=True),
)

_BY_NAME = {k.name: k for k in MANAGED_KEYS}


def spec_for(name: str) -> KeySpec:
    if name not in _BY_NAME:
        raise KeyError(name)
    return _BY_NAME[name]


def _path(settings: Any) -> Path:
    return Path(settings.secrets_file)


def _load(settings: Any) -> dict[str, str]:
    p = _path(settings)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a corrupt store must not crash startup
        log.warning("secrets.load_failed", error=str(exc))
        return {}


def _save(settings: Any, data: dict[str, str]) -> None:
    p = _path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    import contextlib

    with contextlib.suppress(OSError):  # best-effort: keep secrets owner-only
        p.chmod(0o600)


def apply_secrets(settings: Any) -> None:
    """Overlay stored keys onto the live settings object (called at startup)."""
    for name, value in _load(settings).items():
        spec = _BY_NAME.get(name)
        if spec is not None and isinstance(value, str):
            setattr(settings, name, value)


def set_key(settings: Any, name: str, value: str) -> KeySpec:
    spec = spec_for(name)
    data = _load(settings)
    data[name] = value
    _save(settings, data)
    setattr(settings, name, value)
    log.info("secrets.set", key=name)
    return spec


def delete_key(settings: Any, name: str) -> KeySpec:
    spec = spec_for(name)
    data = _load(settings)
    data.pop(name, None)
    _save(settings, data)
    setattr(settings, name, "")
    log.info("secrets.delete", key=name)
    return spec


def _mask(value: str, secret: bool) -> str | None:
    if not value:
        return None
    if not secret:
        return value
    return "••••" + value[-4:] if len(value) > 4 else "••••"


def list_keys(settings: Any) -> list[dict[str, Any]]:
    """Describe every managed key with a *masked* value — never the raw secret."""
    out: list[dict[str, Any]] = []
    for spec in MANAGED_KEYS:
        value = getattr(settings, spec.name, "") or ""
        out.append(
            {
                "name": spec.name,
                "label": spec.label,
                "category": spec.category,
                "description": spec.description,
                "configured": bool(value),
                "masked": _mask(value, spec.secret),
                "secret": spec.secret,
                "requires_rebuild": spec.requires_rebuild,
            }
        )
    return out


__all__ = [
    "MANAGED_KEYS",
    "KeySpec",
    "apply_secrets",
    "set_key",
    "delete_key",
    "list_keys",
    "spec_for",
]
