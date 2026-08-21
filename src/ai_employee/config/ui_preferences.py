"""UI-managed defaults for brain provider/model and coding backend.

Persists choices made in the web console so they survive page refresh and server
restart without editing ``.env``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_employee.config.settings import LLMProvider, Settings
from ai_employee.core.logging import get_logger

log = get_logger("ui_prefs")

_CHAT_MODEL_ATTR = {
    "deepseek": "deepseek_chat_model",
    "openai": "openai_chat_model",
    "gemini": "gemini_chat_model",
    "glm": "glm_chat_model",
    "opencode": "opencode_chat_model",
    "local": "local_chat_model",
}


def _path(settings: Settings) -> Path:
    return Path(settings.ui_preferences_file)


def _load(settings: Settings) -> dict[str, Any]:
    p = _path(settings)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("ui_prefs.load_failed", error=str(exc))
        return {}


def _save(settings: Settings, data: dict[str, Any]) -> None:
    p = _path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def read_preferences(settings: Settings) -> dict[str, Any]:
    """Return saved UI defaults (brain + coding backend)."""
    data = _load(settings)
    brain = None
    if data.get("llm_provider"):
        brain = {
            "provider": data["llm_provider"],
            "model": data.get("llm_model"),
        }
    use_mem = data.get("use_stored_memory")
    parallel = data.get("parallel_steps")
    return {
        "brain": brain,
        "coding_backend": data.get("coding_backend"),
        "use_stored_memory": use_mem if isinstance(use_mem, bool) else None,
        "parallel_steps": parallel if isinstance(parallel, bool) else None,
        "max_parallel_steps": data.get("max_parallel_steps"),
    }


def apply_ui_preferences(settings: Settings) -> bool:
    """Overlay saved UI defaults onto live settings. Returns True if applied."""
    data = _load(settings)
    if not data:
        return False
    applied = False
    provider = data.get("llm_provider")
    if provider:
        try:
            settings.llm_provider = LLMProvider(provider)
            applied = True
        except ValueError:
            log.warning("ui_prefs.invalid_provider", provider=provider)
    model = data.get("llm_model")
    if model and provider and data.get("llm_model_provider") == provider:
        attr = _CHAT_MODEL_ATTR.get(provider)
        if attr:
            setattr(settings, attr, model)
            applied = True
    backend = data.get("coding_backend")
    if backend:
        settings.coding_backend = str(backend)
        applied = True
    use_mem = data.get("use_stored_memory")
    if isinstance(use_mem, bool):
        settings.use_stored_memory = use_mem
        applied = True
    parallel = data.get("parallel_steps")
    if isinstance(parallel, bool):
        settings.parallel_steps = parallel
        applied = True
    max_par = data.get("max_parallel_steps")
    if isinstance(max_par, int) and max_par > 0:
        settings.max_parallel_steps = max_par
        applied = True
    if applied:
        log.info(
            "ui_prefs.applied",
            provider=settings.llm_provider.value,
            coding_backend=settings.coding_backend,
            use_stored_memory=settings.use_stored_memory,
        )
    return applied


def save_brain_default(settings: Settings, provider: str, model: str | None) -> None:
    data = _load(settings)
    data["llm_provider"] = provider
    if model:
        data["llm_model"] = model
        data["llm_model_provider"] = provider
    else:
        data.pop("llm_model", None)
        data.pop("llm_model_provider", None)
    _save(settings, data)
    log.info("ui_prefs.brain_saved", provider=provider, model=model)


def save_coding_default(settings: Settings, backend: str) -> None:
    data = _load(settings)
    data["coding_backend"] = backend
    _save(settings, data)
    log.info("ui_prefs.coding_saved", backend=backend)


def save_parallel_default(
    settings: Settings, parallel_steps: bool, max_parallel_steps: int | None = None
) -> None:
    data = _load(settings)
    data["parallel_steps"] = parallel_steps
    if max_parallel_steps and max_parallel_steps > 0:
        data["max_parallel_steps"] = int(max_parallel_steps)
    _save(settings, data)
    log.info("ui_prefs.parallel_saved", parallel_steps=parallel_steps)


def save_memory_default(settings: Settings, use_stored_memory: bool) -> None:
    data = _load(settings)
    data["use_stored_memory"] = use_stored_memory
    _save(settings, data)
    log.info("ui_prefs.memory_saved", use_stored_memory=use_stored_memory)


__all__ = [
    "apply_ui_preferences",
    "read_preferences",
    "save_brain_default",
    "save_coding_default",
    "save_memory_default",
    "save_parallel_default",
]
