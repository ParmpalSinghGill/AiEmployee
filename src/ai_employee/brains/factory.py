"""Brain factory — turns configuration into a concrete :class:`Brain`.

This is the ONLY place that knows the mapping from provider name to class. Swapping
``LLM_PROVIDER`` in the environment changes the whole system's brain with zero code
changes anywhere else.
"""

from __future__ import annotations

from ai_employee.brains.base import Brain
from ai_employee.config import (
    EmbeddingProvider,
    LLMProvider,
    Settings,
    get_settings,
)
from ai_employee.core.exceptions import ConfigurationError
from ai_employee.core.logging import get_logger

log = get_logger("brain.factory")


def _build(provider: LLMProvider | EmbeddingProvider, s: Settings) -> Brain:
    name = provider.value
    if name == "mock":
        from ai_employee.brains.providers.mock import MockBrain

        return MockBrain(embedding_dim=s.embedding_dim)
    if name == "deepseek":
        from ai_employee.brains.providers.concrete import DeepSeekBrain

        return DeepSeekBrain(s)
    if name == "openai":
        from ai_employee.brains.providers.concrete import OpenAIBrain

        return OpenAIBrain(s)
    if name == "gemini":
        from ai_employee.brains.providers.concrete import GeminiBrain

        return GeminiBrain(s)
    if name == "glm":
        from ai_employee.brains.providers.concrete import GLMBrain

        return GLMBrain(s)
    if name == "opencode":
        from ai_employee.brains.providers.concrete import OpenCodeBrain

        return OpenCodeBrain(s)
    if name == "local":
        from ai_employee.brains.providers.concrete import LocalBrain

        return LocalBrain(s)
    raise ConfigurationError(f"Unknown provider: {name!r}")


# Process-wide singletons. Settings is process-wide, so a single cached brain is
# correct; tests/runtime mutate config then call reset_brain_cache().
_brain: Brain | None = None
_embedder: Brain | None = None


def get_brain(settings: Settings | None = None) -> Brain:
    """Return the configured chat brain (cached process-wide).

    When ``explain_trace`` is enabled the brain is wrapped in
    :class:`~ai_employee.brains.instrument.ExplainBrain` so every LLM call's input
    and output is captured for the dashboard's explainability view. The wrapper is
    transparent — capability flags and identity are mirrored.
    """
    global _brain
    if _brain is None:
        s = settings or get_settings()
        log.info("brain.select", provider=s.llm_provider.value)
        brain = _build(s.llm_provider, s)
        # Always meter token usage (cheap, transparent). Innermost so it sees every
        # call; the explain wrapper, if enabled, sits outside it.
        from ai_employee.brains.metering import MeteredBrain

        brain = MeteredBrain(brain)
        if s.explain_trace:
            from ai_employee.brains.instrument import ExplainBrain

            brain = ExplainBrain(brain)
        _brain = brain
    return _brain


def get_embedder(settings: Settings | None = None) -> Brain:
    """Return a brain capable of embeddings.

    If the chat provider supports embeddings it is reused; otherwise the dedicated
    ``EMBEDDING_PROVIDER`` is used (this is how DeepSeek-as-chat still gets RAG).
    """
    global _embedder
    if _embedder is None:
        s = settings or get_settings()
        chat = get_brain(s)
        if chat.supports_embeddings:
            _embedder = chat
        else:
            log.info("embedder.select", provider=s.embedding_provider.value)
            _embedder = _build(s.embedding_provider, s)
    return _embedder


def reset_brain_cache() -> None:
    """Drop cached brains (call after changing settings at runtime/tests)."""
    global _brain, _embedder
    _brain = None
    _embedder = None
