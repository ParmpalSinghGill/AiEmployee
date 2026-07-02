"""Thin provider subclasses.

Each just wires provider-specific endpoints/models into the shared
:class:`OpenAICompatibleBrain`. Adding a new OpenAI-compatible provider is a few
lines here plus a branch in the factory.
"""

from __future__ import annotations

from ai_employee.brains.providers.openai_compatible import OpenAICompatibleBrain
from ai_employee.config import Settings


class DeepSeekBrain(OpenAICompatibleBrain):
    """DeepSeek — default provider. Has a dedicated reasoning model; no embeddings."""

    def __init__(self, s: Settings) -> None:
        super().__init__(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            chat_model=s.deepseek_chat_model,
            reasoning_model=s.deepseek_reasoning_model,
            embedding_model=None,  # DeepSeek has no embeddings endpoint
            timeout=s.llm_request_timeout,
            max_retries=s.llm_max_retries,
            default_temperature=s.llm_temperature,
            default_max_tokens=s.llm_max_tokens,
            name="deepseek",
        )


class OpenAIBrain(OpenAICompatibleBrain):
    def __init__(self, s: Settings) -> None:
        super().__init__(
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
            chat_model=s.openai_chat_model,
            reasoning_model=s.openai_reasoning_model,
            embedding_model=s.openai_embedding_model,
            timeout=s.llm_request_timeout,
            max_retries=s.llm_max_retries,
            default_temperature=s.llm_temperature,
            default_max_tokens=s.llm_max_tokens,
            name="openai",
        )


class GLMBrain(OpenAICompatibleBrain):
    """GLM / Zhipu via its OpenAI-compatible v4 endpoint."""

    def __init__(self, s: Settings) -> None:
        super().__init__(
            api_key=s.glm_api_key,
            base_url=s.glm_base_url,
            chat_model=s.glm_chat_model,
            reasoning_model=None,
            embedding_model=s.glm_embedding_model,
            timeout=s.llm_request_timeout,
            max_retries=s.llm_max_retries,
            default_temperature=s.llm_temperature,
            default_max_tokens=s.llm_max_tokens,
            name="glm",
        )


class OpenCodeBrain(OpenAICompatibleBrain):
    """OpenCode Zen — a curated, OpenAI-compatible gateway over many hosted models.

    A single key (issued from an opencode.ai workspace) reaches Claude, GPT,
    DeepSeek, GLM, Gemini and more via ``/zen/v1/chat/completions``. No embeddings
    endpoint, so RAG falls back to the configured ``EMBEDDING_PROVIDER``.
    """

    def __init__(self, s: Settings) -> None:
        super().__init__(
            api_key=s.opencode_api_key,
            base_url=s.opencode_base_url,
            chat_model=s.opencode_chat_model,
            reasoning_model=s.opencode_reasoning_model or None,
            embedding_model=None,  # Zen exposes chat models only
            timeout=s.llm_request_timeout,
            max_retries=s.llm_max_retries,
            default_temperature=s.llm_temperature,
            default_max_tokens=s.llm_max_tokens,
            name="opencode",
        )


class LocalBrain(OpenAICompatibleBrain):
    """Any local OpenAI-compatible server (Ollama, vLLM, LM Studio, llama.cpp).

    Defaults target Ollama running Qwen3 8B. Because it speaks the same dialect,
    local models get the full feature set — chat, streaming, tool calling,
    structured output, embeddings, and reasoning ("thinking") — with no special
    cases anywhere else in the framework.
    """

    def __init__(self, s: Settings) -> None:
        super().__init__(
            api_key=s.local_api_key or "local",
            base_url=s.local_base_url,
            chat_model=s.local_chat_model,
            reasoning_model=s.local_reasoning_model or None,
            embedding_model=s.local_embedding_model or None,
            timeout=max(s.llm_request_timeout, 120),  # local generation can be slower
            max_retries=s.llm_max_retries,
            default_temperature=s.llm_temperature,
            default_max_tokens=s.llm_max_tokens,
            name="local",
            strip_think=s.local_strip_think,
            no_think=not s.local_thinking,
        )


class GeminiBrain(OpenAICompatibleBrain):
    """Google Gemini via its OpenAI-compatible endpoint.

    Keeps the whole framework on one chat dialect. (A native
    ``langchain-google-genai`` path can be slotted in here later if needed.)
    """

    OPENAI_COMPAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, s: Settings) -> None:
        super().__init__(
            api_key=s.google_api_key,
            base_url=self.OPENAI_COMPAT_URL,
            chat_model=s.gemini_chat_model,
            reasoning_model=None,
            embedding_model=s.gemini_embedding_model,
            timeout=s.llm_request_timeout,
            max_retries=s.llm_max_retries,
            default_temperature=s.llm_temperature,
            default_max_tokens=s.llm_max_tokens,
            name="gemini",
        )
