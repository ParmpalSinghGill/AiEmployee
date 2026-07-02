"""Central application settings.

Everything the framework needs is configured here and sourced from environment
variables / `.env`. Crucially, switching the LLM provider is a single env change
(`LLM_PROVIDER`) — no business logic is touched.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class LLMProvider(str, Enum):
    mock = "mock"
    deepseek = "deepseek"
    openai = "openai"
    gemini = "gemini"
    glm = "glm"
    opencode = "opencode"  # OpenCode Zen — OpenAI-compatible model gateway
    local = "local"  # any local OpenAI-compatible server (Ollama, vLLM, LM Studio, llama.cpp)


class EmbeddingProvider(str, Enum):
    mock = "mock"
    openai = "openai"
    gemini = "gemini"
    glm = "glm"
    local = "local"


class Settings(BaseSettings):
    """Strongly-typed, validated runtime configuration.

    Instantiate via :func:`get_settings` (cached) rather than constructing directly,
    so the whole process shares one consistent view of configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "AI Employee"
    environment: Environment = Environment.development
    log_level: str = "INFO"
    log_json: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── LLM selection ────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.deepseek
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_request_timeout: int = 60
    llm_max_retries: int = 3

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"
    deepseek_reasoning_model: str = "deepseek-reasoner"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o"
    openai_reasoning_model: str = "o3-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Gemini ───────────────────────────────────────────────────────────────
    google_api_key: str = ""
    gemini_chat_model: str = "gemini-1.5-pro"
    gemini_embedding_model: str = "text-embedding-004"

    # ── GLM / Zhipu ──────────────────────────────────────────────────────────
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_chat_model: str = "glm-4-plus"
    glm_embedding_model: str = "embedding-3"

    # ── OpenCode Zen ─────────────────────────────────────────────────────────
    # OpenCode's curated, OpenAI-compatible model gateway. One key unlocks many
    # hosted models (Claude, GPT, DeepSeek, GLM, Gemini, …). The workspace URL
    # (opencode.ai/workspace/wrk_…) identifies the billing account; the API key
    # is issued from that workspace's dashboard. No embeddings endpoint.
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/v1"
    opencode_chat_model: str = "deepseek-v4-pro"
    opencode_reasoning_model: str = ""  # blank → reuse chat model for reasoning
    opencode_workspace: str = ""  # informational: opencode.ai workspace id (wrk_…)

    # ── Local models (Ollama / vLLM / LM Studio / llama.cpp) ─────────────────
    # Any OpenAI-compatible local server. Defaults target Ollama with Qwen3 8B.
    local_base_url: str = "http://localhost:11434/v1"
    local_api_key: str = "ollama"  # most local servers ignore the key
    local_chat_model: str = "qwen3:8b"
    local_reasoning_model: str = "qwen3:8b"  # Qwen3 has a built-in "thinking" mode
    local_embedding_model: str = "dengcao/Qwen3-Embedding-8B:Q4_K_M"
    # Qwen3 emits <think>…</think> blocks; strip them from the visible answer.
    local_strip_think: bool = True
    # Qwen3's "thinking" makes every call much slower. Off by default so the
    # multi-step agent pipeline stays responsive; the explicit reasoning path
    # (Brain.chat(reasoning=True)) still enables it. Set true to always think.
    local_thinking: bool = False

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.openai
    embedding_dim: int = 1536

    # ── Memory backend ───────────────────────────────────────────────────────
    # auto: use pgvector when reachable, otherwise fall back to in-memory.
    # memory: always in-memory. pgvector: require Postgres (fail loud).
    memory_backend: str = "auto"
    # When False, agents skip long-term/skill recall and the reflect node does not
    # persist new memories (useful after bad recall or when you want a fresh run).
    use_stored_memory: bool = True

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ai_employee"
    postgres_password: str = "ai_employee"
    postgres_db: str = "ai_employee"
    database_url: str | None = None

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Tools / integrations ─────────────────────────────────────────────────
    github_token: str = ""
    tavily_api_key: str = ""
    kaggle_username: str = ""
    kaggle_key: str = ""
    roboflow_api_key: str = ""
    # GPU VRAM available locally (GB); 0 = unknown. Used by estimate_hardware to
    # decide whether a found model can run/train on this machine.
    local_gpu_vram_gb: float = 0.0
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # ── Safety / autonomy ────────────────────────────────────────────────────
    require_approval_for: str = (
        "shell_write,email_send,git_push,db_write,deploy,resource_download,resource_acquisition"
    )
    enable_code_execution: bool = True
    workspace_dir: Path = Path("./data/workspace")
    # JSON store for UI-managed API keys (overlaid on env at startup).
    secrets_file: Path = Path("./data/secrets.json")
    # UI defaults (brain provider/model, coding backend) set from the web console.
    ui_preferences_file: Path = Path("./data/ui_preferences.json")
    # Append-only token-usage ledger (per LLM call: provider, model, tokens, date).
    usage_ledger_file: Path = Path("./data/usage_ledger.jsonl")

    # ── Agent-loop tuning (hardening against the doom-loops the trace exposed) ─
    # Abort an agent's ReAct loop after a tool fails this many times with the
    # SAME error class (e.g. repeated IndentationError) instead of looping blindly.
    agent_repeat_failure_limit: int = 2
    # Shrink the content of older messages once history exceeds this many turns,
    # so per-call latency stops compounding as a long step accumulates context.
    agent_history_window: int = 14
    # Let agents self-provision Python packages via the (validated, non-shell)
    # pip_install tool, so CV/ML tasks don't loop on a missing dependency.
    allow_package_install: bool = True
    # Skip a step whose dependencies produced nothing usable (e.g. "analyse the
    # video" when the search step found no video) instead of hallucinating on it.
    skip_on_failed_dependency: bool = True
    # Don't re-run a step whose agent refused on policy grounds — the retry would
    # just reproduce the same refusal. Accept-with-note instead.
    no_retry_on_refusal: bool = True

    # ── External coding agents (delegate the coding step to a CLI) ────────────
    # brain  → use the configured LLM via the normal tool-using loop (default).
    # claude → Claude Code CLI; codex → Codex CLI; cursor → Cursor Agent CLI.
    # The chosen CLI writes AND runs the code in the workspace.
    coding_backend: str = "brain"  # brain | claude | codex | cursor
    claude_cli_cmd: str = "claude"
    codex_cli_cmd: str = "codex"
    cursor_cli_cmd: str = "cursor-agent"
    # Extra CLI args (space-separated); the task prompt is appended as the last arg.
    claude_cli_args: str = "-p --permission-mode acceptEdits"
    codex_cli_args: str = "exec --full-auto"
    cursor_cli_args: str = "-p --force"
    external_agent_timeout: int = 600

    # ── Explainability ───────────────────────────────────────────────────────
    # Capture the input/output of every LLM call (planner, agents, verifier,
    # synthesis) so the dashboard's "Explain" view can show what each component
    # received, reasoned, and produced. Adds verbose brain.* events; turn off to
    # quiet the event stream in production.
    explain_trace: bool = True

    # ── Persistence (task history + memory, for resume) ──────────────────────
    persist_sessions: bool = True
    sessions_dir: Path = Path("./data/sessions")
    # Generate a concise, well-formed task title with a small LLM call (vs. raw goal).
    auto_name_tasks: bool = True

    # ── Derived values ───────────────────────────────────────────────────────
    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        """Async SQLAlchemy URL, derived from parts unless explicitly overridden."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_sqlalchemy_url(self) -> str:
        """Sync URL (psycopg/asyncpg-free) for Alembic migrations."""
        return self.sqlalchemy_url.replace("+asyncpg", "")

    @property
    def approval_required(self) -> set[str]:
        return {p.strip() for p in self.require_approval_for.split(",") if p.strip()}

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.production


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read configuration (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()
