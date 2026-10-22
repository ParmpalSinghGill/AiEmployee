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


def parse_agent_limits(spec: str) -> dict[str, int]:
    """Parse ``"coding=16, research=12"`` → ``{"coding": 16, "research": 12}``.

    Accepts comma- or semicolon-separated ``name=value`` pairs. Names are
    normalized (lowercased, ``-`` → ``_``); malformed entries are ignored rather
    than raising, so a typo in ``.env`` can never break startup.
    """
    out: dict[str, int] = {}
    for chunk in (spec or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, raw = chunk.partition("=")
        name = name.strip().lower().replace("-", "_")
        if not name:
            continue
        try:
            out[name] = max(0, int(float(raw.strip())))
        except (TypeError, ValueError):
            continue
    return _fold_legacy_agent_names(out)


def _fold_legacy_agent_names(limits: dict[str, int]) -> dict[str, int]:
    """Apply limits written for retired agents (``devops=20``) to their successor.

    An explicit entry for the successor always wins; among several retired names
    that merged into one agent, the largest limit is kept.
    """
    from ai_employee.agents.routing import resolve_agent_name

    explicit = {k: v for k, v in limits.items() if resolve_agent_name(k) == k}
    folded = dict(explicit)
    for name, value in limits.items():
        target = resolve_agent_name(name)
        if target != name and target not in explicit:
            folded[target] = max(value, folded.get(target, value))
    return folded


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
    # Output ceiling per call. 2048 was far too low for this workload: agents write
    # training scripts and READMEs, and on reasoning models the chain-of-thought is
    # billed against the SAME budget — so long reasoning left nothing for the answer
    # and responses came back cut off mid-word (finish_reason="length").
    llm_max_tokens: int = 8192
    llm_request_timeout: int = 60
    llm_max_retries: int = 3
    # Rate limits (HTTP 429 / quota) get their own, longer backoff. A per-minute
    # quota usually clears within a minute; 6 attempts with a 60s cap means the
    # step waits instead of failing after a few seconds. Retry-After is honored
    # when the provider sends it.
    llm_rate_limit_retries: int = 6
    llm_rate_limit_base_wait: float = 5.0
    llm_rate_limit_max_wait: float = 60.0

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
    # Default is a current stable model. The UI picker (api/ui.py ``_SUGGESTED``)
    # lists the full current lineup — 2.5 GA, 3.x previews and the -latest aliases
    # — so any of them can be selected without editing config.
    gemini_chat_model: str = "gemini-2.5-pro"
    gemini_embedding_model: str = "gemini-embedding-001"

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
    # When True, bypass loading learned `SKILL:` memories from prior sessions during
    # agent recall — used for clean zero-shot evaluations where cross-session skill
    # leakage would pollute the result. Long-term recall is unaffected.
    disable_skill_recall: bool = False

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
    huggingface_token: str = ""  # HF_TOKEN — gated/private repos & faster downloads
    # GPU VRAM available locally (GB); 0 = unknown. Used by estimate_hardware to
    # decide whether a found model can run/train on this machine.
    local_gpu_vram_gb: float = 0.0
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # ── MCP connectors (Model Context Protocol servers) ──────────────────────
    # JSON store of enabled connectors + their cached tool lists (UI-managed).
    mcp_config_file: Path = Path("./data/mcp_servers.json")
    # Per-call timeout; also bounds connect/discovery (npx may download on first use).
    mcp_call_timeout: int = 120

    # ── Google Workspace (Gmail / Calendar with YOUR registered OAuth app) ───
    # Drop the client_secret.json you download from Google Cloud console here;
    # every app user can then click "Connect Google account" and approve on
    # Google's consent page — no keys or setup on their side.
    google_oauth_client_file: Path = Path("./data/google_oauth_client.json")
    # Per-account tokens granted by users (chmod 600, refreshed automatically).
    google_accounts_file: Path = Path("./data/google_accounts.json")
    # Scopes requested at consent (comma-separated). gmail.readonly/send are
    # "restricted" scopes — publishing beyond test users needs Google review.
    google_oauth_scopes: str = (
        "openid,https://www.googleapis.com/auth/userinfo.email,"
        "https://www.googleapis.com/auth/gmail.readonly,"
        "https://www.googleapis.com/auth/gmail.send,"
        "https://www.googleapis.com/auth/calendar"
    )

    # ── User accounts (optional — the console works without signing in) ──────
    # Signed-in users are stored here; guests ("Test1", "Test2", …) are never
    # written anywhere — their identity lives only in a signed browser cookie.
    users_file: Path = Path("./data/users.json")
    # HMAC secret for the session cookie (generated on first use, chmod 600).
    auth_secret_file: Path = Path("./data/auth_secret.key")
    # How long a login/guest cookie stays valid.
    auth_session_days: int = 30

    # ── Safety / autonomy ────────────────────────────────────────────────────
    require_approval_for: str = (
        "shell_write,email_send,git_push,db_write,deploy,resource_download,"
        "resource_acquisition,mcp_tool,calendar_write"
    )
    # Rule-based auto-approval (per key: mode/recipient patterns/daily quota),
    # managed from the console (🛡 Autonomy). Enforced in ToolRegistry.execute.
    approval_policies_file: Path = Path("./data/approval_policies.json")
    # Append-only trail of every sensitive-tool execution across all tasks.
    audit_log_file: Path = Path("./data/audit_log.jsonl")
    # Hard per-task token cap (0 = unlimited); enforced at the Brain choke point.
    # Overridable per task from the console.
    task_token_budget: int = 0

    # ── Push notifications (ntfy.sh — approvals reach your phone) ────────────
    # Pick a long random topic, subscribe to it in the ntfy app, set it here.
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"

    # ── Scheduled tasks & watchers (⏰ console tab) ───────────────────────────
    schedules_file: Path = Path("./data/schedules.json")
    # Scheduler tick (seconds); also the floor for watcher poll intervals.
    scheduler_tick_seconds: int = 30
    enable_code_execution: bool = True
    workspace_dir: Path = Path("./data/workspace")
    # YouTube downloads (yt-dlp). Age-restricted videos need logged-in cookies.
    # Example: YTDLP_COOKIES_FROM_BROWSER=chrome  (or firefox, chromium, brave, edge)
    ytdlp_cookies_from_browser: str = ""
    ytdlp_cookies_file: Path | None = None
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
    # A step that must change files (implement/fix/modify/…) but has made this many
    # tool calls without writing anything is told to apply its edit now. 0 = off.
    agent_write_nudge_after: int = 10
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

    # ── Loop check: extend-vs-stop when an agent hits its iteration cap ──────
    # When an agent exhausts max_iterations without answering, a loop check
    # inspects the recent transcript. If it is genuinely stuck (repeating calls
    # or making no progress) the step stops; if it is still advancing it gets
    # another max_iterations window instead of being cut off mid-task.
    agent_loop_check_enabled: bool = True
    # Extra max_iterations windows granted when the loop check says "no loop".
    # Bounds the worst case: total rounds <= max_iterations * (1 + this).
    agent_loop_extensions_max: int = 2
    # How many recent transcript entries the check inspects. 0 => the agent's own
    # max_iterations (e.g. 8 for testing, 12 for coding).
    agent_loop_check_window: int = 0
    # Let an ambiguous verdict be decided by a cheap LLM judge call. Skipped
    # automatically for the mock provider so offline runs stay deterministic.
    agent_loop_check_llm: bool = True
    # A tool call (or a repeated read of the same target) repeating this many
    # times inside the window counts as a definite loop signal. Kept at 4 so a
    # repeat only counts once it exceeds 3 (2–3 repeats are normal working rhythm).
    agent_loop_repeat_threshold: int = 4
    # A short A,B(,C) cycle must repeat at least this many times to count as a
    # loop — normal edit→test→edit→test repeats twice and must not be flagged.
    agent_loop_cycle_min_repeats: int = 4

    # ── Per-agent iteration & loop budgets (editable in .env) ────────────────
    # Base tool rounds an agent gets before its loop check runs. 0 => use each
    # agent's built-in "max_iterations" (set per agent in agents/specs.py, 7–12).
    agent_max_iterations: int = 0
    # Per-agent iteration overrides — name=value, comma-separated. These WIN over
    # both the global value above and the agent's built-in default.
    #   AGENT_MAX_ITERATIONS_BY_NAME=coding=24,research=16,planner=4
    agent_max_iterations_by_name: str = ""
    # Per-agent loop overrides — how many extra iteration windows the loop check
    # may grant when it judges the agent "progressing". Falls back to
    # agent_loop_extensions_max. Total rounds = iterations × (1 + loops).
    #   AGENT_LOOP_EXTENSIONS_BY_NAME=coding=4,research=2
    agent_loop_extensions_by_name: str = ""

    # Run plan steps that don't depend on each other at the same time. Off by
    # default: sequential execution is easier to follow in the live flow view, and
    # concurrent agents multiply token spend and provider rate-limit pressure.
    # Toggled per task from the web console (⚡ Parallel).
    parallel_steps: bool = False
    # Ceiling on steps in flight at once when parallel_steps is on. Keeps a wide
    # plan from opening dozens of simultaneous provider connections.
    max_parallel_steps: int = 4

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
    # Per-string cap (chars) for the Explain trace: LLM messages, the step
    # task/system prompt, and agent answers. 0 = NO truncation (show everything).
    explain_max_chars: int = 0
    # Per-string cap (chars) for tool arguments and tool output in the trace.
    # 0 = NO truncation. Raise/tune from the 🎚 Limits modal or .env.
    explain_tool_max_chars: int = 0

    # ── Persistence (task history + memory, for resume) ──────────────────────
    persist_sessions: bool = True
    sessions_dir: Path = Path("./data/sessions")
    # Persist guest (no-account) sessions too. Default ON so a restart never loses
    # the runs you just did; set false to keep guests strictly memory-only.
    persist_guest_sessions: bool = True
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
    def agent_iteration_overrides(self) -> dict[str, int]:
        """Per-agent base-iteration overrides parsed from the env string."""
        return parse_agent_limits(self.agent_max_iterations_by_name)

    @property
    def agent_loop_overrides(self) -> dict[str, int]:
        """Per-agent loop-extension overrides parsed from the env string."""
        return parse_agent_limits(self.agent_loop_extensions_by_name)

    def iteration_limit_for(self, agent: str, default: int) -> int:
        """Base tool rounds for ``agent`` — per-agent override > global > built-in.

        ``AGENT_MAX_ITERATIONS_BY_NAME`` wins, then ``AGENT_MAX_ITERATIONS``, then
        the agent's own ``max_iterations`` (``default``).
        """
        key = (agent or "").strip().lower().replace("-", "_")
        overrides = self.agent_iteration_overrides
        if overrides.get(key, 0) > 0:
            return overrides[key]
        if self.agent_max_iterations > 0:
            return self.agent_max_iterations
        return max(1, int(default))

    def loop_limit_for(self, agent: str) -> int:
        """Extra loop windows for ``agent`` — per-agent override > global."""
        key = (agent or "").strip().lower().replace("-", "_")
        overrides = self.agent_loop_overrides
        if key in overrides:
            return overrides[key]
        return max(0, int(self.agent_loop_extensions_max))

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
