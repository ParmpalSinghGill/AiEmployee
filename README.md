# AI Employee Framework

A production-grade, **provider-independent** multi-agent framework that behaves like a
highly skilled employee. Give it a high-level goal; it **plans** the work, **delegates**
to specialized agents, **uses tools** autonomously, **verifies** its own output, **learns**
reusable skills, and **remembers** across sessions.

The LLM provider is swappable with a **single environment variable** — no business logic
changes. It runs **fully offline** out of the box (mock brain + in-memory store), so you
can explore the whole system with zero API keys or infrastructure.

```bash
LLM_PROVIDER=mock python -m ai_employee run "Research async Python and write a brief"
```

> **New here?** Read **[docs/setup.md](docs/setup.md)** to install & configure, then
> **[docs/agent-flow.md](docs/agent-flow.md)** to understand how a goal becomes a result.

---

## Highlights

| Capability | What you get |
|---|---|
| **Provider independence** | DeepSeek (default), OpenAI, Gemini, GLM, and a `mock` provider — all behind one `Brain` interface. Swap via `LLM_PROVIDER`. |
| **Multi-agent** | 23 specialized agents (CEO, Planner, Research, Coding, Security, QA, DevOps, …) orchestrated with **LangGraph**. |
| **Hierarchical planning** | Goal → decomposition → agent assignment → execution → verification → reflection → memory. |
| **Tools** | Sandboxed filesystem, Python/shell execution, HTTP, web fetch/search, CSV/JSON — with a human-approval gate for sensitive ops. |
| **Memory** | Short-term, long-term, conversation, task, knowledge, project, skill — over pgvector or in-memory. |
| **RAG** | Ingest PDF, Markdown, Word, PowerPoint, CSV, Excel, web pages, codebases; chunk, embed, retrieve, ground. |
| **Verification** | Every step is self-reviewed; failures retry within a budget; confidence is estimated. |
| **Learning** | Successful multi-step solutions are distilled into reusable *skills* and recalled later. |
| **Web Console** | A dependency-free dashboard at `/ui`: a tab per task, live n8n-style agent/tool flow, logs, results, and follow-up questions. |
| **API + CLI** | FastAPI (`/chat`, `/task`, `/agents`, `/tools`, `/knowledge`, `/memory`, …) and a rich CLI. |

---

## Architecture

```
                        ┌──────────────┐
   high-level goal ───▶ │  AIEmployee  │  (facade / DI container)
                        └──────┬───────┘
                               ▼
                     ┌───────────────────┐    LangGraph
                     │   Orchestrator    │  plan → execute* → synthesize → reflect
                     └─────────┬─────────┘
        ┌───────────┬──────────┼───────────┬────────────┐
        ▼           ▼          ▼           ▼            ▼
   ┌────────┐  ┌─────────┐ ┌────────┐ ┌─────────┐  ┌──────────┐
   │ Brains │  │ Agents  │ │ Tools  │ │ Memory  │  │Knowledge │
   │(swap-  │  │(23 spec-│ │(registry│ │(pgvector│  │  (RAG)   │
   │ pable) │  │ialized) │ │+approval│ │/in-mem) │  │          │
   └────────┘  └─────────┘ └────────┘ └─────────┘  └──────────┘
        │                                  │
        └────── Verification ──────────────┘   (self-review, retry, confidence)
```

Clean-architecture layers under `src/ai_employee/`:

```
config/        strongly-typed settings (provider switch lives here)
core/          logging, exceptions, the AIEmployee facade
brains/        the Brain abstraction + providers (business logic never imports an SDK)
tools/         tool base, registry, sandboxed built-ins, approval gating
memory/        vector stores (pgvector / in-memory), manager, resilient fallback
planning/      hierarchical planner (goal → steps)
agents/        base ReAct agent, declarative specs, custom agents
verification/  self-review / confidence
knowledge/     loaders, chunking, RAG
workflows/     LangGraph orchestration graph
database/      SQLAlchemy 2 async models + session
api/           FastAPI app, routes, schemas
```

See [`docs/architecture.md`](docs/architecture.md) for the full design.

---

## Quickstart

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate AIEmployee
pip install -e .
```

(Or `make env && conda activate AIEmployee && make dev`.)

### 2. Run offline (no keys needed)

```bash
cp .env.example .env          # defaults to DeepSeek; set LLM_PROVIDER=mock to stay offline
LLM_PROVIDER=mock python -m ai_employee run "Plan a product launch"
LLM_PROVIDER=mock python -m ai_employee agents
LLM_PROVIDER=mock python -m ai_employee ingest ./README.md
LLM_PROVIDER=mock python -m ai_employee query "What providers are supported?"
```

### 3. Use a real provider

Set the provider and its key in `.env`:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
EMBEDDING_PROVIDER=openai     # DeepSeek has no embeddings; RAG falls back to this
OPENAI_API_KEY=sk-...
```

Then the exact same commands now use the real model. **No code changes.**

### 4. Serve the API + Web Console

```bash
python -m ai_employee serve        # UI: http://localhost:8000/ui  ·  API docs: /docs
```

Open **http://localhost:8000/ui** for the dashboard:

- **Switch the brain live** — the top-right badge shows the active provider + model;
  click it to change provider/model (lists local Ollama models) and apply without a restart.
- **One tab per task** — run many tasks concurrently, switch between them.
- **n8n-style live flow** — watch the plan, each step's agent, the tools it calls
  (with status + latency), and the verification badge update in real time.
- **Logs tab** — the full structured event stream for the task.
- **Result tab** — the final answer, confidence, and per-step outputs.
- **Follow-ups** — ask a follow-up question on any task; it reuses the task's session
  (so conversation + memory carry over) and appears as a new run inside the tab.
- Tasks paused for **human approval** show an *Approve & continue* button.
- **Saved tasks (resume).** Every task — goal, full agent/tool event history, and the
  result — is persisted to `data/sessions/` and listed in the left **Saved tasks** panel.
  Reopen any past task to view its history or add a follow-up; memory is reloaded so the
  employee continues with full context, even across server restarts.
  Configure with `PERSIST_SESSIONS` / `SESSIONS_DIR`.

### 5. Full stack with Docker

```bash
docker compose up --build          # Postgres+pgvector, Redis, API, migrations
```

---

## Switching providers

The provider is chosen entirely from configuration (`brains/factory.py`):

| `LLM_PROVIDER` | Chat | Reasoning | Embeddings |
|---|---|---|---|
| `mock` | ✓ (offline) | ✓ | ✓ (hash-based) |
| `deepseek` *(default)* | `deepseek-chat` | `deepseek-reasoner` | — (use `EMBEDDING_PROVIDER`) |
| `openai` | `gpt-4o` | `o3-mini` | `text-embedding-3-small` |
| `gemini` | `gemini-1.5-pro` | — | `text-embedding-004` |
| `glm` | `glm-4-plus` | — | `embedding-3` |
| `local` | `qwen3:8b` (Ollama) | `qwen3:8b` | `Qwen3-Embedding-8B` |

Full guide: **[`docs/providers.md`](docs/providers.md)** — switch the brain, change just the
model, or run **local models** (Ollama / vLLM / LM Studio). Adding a *new* provider is a few
lines — see [`docs/extending.md`](docs/extending.md).

### Run locally with Qwen3 8B (Ollama)

```bash
ollama pull qwen3:8b
```
```dotenv
LLM_PROVIDER=local            # OpenAI-compatible; defaults to Ollama @ :11434
LOCAL_CHAT_MODEL=qwen3:8b
MEMORY_BACKEND=memory
```
```bash
python -m ai_employee run "Summarize the benefits of local LLMs"
```

---

## Development

```bash
make test        # offline suite (mock provider, in-memory store)
make lint        # ruff
make format      # black + ruff --fix
make typecheck   # mypy
```

All 34 tests run offline in well under a second.

## License

Apache-2.0.
