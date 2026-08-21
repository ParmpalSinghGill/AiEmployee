# AI Employee Framework

> Not a chatbot — an AI *employee*. Give it one goal; it plans the work, delegates to
> specialized agents, uses real tools, verifies its own output, and remembers what it did.

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-266%20offline-brightgreen.svg">
  <img alt="Providers" src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20GLM%20%7C%20local-orange.svg">
</p>

Most AI tools give you a chat box: you ask, it answers, and you still do the work. This
framework takes a **high-level goal** and runs it to completion — decomposing it into
steps, assigning each to one of **29 specialized agents**, calling **24 sandboxed tools**,
self-reviewing every step, and persisting the whole run so it survives a restart.

The LLM provider is swappable with a **single environment variable** — business logic
never imports an LLM SDK. It runs **fully offline** out of the box (mock brain +
in-memory store), so you can explore the entire system with zero API keys or
infrastructure.

```bash
LLM_PROVIDER=mock python -m ai_employee run "Research async Python and write a brief"
```

### What a real run looks like

Given one sentence:

> *"Find a pretrained model and a dataset for detecting defects on a factory production
> line, tell me the GPU I need to fine-tune it, and write runnable training code with a
> README."*

the employee plans five steps, runs the independent ones **in parallel**, searches
Hugging Face / Kaggle / Roboflow / arXiv for real, estimates VRAM for the chosen model,
writes a training script into the workspace, and documents how to run it — pausing for
your approval before it downloads anything large.

> **New here?** Read **[docs/setup.md](docs/setup.md)** to install & configure, then
> **[docs/agent-flow.md](docs/agent-flow.md)** to understand how a goal becomes a result.

---

## Highlights

| Capability | What you get |
|---|---|
| **Provider independence** | DeepSeek (default), OpenAI, Gemini, GLM, and a `mock` provider — all behind one `Brain` interface. Swap via `LLM_PROVIDER`. |
| **Multi-agent** | **29 specialized agents** (CEO, Planner, Scout, Coding, Security, QA, DevOps, …) orchestrated with **LangGraph**. |
| **Parallel execution** | Independent plan steps run **concurrently**; dependent steps wait and start the instant they're unblocked. One toggle in the console. |
| **Hierarchical planning** | Goal → decomposition → agent assignment → execution → verification → reflection → memory. |
| **Tools** | **24 sandboxed tools**: filesystem, Python/shell execution, HTTP, web fetch/search, CSV/JSON, hub search (HF/Kaggle/Roboflow/arXiv/GitHub) — with a human-approval gate for sensitive ops. |
| **Asset reuse** | Downloaded models/datasets are indexed workspace-wide and **never fetched twice** — enforced in the download tool, not just suggested to the model. |
| **Memory** | Short-term, long-term, conversation, task, knowledge, project, skill — over pgvector or in-memory. |
| **RAG** | Ingest PDF, Markdown, Word, PowerPoint, CSV, Excel, web pages, codebases; chunk, embed, retrieve, ground. |
| **Verification** | Every step is self-reviewed; failures retry within a budget; confidence is estimated. |
| **Learning** | Successful multi-step solutions are distilled into reusable *skills* and recalled later. |
| **Explainability** | An **Explain** tab per run: what each agent was asked, what the model chose, every tool call and the verifier's verdict. |
| **Web Console** | A dependency-free dashboard at `/ui`: a tab per task, live n8n-style agent/tool flow, logs, results, and follow-up questions. |
| **Resumable** | Every run is checkpointed mid-flight. Stop a task — or kill the server — and continue from the last completed step. |
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

### The team (29 agents)

<details>
<summary><b>Every agent, grouped by what it does</b> — <code>python -m ai_employee agents</code></summary>

| Group | Agents |
|---|---|
| **Orchestration** | `ceo` · `planner` · `task_manager` · `plan_researcher` · `human_approval` |
| **Research & acquisition** | `research` · `browser` · `scout` · `hardware_analyst` · `resource_acquirer` · `resource_downloader` |
| **Engineering** | `coding` · `codesmith` · `reviewer` · `testing` · `qa` · `security` · `git` |
| **Data & knowledge** | `data_analysis` · `database` · `rag` · `knowledge_manager` · `documentation` |
| **Infrastructure** | `devops` · `monitoring` |
| **Memory & learning** | `memory_manager` · `learning` |
| **Productivity** | `email` · `calendar` |

Most are declarative `AgentSpec`s in `agents/specs.py` — adding one is a few lines, no
new class. See [`docs/extending.md`](docs/extending.md).

</details>

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
python -m ai_employee serve        # UI: http://localhost:8010/ui  ·  API docs: /docs
```

Open **http://localhost:8010/ui** for the dashboard:

- **Switch the brain live** — the top-right badge shows the active provider + model;
  click it to change provider/model (lists local Ollama models) and apply without a restart.
- **One tab per task** — run many tasks concurrently, switch between them.
- **n8n-style live flow** — watch the plan, each step's agent, the tools it calls
  (with status + latency), and the verification badge update in real time.
- **⚡ Parallel toggle** — run independent plan steps at the same time; a banner names
  the agents working concurrently and each live step pulses in the flow.
- **Explain tab** — per step: the system prompt, the task input, every LLM round-trip
  (messages in / response out), each tool call, and the verifier's verdict.
- **Logs tab** — the full structured event stream for the task.
- **Result tab** — the final answer, confidence, per-step outputs, charts rendered
  inline, and every file the run produced.
- **Follow-ups** — ask a follow-up question on any task; it reuses the task's session
  (so conversation + memory carry over) and appears as a new run inside the tab.
- Tasks paused for **human approval** show an *Approve & continue* button.
- **Stopped runs stay resumable** — stop a task (or lose the server) and it reports how
  many steps were saved, with a *Resume from last checkpoint* button.
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

## Design principles

These are enforced, not aspirational — the test suite fails if they're broken.

1. **Business logic never imports an LLM SDK.** Everything goes through the `Brain`
   abstraction. Providers live in `brains/providers/`; adding one is a subclass plus a
   branch in `brains/factory.py`. This is what makes the provider a config value.
2. **The provider is chosen by config, not code.** `LLM_PROVIDER` is the single switch.
3. **It must run offline.** `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=mock`,
   `MEMORY_BACKEND=memory` keeps the full pipeline and every test green with no network
   and no database.
4. **Async-first.** Public APIs are `async`; nothing blocks the event loop.
5. **Sensitive operations are approval-gated.** Downloads, shell writes, `git push`,
   deploys and DB writes stop and wait for a human (`REQUIRE_APPROVAL_FOR`).
6. **Inputs are not outputs.** Downloaded datasets and model weights are indexed as
   *inputs*; the Result view shows what the run *produced* — charts, reports, code.

---

## Development

```bash
make test        # offline suite (mock provider, in-memory store)
make lint        # ruff
make format      # black + ruff --fix
make typecheck   # mypy
```

All **266 tests** run offline in seconds — no network, no database, no API keys.

---

## Documentation

| Guide | What's in it |
|---|---|
| [docs/setup.md](docs/setup.md) | Install, configure, first run |
| [docs/operator-sop.html](docs/operator-sop.html) | Operator SOP for the web console — for non-technical users (open in a browser) |
| [docs/agent-flow.md](docs/agent-flow.md) | How a goal becomes a result, step by step |
| [docs/architecture.md](docs/architecture.md) | Full design; [diagrams](docs/architecture-diagrams.md) |
| [docs/providers.md](docs/providers.md) | Switching LLMs, running local models |
| [docs/extending.md](docs/extending.md) | Add a tool, an agent, or a provider |
| [docs/memory.md](docs/memory.md) | Memory types and backends |
| [docs/deployment.md](docs/deployment.md) | Docker, Postgres/pgvector, production notes |

---

## Contributing

Issues and PRs are welcome — especially new tools, agents and providers, which are
designed to be cheap to add.

```bash
make dev                  # install dev deps + editable install
make lint && make test    # required before opening a PR
```

Please keep the offline suite green: if a change needs the network or a database to
pass, it needs a mock.

---

## License

[Apache-2.0](LICENSE) — free for commercial use, modification and distribution.
