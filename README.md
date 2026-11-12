# AI Employee Framework

> Not a chatbot — an AI *employee*. Give it one goal; it plans the work, delegates to
> specialized agents, uses real tools, verifies its own output, and remembers what it did.

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-708%20offline-brightgreen.svg">
  <img alt="Providers" src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20GLM%20%7C%20local-orange.svg">
</p>

Most AI tools give you a chat box: you ask, it answers, and you still do the work. This
framework takes a **high-level goal** and runs it to completion — decomposing it into
steps executed by the **right specialist agent** for each step (research for finding
things, coding for code, data_analysis for data, …), calling **27 sandboxed tools**
(34 with Google Workspace connected) plus any **MCP connectors** you enable,
self-reviewing every step, and persisting the whole run so it survives a restart. A team
of **8 specialist agents** — each owning one kind of work — is assigned step by step by
the planner, and a CEO agent combines their verified results into the final answer.

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
| **Provider independence** | DeepSeek (default), OpenAI, Gemini, GLM, OpenCode, any local OpenAI-compatible server (Ollama, vLLM, LM Studio), and a `mock` provider — all behind one `Brain` interface. Swap via `LLM_PROVIDER`. |
| **Multi-agent** | **8 specialist workers** (research, coding, testing, git, data_analysis, documentation, reviewer, comms) plus CEO / Planner / human-approval roles, orchestrated with **LangGraph**. Each kind of work has exactly one owner, so the planner rarely misroutes: research steps run the research agent, coding steps run the coding agent. |
| **Parallel execution** | Independent plan steps run **concurrently**; dependent steps wait and start the instant they're unblocked. One toggle in the console. |
| **Hierarchical planning** | Goal → decomposition → specialist execution → verification → reflection → memory. |
| **Tools** | **27 sandboxed tools** (34 with Google Workspace): filesystem, Python/shell execution, HTTP, web fetch/search, CSV/JSON, hub search (HF/Kaggle/Roboflow/arXiv/GitHub) — with a human-approval gate for sensitive ops. |
| **MCP connectors** | **25 presets** in the 🔌 MCP panel — GitHub, Git, Slack, Postgres, SQLite, Brave Search, Fetch, Playwright, Context7, Google Maps, Notion, Linear, Sentry, Jira/Confluence, Asana, Zapier, or any custom stdio/HTTP server. Remote servers connect with **one-click OAuth sign-in** (no keys); each connector's `mcp_*` tools are routed only to the agents that need them, with an optional approval gate. |
| **Google Workspace** | Connect a Google account once: Gmail, Calendar and Drive tools for the `comms` and `research` agents. Sending mail or creating events is approval-gated. |
| **Schedules & watchers** | ⏰ Run a task on a schedule, or watch a Gmail inbox and trigger a task when a matching mail arrives. |
| **Autonomy & safety** | 🛡 Rule-based auto-approval policies, an audit trail of every outward action, push notifications (ntfy), and per-task token budgets. |
| **User accounts** | 👤 Guest by default (no sign-up), or Google sign-in for a personal account; saved tasks are private to their owner. |
| **Org chart** | 🗺 A live team view with the CEO at the top — arrows appear only for real dispatches, hand-offs and reports as the run happens. |
| **Asset reuse** | Downloaded models/datasets are indexed workspace-wide and **never fetched twice** — enforced in the download tool, not just suggested to the model. |
| **Memory** | Short-term, long-term, conversation, task, knowledge, project, skill — over pgvector or in-memory. |
| **RAG** | Ingest PDF, Markdown, Word, PowerPoint, CSV, Excel, web pages, codebases; chunk, embed, retrieve, ground. |
| **Verification** | Every step is reviewed against evidence — files on disk, real downloads, git diffs, actual tool calls — not just the agent's claims. Failures retry once, continuing from the previous attempt; confidence is estimated. |
| **Learning** | Successful multi-step solutions are distilled into reusable *skills* and recalled later. Review, edit or delete them in the 💡 Skills panel. |
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
   │(swap-  │  │(8 work- │ │(registry│ │(pgvector│  │  (RAG)   │
   │ pable) │  │ers+CEO) │ │+approval│ │/in-mem) │  │          │
   └────────┘  └─────────┘ └────────┘ └─────────┘  └──────────┘
        │                                  │
        └────── Verification ──────────────┘   (self-review, retry, confidence)
```

### The team (8 workers + 3 orchestration roles)

<details>
<summary><b>Every agent, and the one kind of work it owns</b> — <code>python -m ai_employee agents</code></summary>

| Agent | Owns |
|---|---|
| `research` | External information: web research, ML models/datasets/papers on public hubs, downloads, GPU/VRAM estimates |
| `coding` | Local code: app code, bug fixes, Docker/CI/deploy config, SQL/schemas, ML scripts (no web access) |
| `testing` | Writing/running tests and pass/fail validation against requirements |
| `git` | Clone, checkout, status/diff, commit |
| `data_analysis` | Fetching public data by code, EDA, KPIs, charts |
| `documentation` | READMEs, reports, written explanations |
| `reviewer` | Read-only code, security and logs/health review |
| `comms` | Gmail + Google Calendar (offered to the planner only when Google is connected) |
| *orchestration* | `planner` · `ceo` · `human_approval` (never assigned a step) |

Retired names from the older 26-agent roster (`scout`, `codesmith`, `devops`, `qa`, …)
still resolve to their successors via `LEGACY_AGENT_ALIASES` in `agents/routing.py`.

**How a run is overseen:** after every step the **Verifier** checks the output against
evidence and sends a failed step back for one retry. The **CEO** runs once at the end: it
merges all step results — each marked completed, unverified or skipped — into the final
answer, without inventing anything a skipped step didn't produce.

The workers are declarative `AgentSpec`s in `agents/specs.py` — adding one is a few
lines, no new class. See [`docs/extending.md`](docs/extending.md).

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

### 6. Logging & monitoring (OpenTelemetry · Prometheus · Loki · Tempo · Grafana)

`./start.sh` (same as `./start_server.sh`) starts five Docker containers, starts the
server with `OTEL_ENABLED=true`, and prints a link to each dashboard. The app sends
traces, metrics and logs over OpenTelemetry to one Collector, which forwards each
signal to its store. Grafana reads all three stores.

```
AI Employee ──OTLP/HTTP :4318──▶ OpenTelemetry Collector ─┬─ traces  ──▶ Tempo       :3200
                                                          ├─ metrics ──▶ Prometheus  :9090
                                                          └─ logs    ──▶ Loki        :3100
                                       Grafana :3000  ◀── reads all three, "AI Employee" dashboard
```

```bash
./start.sh                       # containers + server, then prints the links below
OBSERVABILITY=0 ./start.sh       # server only, no Docker stack
./stop_server.sh                 # stop the server (containers keep running)
./stop_server.sh --all           # also stop the containers (data is kept)

# Manage the stack directly
docker compose -f docker-compose.observability.yml up -d
docker compose -f docker-compose.observability.yml ps      # check the containers are running
docker compose -f docker-compose.observability.yml logs -f grafana
docker compose -f docker-compose.observability.yml down -v # remove containers and all stored data
```

**Containers** (all in the Docker project `aiemployee-observability`):

| Container | Image | Host port | What it does |
|---|---|---|---|
| `otel-collector` | `otel/opentelemetry-collector-contrib:0.116.1` | 4317 (gRPC), 4318 (HTTP) | Receives OTLP from the app and forwards each signal to its store |
| `prometheus` | `prom/prometheus:v3.1.0` | 9090 | Stores metrics; scrapes the Collector, and Tempo writes span metrics to it |
| `loki` | `grafana/loki:3.3.2` | 3100 | Stores logs (each log field is queryable) |
| `tempo` | `grafana/tempo:2.6.1` | 3200 | Stores traces; generates the service map |
| `grafana` | `grafana/grafana:11.4.0` | 3000 | Dashboards; no login needed locally (admin / admin if asked) |

**When it's running, check it with `docker compose -f docker-compose.observability.yml ps`:**

```
NAME                                        SERVICE          STATUS         PORTS
aiemployee-observability-grafana-1          grafana          Up             0.0.0.0:3000->3000/tcp
aiemployee-observability-loki-1             loki             Up             0.0.0.0:3100->3100/tcp
aiemployee-observability-otel-collector-1   otel-collector   Up             0.0.0.0:4317-4318->4317-4318/tcp
aiemployee-observability-prometheus-1       prometheus       Up             0.0.0.0:9090->9090/tcp
aiemployee-observability-tempo-1            tempo            Up             0.0.0.0:3200->3200/tcp
```

**Links** (`start.sh` prints these):

| What | Link |
|---|---|
| Grafana dashboard ("AI Employee") | http://localhost:3000/d/ai-employee/ai-employee |
| Logs: Grafana → Explore → Loki, query `{service_name="ai-employee"}` | http://localhost:3000/explore |
| Traces: Grafana → Explore → Tempo, TraceQL `{resource.service.name="ai-employee"}` | http://localhost:3000/explore |
| Prometheus (metrics, query `ai_employee_*`) | http://localhost:9090/graph |
| Prometheus scrape targets (all should be UP) | http://localhost:9090/targets |
| Loki / Tempo health | http://localhost:3100/ready · http://localhost:3200/ready |

**What's captured**

- **Traces.** Each task is one trace: the HTTP request, then `task.run`, then each plan
  `step`, then the `agent` turn, then every `llm.chat` / `llm.structured_output` call
  (model and token counts) and every `tool` call (status and error). Failed spans are
  marked red. Approval pauses and user stops are not counted as errors.
- **Metrics.** `ai_employee_tasks_total`, `_steps_total`, `_agent_runs_total`,
  `_llm_calls_total`, `_llm_tokens_total`, `_tool_calls_total` and
  `_http_requests_total`, each with a matching `_duration_seconds` histogram, labelled by
  status, agent, model, tool or route.
- **Logs.** Every structured log event at INFO or above, the same stream the console's
  Logs tab shows. Fields such as `task_id`, `agent`, `tool` and `step` become Loki
  fields, and each line carries its `trace_id`. In Grafana, a log line's **View trace**
  button opens its trace, and a span's **Logs for this span** button shows its logs.

The dashboard has panels for task outcomes, LLM latency and tokens per model, agent and
step activity, top and failing tools, HTTP latency, log volume, a filterable log viewer,
recent task traces, slow or failing spans, and the service map.

Telemetry is **off by default** (`OTEL_ENABLED=false`) so offline runs and the tests make
no network calls. To use it without `start.sh`, start the stack and set `OTEL_ENABLED=true`
(and `OTEL_EXPORTER_OTLP_ENDPOINT` if the Collector is not on `localhost:4318`). The
code is in `src/ai_employee/core/telemetry.py`. The stack's config files are in
`deploy/observability/`. You can change host ports with `GRAFANA_PORT`, `PROMETHEUS_PORT`,
`LOKI_PORT`, `TEMPO_PORT` and `OTLP_HTTP_PORT`.

---

## Switching providers

The provider is chosen entirely from configuration (`brains/factory.py`):

| `LLM_PROVIDER` | Chat | Reasoning | Embeddings |
|---|---|---|---|
| `mock` | ✓ (offline) | ✓ | ✓ (hash-based) |
| `deepseek` *(default)* | `deepseek-chat` | `deepseek-reasoner` | — (use `EMBEDDING_PROVIDER`) |
| `openai` | `gpt-4o` | `o3-mini` | `text-embedding-3-small` |
| `gemini` | `gemini-2.5-pro` | — | `gemini-embedding-001` |
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

All **708 tests** run offline in seconds — no network, no database, no API keys.

---

## Documentation

| Guide | What's in it |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | **The complete, code-verified explanation of the whole system** — every layer, all three orchestrators, the agent loop, tools, memory, RAG, verification, config, and where state lives |
| [docs/test-scenarios.md](docs/test-scenarios.md) | Realtime acceptance tests — 5 goals graded on agent routing and result quality (`python scripts/run_scenarios.py`) |
| [docs/resource-isolation.md](docs/resource-isolation.md) | Task ↔ source links, reference-counted deletion, and isolating tasks (shared → private) |
| [docs/setup.md](docs/setup.md) | Install, configure, first run |
| [docs/operator-sop.md](docs/operator-sop.md) | Operator SOP for the web console — for non-technical users ([printable HTML](docs/operator-sop.html)) |
| [docs/agent-flow.md](docs/agent-flow.md) | How a goal becomes a result, step by step |
| [docs/architecture.md](docs/architecture.md) | Full design; [diagrams](docs/architecture-diagrams.md) |
| [docs/providers.md](docs/providers.md) | Switching LLMs, running local models |
| [docs/extending.md](docs/extending.md) | Add a tool, an agent, or a provider |
| [docs/memory.md](docs/memory.md) | Memory types and backends |
| [docs/google-oauth-setup.md](docs/google-oauth-setup.md) | Set up your own Google OAuth client for Gmail / Calendar / Drive and Google sign-in |
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
