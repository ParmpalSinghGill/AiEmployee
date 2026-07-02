# CLAUDE.md

Guidance for Claude Code (and other AI agents) working in this repository.

## What this is

The **AI Employee Framework**: a provider-independent, multi-agent system
(LangChain + LangGraph + FastAPI + Postgres/pgvector + Redis). It accepts
high-level goals, plans, delegates to specialized agents, uses tools, verifies
output, and learns reusable skills.

## Golden rules

1. **Business logic NEVER imports an LLM SDK.** Always go through the `Brain`
   abstraction (`ai_employee.brains`). To add/inspect a provider, edit
   `brains/providers/` and `brains/factory.py` only.
2. **The provider is chosen by config**, not code. `LLM_PROVIDER` in `.env` is the
   single switch. Don't hard-code provider behavior elsewhere.
3. **It must run offline.** `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=mock`,
   `MEMORY_BACKEND=memory` must keep the full pipeline and tests green with no
   network or database. The test suite enforces this.
4. **Async-first.** Public APIs are `async`. Don't block the event loop.
5. **Sensitive ops are approval-gated.** Tools with an `approval_key` in
   `settings.approval_required` must not execute without `approved=True`.

## Environment

- Conda env: **`AIEmployee`** (Python 3.12). `conda activate AIEmployee`.
- Install dev deps: `make dev` (`pip install -r requirements-dev.txt && pip install -e .`).

## Common commands

```bash
make test         # pytest, offline (mock provider + in-memory memory)
make lint         # ruff check
make format       # black + ruff --fix
make typecheck    # mypy
python -m ai_employee run "<goal>"     # run the pipeline from the CLI
python -m ai_employee serve            # start the API
```

Tests force offline config via `tests/conftest.py` (sets `LLM_PROVIDER=mock` etc.).

## Where things live

| Concern | Location |
|---|---|
| Settings / provider switch | `config/settings.py`, `brains/factory.py` |
| LLM abstraction | `brains/base.py`, `brains/schemas.py` |
| Providers | `brains/providers/` (`mock`, `openai_compatible`, `concrete`) |
| Tools | `tools/base.py`, `tools/registry.py`, `tools/builtins/` |
| Agents | `agents/base.py`, `agents/specs.py` (declarative), `agents/custom.py` |
| Planner | `planning/planner.py` |
| Orchestration (LangGraph) | `workflows/employee_graph.py` |
| Memory | `memory/` (`in_memory`, `pg_store`, `resilient`, `manager`) |
| RAG | `knowledge/` (`loaders`, `chunking`, `rag`) |
| API | `api/` (`app`, `routes`, `schemas`, `deps`) |
| Web UI | `api/ui.py` + `api/static/` (SPA); task runner in `observability/tasks.py` |
| Persistence (save/resume) | `observability/store.py` (`SessionStore`); memory dump/restore in `core/employee.py` |
| DB models / migrations | `database/models.py`, `migrations/` |

## Extending

- **New tool:** subclass `tools.base.Tool`, define an `Args` Pydantic model + `_run`,
  register it in `tools/builtins/__init__.py`. Set `approval_key` if it mutates state.
- **New agent:** add an `AgentSpec` to `agents/specs.py` (or a custom class in
  `agents/custom.py` for bespoke logic).
- **New provider:** add a subclass in `brains/providers/` and a branch in
  `brains/factory.py._build`.

After any change: `make lint && make test`.

## Gotchas

- `database/models.py` reads `EMBEDDING_DIM` at import to size the `Vector` column —
  keep it consistent with your embedding model and migrations.
- `MEMORY_BACKEND=auto` prefers pgvector but **degrades to in-memory** on DB failure
  (see `memory/resilient.py`). Use `pgvector` explicitly in production to fail loud.
- The mock brain synthesizes JSON for structured output by reading the schema from the
  system prompt — keep `Brain.structured_output`'s prompt format in sync if you change it.
- **UI event capture is decoupled from console verbosity.** `core/logging.py` lets `INFO`+
  events always reach the processor chain (so `observability` observers capture them) and
  gates only *console printing* by `LOG_LEVEL`. Don't reintroduce level filtering at the
  `wrapper_class` — it would silently break the dashboard's live flow/logs. The task
  runner attributes events to a task via the `task_id` contextvar bound in
  `TaskManager._execute`.
