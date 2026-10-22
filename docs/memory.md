# Memory & RAG architecture

## Memory types (namespaces)

All memory is modeled as namespaces over one `VectorStore`, so a single API serves them:

| Namespace | Purpose |
|---|---|
| `short_term` | rolling conversation working set |
| `long_term` | durable semantic memory of outcomes/facts |
| `conversation` | full message history (per `session_id`) |
| `task` | goals, plans, results |
| `knowledge` | RAG-ingested document chunks |
| `project` | project-scoped context |
| `skill` | reusable workflows the employee learned |

`MemoryManager` (`memory/manager.py`) is the high-level API: `record`, `recall`,
`record_message`, `conversation_window`, `remember_skill`, `find_skill`. It embeds on
write (via the embedder `Brain`) and does cosine semantic search on read.

Learned skills can be reviewed and managed from the Web UI: click **💡 Skills** in the
top bar (or **Manage learned skills…** in the Memory menu) to list, create, edit, or
delete them. Saving re-embeds the corrected skill and persists it through the normal
memory backend, so future tasks recall the fixed version instead of a bad one.

**Task-specific guard** (`memory/skill_policy.py`). Skills must be general — scoped to
a language, framework/repo family, dataset type, or tooling pattern. A skill is
rejected on save, skipped on recall, and dropped on snapshot import when it contains a
benchmark instance id (`owner__repo-1234`), a 40-hex commit SHA, raw task-dump headers
(`## 1. Repository Information`), an issue/PR/ticket number, a one-bug-fix title, or a
conversational follow-up ("do you update the code"). Saved skills are capped at 150
words and merged by title. The same check (`task_specific_reason`) also guards
`long_term` `GOAL:` memories — `find_knowledge` never injects a stored raw benchmark
prompt, and the reflect node no longer stores one.
`MemoryManager.purge_task_specific_memories()` removes legacy offenders.

## Backends

- **In-memory** (`in_memory.py`) — cosine over numpy. Default offline; zero infra.
- **pgvector** (`pg_store.py`) — cosine distance over the `memories` table; IVFFlat index.
- **Resilient** (`resilient.py`) — wraps pgvector and transparently **falls back to
  in-memory** on the first backend error (logged once). Used in `MEMORY_BACKEND=auto`.

Select with `MEMORY_BACKEND`:

| value | behavior |
|---|---|
| `auto` *(default)* | pgvector when reachable, else in-memory; resilient at runtime |
| `memory` | always in-memory |
| `pgvector` | require Postgres; fail loud if unavailable (use in production) |

## RAG pipeline

`knowledge/rag.py::KnowledgeBase`:

1. **ingest(source)** — `loaders.load_source` handles paths, directories, URLs and many
   file types (PDF, DOCX, PPTX, XLSX, CSV, HTML, Markdown, code). Each document is
   chunked (`chunking.chunk_text`, overlapping windows on paragraph/sentence boundaries),
   embedded, and stored in the `knowledge` namespace with source metadata.
2. **retrieve(query, k)** — semantic recall from `knowledge`.
3. **answer(question, k)** — retrieves top-k chunks and instructs the brain to answer
   **only** from that context, returning the answer plus citations.

Because retrieval reuses the embedder `Brain`, RAG works with whichever embedding
provider is configured — including the offline mock embeddings.

## Persistence & resume

Tasks and memory are saved to disk so work survives restarts (`observability/store.py`):

```
data/sessions/
  <session_id>/
    session.json          # title, created time, run order
    runs/<task_id>.json   # goal + full agent/tool event history + result
  _memory.jsonl           # exported memory records (for resume context)
```

- On startup, `AIEmployee` reloads `_memory.jsonl` into the store, so a reopened task
  continues with its accumulated `long_term`/`skill` context (you'll see
  `memory.restored records=N` in the logs).
- pgvector is already durable in Postgres, so the file dump is only used for the
  in-memory / resilient backends.
- The Web UI's **Saved tasks** sidebar lists every persisted session to reopen or continue.
- Controlled by `PERSIST_SESSIONS` and `SESSIONS_DIR`.
