# Architecture

The framework follows **clean architecture**: dependencies point inward, the LLM is an
implementation detail behind an interface, and every layer is independently testable.

## Layer map

```
api/  ──▶  core/ (AIEmployee) ──▶  workflows/ (Orchestrator, LangGraph)
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
   planning/        agents/          tools/           memory/        knowledge/
        │               │               │               │               │
        └──────────────▶ brains/ (Brain abstraction) ◀──┴───────────────┘
                                        │
                                  config/  core/(logging, errors)
```

Nothing below `brains/` knows which vendor is in use.

## The Brain abstraction

`brains/base.py` defines `Brain` with a uniform async surface:

- `chat(messages, tools=…, reasoning=…) -> ChatResult`
- `stream(messages) -> AsyncIterator[str]`
- `structured_output(messages, schema) -> BaseModel`
- `embed(texts) -> list[list[float]]`
- capability flags: `supports_tools`, `supports_streaming`,
  `supports_structured_output`, `supports_embeddings`, `supports_reasoning`

Providers:

- `providers/mock.py` — deterministic, offline; powers tests and key-less exploration.
- `providers/openai_compatible.py` — one robust implementation over the OpenAI Chat
  Completions dialect (tenacity retries, tool calls, JSON mode, embeddings).
- `providers/concrete.py` — thin subclasses for **DeepSeek / OpenAI / GLM / Gemini**
  (Gemini via its OpenAI-compatible endpoint).

`brains/factory.py` is the *only* place mapping `LLM_PROVIDER` → class. `get_embedder`
reuses the chat provider if it embeds, else falls back to `EMBEDDING_PROVIDER` (this is
how DeepSeek-as-chat still gets RAG).

## Orchestration (LangGraph)

`workflows/employee_graph.py` compiles a `StateGraph`:

```
plan ──▶ execute ──(more steps?)──▶ execute
            │                          │
            │ (approval needed)        ▼ (done)
            ▼                      synthesize ──▶ reflect ──▶ END
           END
```

- **plan** — `PlannerAgent` decomposes the goal into agent-assigned steps (structured
  output → validated `Plan`).
- **execute** — runs the step's agent, then the `Verifier` reviews the output; failed
  steps retry within a bounded budget. If a tool needs approval, the run halts and
  surfaces the request.
- **synthesize** — the `CEOAgent` integrates all step outputs into the final answer.
- **reflect** — persists the outcome to long-term memory and distills a reusable *skill*.

State is a `TypedDict` with reducer-annotated fields so LangGraph merges step results and
traces correctly across the loop.

## Agents

`agents/base.py` implements `ToolUsingAgent` — a ReAct loop: the brain proposes tool
calls, the registry executes them (enforcing approval), observations feed back, and the
loop ends on a final answer or the iteration cap. Confidence is estimated from the tool
trace and answer.

Most agents are **data** (`agents/specs.py`): role, system prompt, allowed tools.
`PlannerAgent`, `CEOAgent`, and `HumanApprovalAgent` have bespoke logic in
`agents/custom.py`.

## Tools

`tools/base.Tool` pairs a Pydantic `Args` model (which *is* the JSON schema advertised to
the LLM) with an async `_run`. The `ToolRegistry` exposes specs to the brain and routes
execution through the approval policy. Built-ins are sandboxed to `WORKSPACE_DIR`.

## Memory

See [`memory.md`](memory.md). Seven namespaces over a pluggable `VectorStore`
(in-memory or pgvector), with a resilient wrapper for graceful degradation.

## Verification

`verification/verifier.py` runs an independent, skeptical self-review against each step's
success criteria and returns a `Verdict {passed, confidence, issues, suggestion}` that
drives retries.
```
