# Agent flow — how a goal becomes a result

This is the end-to-end path a high-level goal takes through the framework. The whole
pipeline is a **LangGraph** state machine (`src/ai_employee/workflows/employee_graph.py`)
driven by the provider-independent `Brain`.

```
                 ┌──────────────────────────── AIEmployee.run(goal) ───────────────────────────┐
                 │                          (src/ai_employee/core/employee.py)                  │
                 ▼
        ┌──────────────┐      ┌─────────────────────────────┐      ┌──────────────┐   ┌──────────┐
  goal ─►    PLAN      ├─────►│           EXECUTE           ├─────►│  SYNTHESIZE  ├──►│ REFLECT  ├──► answer
        │ (Planner)    │      │   loop over plan steps      │      │   (CEO)      │   │ (Memory) │
        └──────────────┘      └─────────────────────────────┘      └──────────────┘   └──────────┘
              │                          │  ▲                              │                │
              │ structured_output        │  │ verify + bounded retry       │ integrate      │ store outcome
              ▼                          ▼  │                              ▼ step outputs    ▼ + learn skill
         Plan{steps:[{id, desc,     pick agent → ReAct loop          final deliverable   long_term + skill
         agent, depends_on,         (tools, observations)                                   memory
         success_criteria}]}              │
                                          ▼
                                   ToolRegistry.execute  ── approval gate ──► (human approval if sensitive)
```

## Step by step

### 0. Entry — `AIEmployee.run(goal)`
The facade wires the brain, tools, agents, memory and the orchestrator, then invokes the
compiled LangGraph graph with the goal and a `session_id`.

### 1. PLAN — the Planner agent
`planning/planner.py` asks the `Brain` for **structured output** (validated Pydantic),
decomposing the goal into ordered steps:

```json
{ "goal": "...", "steps": [
  { "id": 1, "description": "...", "agent": "research", "depends_on": [],
    "success_criteria": "..." }, ... ] }
```
Each step is assigned to the best-fit agent from the roster (**23 agents total** — 20
declarative ones in `agents/specs.py` + 3 bespoke classes in `agents/custom.py`; see
[The agent roster](#the-agent-roster) below). Invalid assignments are repaired.

### 2. EXECUTE — one step at a time (looped)
For each step the orchestrator runs the assigned agent. Most agents are a
**ReAct-style `ToolUsingAgent`** (`agents/base.py`):

1. **Recall context** — pull relevant `long_term` + `skill` memories into the prompt.
2. **Reason** — `Brain.chat(messages, tools=…)`; the model either answers or requests tools.
3. **Act** — each requested tool runs via `ToolRegistry.execute` (sandboxed; see below).
4. **Observe** — tool results are appended as messages; loop back to step 2.
5. **Finish** — when the model answers (or hits `max_iterations`), return the result.

**Verification** (`verification/verifier.py`): an independent, skeptical review of the
step's output against its `success_criteria` → `Verdict{passed, confidence, issues}`.
If it fails, the step **retries** (bounded) with the reviewer's feedback.

**Tools & approval** (`tools/`): every tool declares a Pydantic args schema (advertised
to the model as a JSON-schema tool spec). Tools are sandboxed to `WORKSPACE_DIR`.
Sensitive tools (write/shell/email/git-push/deploy) are **approval-gated** — they raise
`ApprovalRequired`, the run pauses, and the UI shows an *Approve & continue* button.

### 3. SYNTHESIZE — the CEO agent
`agents/custom.py::CEOAgent` integrates all step outputs into one coherent final answer,
resolving contradictions.

### 4. REFLECT — learn for next time
The outcome is written to `long_term` memory, and multi-step solutions are distilled into
a reusable **skill** (so similar future goals are faster/better). See `docs/memory.md`.

### 5. Result
`run()` returns `{ answer, confidence, plan, steps, needs_approval, trace }`.

---

## Who does what

| Layer | Responsibility | Code |
|---|---|---|
| **Brain** | provider-independent LLM (chat, tools, structured output, embeddings) | `brains/` |
| **Planner** | goal → ordered, agent-assigned steps | `planning/planner.py` |
| **Agents** | execute a step via reason→tool→observe loop | `agents/` |
| **Tools** | sandboxed capabilities (fs, code, web, data) w/ approval gate | `tools/` |
| **Verifier** | self-review each step, drive retries, estimate confidence | `verification/` |
| **Memory** | recall context, store outcomes, learn skills | `memory/` |
| **Orchestrator** | the LangGraph state machine tying it together | `workflows/` |
| **Observability** | capture every event per task for the UI | `observability/` |

## The agent roster

**23 agents** are registered (`agents/registry.py::build_agent_registry`). The Planner
routes each step to one of them by `name`. The live list is also available at runtime via
`GET /agents` (and `emp.agents.catalog()`).

**20 declarative agents** — pure `AgentSpec` data in `agents/specs.py` (role, system
prompt, allowed tools). Grouped by `category`:

| Category | Agent (`name`) | Role | Tools |
|---|---|---|---|
| knowledge | `research` | Research Analyst | web + read/write |
| knowledge | `rag` | Knowledge Retriever | read_file |
| knowledge | `knowledge_manager` | Knowledge Manager | filesystem |
| knowledge | `learning` | Learning Agent | web + filesystem |
| engineering | `coding` | Software Engineer | code (switchable backend) |
| engineering | `documentation` | Technical Writer | filesystem |
| engineering | `git` | Version Control Engineer | shell + filesystem |
| engineering | `testing` | Test Engineer | code |
| quality | `qa` | QA Engineer | read_file, python_exec |
| quality | `reviewer` | Code Reviewer | read_file, list_dir |
| quality | `security` | Security Engineer | read_file, list_dir, shell |
| data | `data_analysis` | Data Analyst | csv/json/python |
| data | `database` | Database Engineer | python + read/write |
| ops | `devops` | DevOps Engineer | code |
| ops | `monitoring` | Monitoring Engineer | read_file, shell, http |
| comms | `email` | Email Assistant | read_file (send = approval) |
| comms | `calendar` | Calendar Assistant | none (reasoning only) |
| management | `task_manager` | Task Manager | none (reasoning only) |
| management | `memory_manager` | Memory Manager | none (reasoning only) |
| web | `browser` | Browser Operator | web |

**3 bespoke agents** — real classes in `agents/custom.py` with logic too specific for a
spec:

| Agent (`name`) | Role | What's special |
|---|---|---|
| `planner` | Planner | decomposes the goal into ordered, agent-assigned steps (structured output) |
| `ceo` | Chief Executive | synthesizes all step outputs into the final answer |
| `human_approval` | Human Approval Gate | pauses the run for sign-off on sensitive ops |

> Note: `coding` is listed as a spec but is instantiated as the custom `CodingAgent` so it
> can switch backend (brain / Claude Code / Codex). It still counts as one of the 20.

## Watching the flow live

In the Web UI (`/ui`) the **Flow** tab renders this exact pipeline as it happens —
plan node → a node per step (with the agent, the tool chips it called, and the verify
badge) → synthesize → reflect → result. The **Logs** tab shows the raw event stream and
the **Result** tab shows the final answer + per-step outputs.

## Concrete trace (offline mock)

```
planner.plan_ready      steps=3
agent.start             agent=research   step=1
tool.execute            tool=web_search  step=1
tool.result             ok=true  latency_ms=312  step=1
verifier.verdict        passed=true  confidence=0.95  step=1
agent.start             agent=coding     step=2
...
workflow.node           node=synthesize
workflow.node           node=reflect
task.completed          confidence=0.83
```

## Direct (non-pipeline) entry points

- `emp.chat(msg)` — fast single-turn chat with conversation memory (no planning).
- `emp.ask_agent("research", task)` — run one agent directly, skipping the pipeline.
- `emp.knowledge.answer(question)` — RAG answer grounded in ingested documents.

Use these for quick interactions, especially on slower local models where the full
multi-step pipeline takes longer.
