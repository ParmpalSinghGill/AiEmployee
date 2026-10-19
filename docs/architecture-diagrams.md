# Architecture diagrams

Companion to [architecture.md](architecture.md), which carries the prose. This file
carries the pictures: the layered module map, the LangGraph state machine, the agent's
ReAct loop, and the runtime path of one task through the web console.

Diagrams describe the code **as committed** (`src/ai_employee/`), not an aspirational
design. A richer, styled version of the same four diagrams is in
[architecture-diagrams.html](architecture-diagrams.html) — open it in a browser.

Counts as of this writing: **29 agents**, **24 tools** (29 with Google Workspace
connected), **7 providers**, **3 orchestrators**, **7 memory namespaces**, **25 MCP
connector presets**.

---

## 1. Layered structure

Dependencies point downward only. `Brain` is the choke point every model call passes
through — swapping `LLM_PROVIDER` changes the whole system's reasoning engine without
touching a line above it.

```mermaid
flowchart TD
    subgraph entry["ENTRY"]
        CLI["CLI<br/><code>cli.py</code>"]
        API["FastAPI<br/><code>api/routes.py</code>"]
        UI["Web console /ui<br/><code>api/ui.py</code> + static SPA"]
        TM["TaskManager<br/><code>observability/tasks.py</code>"]
    end

    FACADE["<b>AIEmployee</b> — facade &amp; DI container<br/><code>core/employee.py</code><br/>run · plan · research · download · chat"]

    subgraph graphs["ORCHESTRATORS — three LangGraph state machines"]
        ORCH["Orchestrator<br/><code>employee_graph.py</code>"]
        PLAN["Plan<br/><code>plan_graph.py</code>"]
        RES["Research<br/><code>research_graph.py</code>"]
    end

    subgraph cap["CAPABILITY"]
        AG["Agents<br/>registry · specs.py<br/>29 roles"]
        TL["Tools<br/>registry · builtins/<br/>approval-gated"]
        ME["Memory<br/>manager · resilient<br/>7 namespaces"]
        KN["Knowledge<br/>loaders · chunking<br/>RAG over memory"]
    end

    VER["Verifier — self-review, retry budget, confidence"]
    BRAIN["<b>Brain</b> — the only interface that talks to a model<br/>chat · stream · complete · structured_output · embed"]
    PROV["deepseek · openai · gemini · glm · opencode · local · mock"]

    CLI --> FACADE
    API --> FACADE
    UI -->|"spawns background task"| TM
    TM --> FACADE
    FACADE --> ORCH & PLAN & RES
    ORCH --> AG & TL & ME
    RES --> KN
    PLAN --> AG
    TL -.-> VER
    ME -.-> VER
    AG --> BRAIN
    VER --> BRAIN
    KN --> BRAIN
    BRAIN -->|"LLM_PROVIDER selects one"| PROV
```

Every arrow is a compile-time import direction. Nothing above the `Brain` band knows
which provider is loaded — `brains/factory.py` is the single place that maps a name to a
class, which is what makes the `mock` provider a complete offline substitute rather than
a stub.

---

## 2. The execution graph

The main `Orchestrator` is a LangGraph state machine over a typed `EmployeeState`. The
interesting part is not the happy path — it is the two ways a run leaves the loop early,
and the state that survives when it does.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> route

    route --> plan: fresh goal
    route --> execute: resume_execute &amp; plan exists<br/>(skip planning)

    plan --> execute: steps[]

    execute --> execute: continue · cursor+1
    execute --> halt: needs_approval
    execute --> synthesize: cursor ≥ len(steps)

    halt --> [*]: paused, checkpoint saved
    halt --> route: approve → re-enter with<br/>plan, cursor, step_results, approved_ops

    synthesize --> reflect
    reflect --> [*]

    note right of execute
        per step:
        · skip if a dependency yielded nothing usable
        · agent.run() → Verifier.verify()
        · retry ≤ 1, unless the output is a policy refusal
        · audit_resources() checks claimed downloads exist
    end note

    note right of reflect
        writes long_term + skill memory
        ONLY if confidence ≥ 0.5,
        all steps verified,
        and no resource issues
    end note
```

The **halt path** is what makes long-running work survivable: the run stops mid-plan,
`checkpoint_from_state` serializes cursor and results, and approval re-enters at
`execute` rather than re-planning from scratch. The **reflect** node is deliberately
conservative — an unverified or low-confidence run teaches the system nothing rather
than poisoning recall.

---

## 3. Inside a step: the agent loop

Each `execute` node hands one step to a `ToolUsingAgent` (`agents/base.py`) — a ReAct
loop bounded on four separate axes, because an unbounded tool loop is the default
failure mode of this architecture.

```mermaid
flowchart LR
    TASK(["step task"]) --> BUILD["build messages<br/>+ recalled memory"]
    BUILD --> COMPACT["compact<br/>history window"]
    COMPACT --> CHAT["<b>brain.chat()</b><br/>messages + tool specs<br/>temperature 0.2"]
    CHAT --> Q{"tool<br/>calls?"}

    Q -->|no| ANS(["<b>answer</b><br/>AgentResult"])
    Q -->|yes| REG["ToolRegistry.execute<br/>approved = is_approved(call)"]

    REG -->|"approval_key set<br/>&amp; not approved"| APPR(["<b>ApprovalRequired</b><br/>halts the whole graph"])
    REG -->|"observation → messages"| COMPACT

    BREAK{{"circuit breakers —<br/>any one ends the loop"}} -.-> Q
    B1["max_iterations<br/>per agent"] --- BREAK
    B2["3× duplicate<br/>identical calls"] --- BREAK
    B3["repeat failure<br/>per error class"] --- BREAK
    B4["policy refusal<br/>no retry"] --- BREAK
```

The loop's exit conditions carry the engineering. A duplicate-call counter, a
per-error-class failure counter, and a refusal detector each break out with a `[SYSTEM]`
nudge injected into the observation, so the agent synthesizes from what it already has
instead of burning the iteration budget. The **approval branch** does not return an
error — it raises through the agent and the graph node alike, which is why a sensitive
tool can pause a whole multi-step run.

---

## 4. Runtime path of one task

The console never blocks on a run. `TaskManager` launches the pipeline in its own
asyncio task with a `task_id` bound to the logging context; the SPA polls a snapshot
cursor. That contextvar is the entire mechanism by which deeply nested log events find
the right browser tab.

```mermaid
sequenceDiagram
    autonumber
    participant SPA as Browser · static/app.js
    participant TM as TaskManager (API process)
    participant BG as Background asyncio task
    participant LOG as Processor chain · core/logging.py
    participant REC as TaskRecord

    SPA->>TM: POST /ui/api/tasks
    TM->>BG: spawn, bind task_id contextvar
    TM-->>SPA: returns immediately (task id)

    activate BG
    BG->>BG: employee.run() — mode: run | plan | research | download
    BG->>LOG: structlog events at INFO+
    deactivate BG

    LOG->>REC: _route_log by task_id → flow · logs · explain
    Note over LOG,REC: console printing only is gated by LOG_LEVEL — capture is never level-filtered

    loop while running
        SPA->>REC: GET /ui/api/tasks/{id}?since=n
        REC-->>SPA: incremental snapshot
    end

    opt approval gate hit
        REC-->>SPA: status = awaiting_approval
        SPA->>TM: POST /ui/api/tasks/{id}/approve
        TM->>BG: resume from saved checkpoint
    end
```

Log capture is intentionally decoupled from console verbosity: `INFO` and above always
reach the processor chain so observers can attribute them, and only *printing* is
filtered by `LOG_LEVEL`. Reintroducing level filtering at the `wrapper_class` would
silently blank the dashboard's live flow while leaving every test green.

---

## Invariants the design rests on

These are not style preferences — each one is load-bearing, and violating it breaks a
property the rest of the architecture assumes.

| Invariant | What it means |
|---|---|
| **No SDK above the Brain** | Business logic never imports an LLM SDK. Provider knowledge lives in `brains/providers/` and `brains/factory.py`, nowhere else. |
| **Config, not code** | `LLM_PROVIDER` is the single switch. No module branches on provider identity to change behavior. |
| **Offline is first-class** | `mock` brain + `mock` embeddings + in-memory store must run the full pipeline with no network and no database. The test suite enforces it. |
| **Async all the way down** | Public APIs are `async`; nothing blocks the event loop, because the task runner shares it with the API process. |
| **Sensitive ops are gated** | A tool with an `approval_key` in `approval_required` cannot execute without `approved=True` — the gate is in the registry, not in any agent's prompt. |

Approval keys currently enforced: `db_write`, `deploy`, `email_send`, `git_push`,
`resource_acquisition`, `resource_download`, `shell_write`, `calendar_write`, `mcp_tool`.
