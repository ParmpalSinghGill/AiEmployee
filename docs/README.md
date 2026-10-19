# Documentation index

- [how-it-works.md](how-it-works.md) — **the complete, code-verified explanation of the whole system**: every layer, the Brain boundary, all three orchestrators, the agent ReAct loop, tools/approval/MCP, memory, RAG, verification, observability, config, and where every byte of state lives. Start here if you want to understand everything end to end.
- [test-scenarios.md](test-scenarios.md) — **realtime acceptance tests**: five goals graded on agent routing and result quality, runnable against the offline mock or a real provider (`python scripts/run_scenarios.py`).
- [resource-isolation.md](resource-isolation.md) — task ↔ source links, reference-counted deletion, and the shared-vs-private isolation action (console 🗂 Resources button).
- [setup.md](setup.md) — **start here to run it**: install, configure `.env`, run (CLI / UI / Python).
- [agent-flow.md](agent-flow.md) — how a goal flows through plan → execute → verify → synthesize → reflect.
- [providers.md](providers.md) — switch the brain, change the model, run local models (Ollama/Qwen3, vLLM, LM Studio).
- [architecture.md](architecture.md) — layers, the Brain abstraction, LangGraph orchestration, agents, tools, verification.
- [architecture-diagrams.md](architecture-diagrams.md) — the same design as diagrams: module map, execution graph, agent loop, runtime path.
- [repo-flow.html](repo-flow.html) — **the whole repo on one flow diagram**: every entry point, the task runner, orchestrator, gates, brains, memory, and where state lives. Self-contained page — open it in any browser, fully offline. Standalone images in [images/](images/): [repo-flow-master.png](images/repo-flow-master.png) (11476×6584) / [.svg](images/repo-flow-master.svg), and [repo-flow-scheduler.png](images/repo-flow-scheduler.png) / [.svg](images/repo-flow-scheduler.svg).
- [memory.md](memory.md) — memory types, backends (pgvector / in-memory / resilient), RAG pipeline.
- [extending.md](extending.md) — add a provider, tool, agent, endpoint, or loader.
- [deployment.md](deployment.md) — Conda, Docker Compose, migrations, secrets, health, safety.

See the top-level [README](../README.md) for the quickstart and [CLAUDE.md](../CLAUDE.md)
for agent/contributor rules.
