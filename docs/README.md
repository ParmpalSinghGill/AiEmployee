# Documentation index

- [setup.md](setup.md) — **start here**: install, configure `.env`, run (CLI / UI / Python).
- [agent-flow.md](agent-flow.md) — how a goal flows through plan → execute → verify → synthesize → reflect.
- [providers.md](providers.md) — switch the brain, change the model, run local models (Ollama/Qwen3, vLLM, LM Studio).
- [architecture.md](architecture.md) — layers, the Brain abstraction, LangGraph orchestration, agents, tools, verification.
- [architecture-diagrams.md](architecture-diagrams.md) — the same design as diagrams: module map, execution graph, agent loop, runtime path.
- [memory.md](memory.md) — memory types, backends (pgvector / in-memory / resilient), RAG pipeline.
- [extending.md](extending.md) — add a provider, tool, agent, endpoint, or loader.
- [deployment.md](deployment.md) — Conda, Docker Compose, migrations, secrets, health, safety.

See the top-level [README](../README.md) for the quickstart and [CLAUDE.md](../CLAUDE.md)
for agent/contributor rules.
