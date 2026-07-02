# Setup guide

Step-by-step setup for the AI Employee Framework. The whole thing runs offline with
zero infrastructure, then you point it at a real model when ready.

---

## 1. Prerequisites

- **Conda** (Miniconda/Anaconda) — for the Python 3.12 environment
- **Git** (optional)
- For local models: **[Ollama](https://ollama.com)** (optional)
- For durable memory/DB: **Docker** (optional — Postgres+pgvector)

Everything except Conda is optional.

---

## 2. Create the environment

```bash
cd /media/parmpal/Data/Codes/VibeCoding/AIEmployee
conda env create -f environment.yml      # creates the "AIEmployee" env (Python 3.12)
conda activate AIEmployee
pip install -e .                          # install the package + CLI
```

(Or `make env && conda activate AIEmployee && make dev`.)

---

## 3. Configure (`.env`)

Your settings live in a **`.env`** file at the project root (it overrides the code
defaults in `src/ai_employee/config/settings.py`). Start from the template:

```bash
cp .env.example .env
```

Pick **one** of these in `.env`:

### A) Offline mock (no keys, placeholder answers — for trying the system)
```dotenv
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=mock
MEMORY_BACKEND=memory
```

### B) Local model with Ollama + Qwen3 8B (private, free)
```bash
ollama pull qwen3:8b
```
```dotenv
LLM_PROVIDER=local
LOCAL_CHAT_MODEL=qwen3:8b
LOCAL_BASE_URL=http://localhost:11434/v1
EMBEDDING_PROVIDER=mock        # or local + EMBEDDING_DIM=4096 for real embeddings
MEMORY_BACKEND=memory
```

### C) Hosted provider (DeepSeek / OpenAI / Gemini / GLM)
```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
EMBEDDING_PROVIDER=openai       # DeepSeek has no embeddings; falls back to this
OPENAI_API_KEY=sk-...
```

> **Precedence:** GUI badge (live) → shell env var → `.env` → `settings.py` defaults.

Check what's active at any time:
```bash
ai-employee config
```

---

## 4. Run it

### Web Console (recommended)
```bash
ai-employee serve            # → http://localhost:8000/ui
```
In the UI you can: type a goal, watch the live agent/tool flow, read logs, see the
result, ask follow-ups, switch the brain (top-right badge), and reopen saved tasks.

### CLI
```bash
ai-employee run "Research async Python and write a brief"   # full multi-agent pipeline
ai-employee chat "Give me 3 tips for labeling a dataset"     # fast single-turn
ai-employee ask research "Find public fire-detection datasets"
ai-employee ingest ./docs                                    # add docs to the knowledge base
ai-employee query "What providers are supported?"
ai-employee agents          # list agents
ai-employee tools           # list tools
ai-employee config          # show active brain/model
```

### Python
```python
import asyncio
from ai_employee import AIEmployee

async def main():
    emp = AIEmployee()
    result = await emp.run("Summarize the benefits of local LLMs")
    print(result["answer"])
    await emp.aclose()

asyncio.run(main())
```

---

## 5. Optional: durable memory with Postgres + pgvector

For persistent vector memory across machines (instead of the in-memory store):

```bash
docker compose up -d postgres     # starts pgvector/pgvector:pg16
alembic upgrade head              # creates the extension + tables
```
```dotenv
MEMORY_BACKEND=pgvector
POSTGRES_HOST=localhost
POSTGRES_USER=ai_employee
POSTGRES_PASSWORD=ai_employee
POSTGRES_DB=ai_employee
EMBEDDING_DIM=1536                # must match your embedding model
```

Or run the **whole stack** (Postgres + Redis + API) in containers:
```bash
docker compose up --build         # → http://localhost:8000/ui
```

---

## 6. Verify

```bash
make test        # 50 offline tests (mock provider, in-memory store) — should pass
make lint        # ruff
ai-employee config   # confirm the active brain is what you expect
```

Next: read **[agent-flow.md](agent-flow.md)** to understand how a goal becomes a result.
