# Choosing your brain & model

The LLM is an implementation detail behind the `Brain` interface, so **changing the
brain or the model is configuration only** — never code. Everything is driven by
`.env` (or real environment variables).

## Switch the whole brain

Set one variable:

```dotenv
LLM_PROVIDER=deepseek     # mock | deepseek | openai | gemini | glm | local
```

Then restart (`python -m ai_employee serve` / re-run the CLI). **Check what's active** any of these ways:

```bash
python -m ai_employee config         # prints the brain, model, embeddings, memory backend
curl -s localhost:8000/config        # JSON: provider, model, offline_mock, has_api_key…
```
- In the **Web UI** the active brain + model show as a badge in the top-right header
  (it turns amber and warns when you're on the offline mock or missing an API key).
- On startup the logs print `brain.select provider=…`.

> If you see `provider=mock` / `offline (placeholder answers)`, you're on the offline
> mock and answers are fake — set `LLM_PROVIDER` to a real provider below.

## Switch from the Web UI (no restart)

Click the **brain badge** in the top-right of `/ui` to open a picker: choose the
provider and model (for `local` it lists your installed Ollama models live), optionally
paste an API key for this session, and hit **Apply**. The active brain swaps immediately
— in-flight memory is preserved — and the badge updates. This changes the running
process only; set `.env` for a permanent default.

## Change just the model (same provider)

Each provider has its own model knobs. Override the chat/reasoning/embedding model
without touching the provider:

| Provider | Chat model var | Reasoning model var | Embedding model var |
|---|---|---|---|
| deepseek | `DEEPSEEK_CHAT_MODEL` | `DEEPSEEK_REASONING_MODEL` | — |
| openai | `OPENAI_CHAT_MODEL` | `OPENAI_REASONING_MODEL` | `OPENAI_EMBEDDING_MODEL` |
| gemini | `GEMINI_CHAT_MODEL` | — | `GEMINI_EMBEDDING_MODEL` |
| glm | `GLM_CHAT_MODEL` | — | `GLM_EMBEDDING_MODEL` |
| local | `LOCAL_CHAT_MODEL` | `LOCAL_REASONING_MODEL` | `LOCAL_EMBEDDING_MODEL` |

Example — use a smaller/faster OpenAI model:

```dotenv
LLM_PROVIDER=openai
OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Embeddings can use a different provider

RAG and long-term memory need embeddings. If your chat provider has none (e.g.
DeepSeek), set a separate embedder — no other change required:

```dotenv
LLM_PROVIDER=deepseek
EMBEDDING_PROVIDER=openai      # mock | openai | gemini | glm | local
EMBEDDING_DIM=1536            # MUST match the embedding model's dimension
```

---

## Run models locally (Ollama / vLLM / LM Studio / llama.cpp)

`LLM_PROVIDER=local` points the brain at any **OpenAI-compatible** local server.
Defaults target [Ollama](https://ollama.com) running **Qwen3 8B**.

### Quickstart with Ollama + Qwen3 8B

```bash
# 1. Install Ollama (https://ollama.com), then pull the models:
ollama pull qwen3:8b
ollama pull dengcao/Qwen3-Embedding-8B:Q4_K_M   # optional, for local RAG/memory

# 2. Configure .env:
```
```dotenv
LLM_PROVIDER=local
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_CHAT_MODEL=qwen3:8b
LOCAL_REASONING_MODEL=qwen3:8b      # Qwen3 has a built-in "thinking" mode
LOCAL_STRIP_THINK=true             # hide <think>…</think> from the visible answer

# Local embeddings (optional). Match EMBEDDING_DIM to the model!
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-8B:Q4_K_M
EMBEDDING_DIM=4096                  # Qwen3-Embedding-8B = 4096

# Avoid the Postgres warning unless you've set up pgvector with this dim:
MEMORY_BACKEND=memory
```
```bash
# 3. Run it:
python -m ai_employee run "Summarize the benefits of local LLMs"
python -m ai_employee serve        # then open http://localhost:8000/ui
```

### Other local servers

Anything OpenAI-compatible works — just change `LOCAL_BASE_URL` and model names:

```dotenv
# vLLM:        LOCAL_BASE_URL=http://localhost:8000/v1   LOCAL_CHAT_MODEL=Qwen/Qwen3-8B
# LM Studio:   LOCAL_BASE_URL=http://localhost:1234/v1
# llama.cpp:   LOCAL_BASE_URL=http://localhost:8080/v1   (server --api)
```

### Notes & tips

- **Embedding dimensions must match.** Qwen3-Embedding-8B = 4096, `nomic-embed-text`
  = 768, `mxbai-embed-large` = 1024. Set `EMBEDDING_DIM` accordingly (it sizes the
  pgvector column; the in-memory store accepts any length).
- **Tool calling** requires a tool-capable model (Qwen3 is). If a model can't call
  tools, agents still work — they just answer directly without tools.
- **Reasoning / thinking.** Qwen3 emits `<think>…</think>`; with `LOCAL_STRIP_THINK=true`
  that reasoning is moved into `ChatResult.extra["reasoning_content"]` and kept out of
  the user-facing answer.
- **Performance.** An 8B model on a single 8 GB GPU is comfortable but slower than a
  hosted API — multi-step tasks make several calls, so a full pipeline run can take a
  few minutes. Use a smaller model (e.g. `qwen3:4b`) for faster iteration.
