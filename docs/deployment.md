# Deployment

## Local (Conda)

```bash
conda env create -f environment.yml
conda activate AIEmployee
pip install -e .
cp .env.example .env            # set LLM_PROVIDER and the matching API key
python -m ai_employee serve     # http://localhost:8000/docs
```

## Docker Compose (full stack)

Brings up Postgres+pgvector, Redis, runs migrations, then serves the API:

```bash
cp .env.example .env            # fill in provider keys
docker compose up --build
```

Services:
- `postgres` — `pgvector/pgvector:pg16`, persisted in the `pgdata` volume.
- `redis` — `redis:7-alpine`.
- `api` — built from the `Dockerfile`; runs `alembic upgrade head` then `uvicorn`.

Override anything via environment variables (see `docker-compose.yml`). For production set
`MEMORY_BACKEND=pgvector` and `ENVIRONMENT=production`.

## Database migrations

The schema is managed by Alembic over the async (`asyncpg`) driver:

```bash
alembic upgrade head                       # apply
alembic revision --autogenerate -m "msg"   # create a new migration after model changes
```

The bootstrap migration (`migrations/versions/0001_initial.py`) enables the `vector`
extension and creates all tables. `EMBEDDING_DIM` must match your embedding model and the
`Vector(dim)` columns — if you change models, regenerate the embeddings/migration.

## Configuration & secrets

- Everything is configured through environment variables / `.env` (`config/settings.py`).
- **Never commit `.env`** — it is gitignored. Provide secrets via your orchestrator's
  secret manager in production.
- The `DATABASE_URL` is derived from the `POSTGRES_*` parts unless explicitly set.

## Health & observability

- `GET /health` — liveness probe (used by the Docker healthcheck).
- Structured logs via `structlog`; set `LOG_JSON=true` in production for aggregation.
- Agent/tool activity is traceable (latency, tool calls) and persistable to the
  `agent_events` table.

## Autonomy & safety in production

Operations in `REQUIRE_APPROVAL_FOR` (e.g. `shell_write,email_send,git_push,db_write,
deploy`) require explicit approval. Pass approved operation keys per task (API:
`approved_ops`) only after a human has authorized them. Set `ENABLE_CODE_EXECUTION=false`
to hard-disable the code/shell tools.
