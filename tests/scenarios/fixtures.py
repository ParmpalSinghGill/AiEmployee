"""Seed fixtures for the scenario suite.

Everything here writes *into a scenario workspace* so the sandboxed tools can read
it. Keep the content small and deterministic so the evaluators can compute ground
truth without guessing.
"""

from __future__ import annotations

from pathlib import Path

# ── S2: a deterministic sales file ───────────────────────────────────────────
# Region totals: North 310, South 630, East 450, West 180  (overall 1570)
# Month totals : 2024-01 = 500, 2024-02 = 520, 2024-03 = 550
# => strongest month = March (2024-03), weakest = January (2024-01)
SALES_CSV = """date,region,sales,units
2024-01-05,North,100,10
2024-01-12,South,200,20
2024-01-19,East,150,15
2024-01-26,West,50,5
2024-02-05,North,120,12
2024-02-12,South,180,18
2024-02-19,East,160,16
2024-02-26,West,60,6
2024-03-05,North,90,9
2024-03-12,South,250,25
2024-03-19,East,140,14
2024-03-26,West,70,7
"""

# ── S3: a tiny project that lacks a /health endpoint ─────────────────────────
SAMPLE_APP = '''"""A minimal demo service with no status route yet (scenario S3)."""

from fastapi import FastAPI

app = FastAPI(title="Sample App")


@app.get("/")
def root() -> dict[str, str]:
    return {"hello": "world"}


@app.get("/items/{item_id}")
def read_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}
'''

SAMPLE_APP_TEST = '''"""Existing tests for the sample app (no status-route coverage yet)."""

from fastapi.testclient import TestClient

from sampleapp.app import app

client = TestClient(app)


def test_root() -> None:
    assert client.get("/").status_code == 200


def test_read_item() -> None:
    resp = client.get("/items/7")
    assert resp.status_code == 200
    assert resp.json() == {"item_id": 7}
'''

SAMPLE_README = """# Sample App

A minimal FastAPI demo used by the endpoint and deployment scenarios.

## Run

```bash
uvicorn sampleapp.app:app --reload
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Hello world |
| GET | `/items/{item_id}` | Echo an item id |
"""

SAMPLE_REQUIREMENTS = "fastapi\nuvicorn\nhttpx\npytest\n"

# ── S5: a deployment setup with deliberate, reviewable problems ──────────────
DOCKERFILE = """# Intentionally imperfect deployment config (scenario S5) — DO NOT copy.
FROM python:latest

ENV DB_PASSWORD=supersecret123
ENV API_TOKEN=ghp_hardcodedtokenvalue

WORKDIR /app

# Copies everything, including .env, .git and any local secrets.
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# No USER directive: the container runs as root.
# No HEALTHCHECK.
EXPOSE 8000
CMD ["uvicorn", "sampleapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

COMPOSE = """# Intentionally imperfect compose file (scenario S5).
version: "3.9"
services:
  api:
    image: sampleapp:latest
    ports:
      - "8000:8000"
    privileged: true
    environment:
      - POSTGRES_PASSWORD=hunter2
      - DEBUG=true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    # No healthcheck, no restart policy, no resource limits.
  db:
    image: postgres:latest
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
    ports:
      - "5432:5432"
"""

CI_YML = """# Intentionally imperfect CI (scenario S5).
name: ci
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@master
      - name: Install
        run: pip install -r requirements.txt
      - name: Test
        continue-on-error: true
        run: pytest
      - name: Show token
        run: echo "token is ${{ secrets.PROD_TOKEN }}"
"""

DEPLOY_README = """# deploy_review

A small, deliberately flawed deployment setup for the review-only scenario.
Files: `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`.
"""


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# ── Seeders ──────────────────────────────────────────────────────────────────
def seed_sales(root: Path) -> None:
    _write(root, "data/sales.csv", SALES_CSV)


def seed_sample_project(root: Path) -> None:
    _write(root, "sampleapp/__init__.py", "")
    _write(root, "sampleapp/app.py", SAMPLE_APP)
    _write(root, "sampleapp/tests/__init__.py", "")
    _write(root, "sampleapp/tests/test_app.py", SAMPLE_APP_TEST)
    _write(root, "sampleapp/README.md", SAMPLE_README)
    _write(root, "sampleapp/requirements.txt", SAMPLE_REQUIREMENTS)


def seed_deployment(root: Path) -> None:
    _write(root, "deploy_review/Dockerfile", DOCKERFILE)
    _write(root, "deploy_review/docker-compose.yml", COMPOSE)
    _write(root, "deploy_review/.github/workflows/ci.yml", CI_YML)
    _write(root, "deploy_review/README.md", DEPLOY_README)


def seed_none(root: Path) -> None:  # pragma: no cover - trivial
    return None


__all__ = [
    "SALES_CSV",
    "seed_sales",
    "seed_sample_project",
    "seed_deployment",
    "seed_none",
]
