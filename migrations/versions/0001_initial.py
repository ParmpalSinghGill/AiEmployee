"""initial schema (pgvector extension + all tables)

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-28

This bootstrap migration enables the pgvector extension and creates the full
schema from the ORM metadata (the models are the source of truth). Subsequent
migrations should be generated with ``alembic revision --autogenerate``.
"""

from __future__ import annotations

from alembic import op

from ai_employee.database.base import Base
from ai_employee.database import models  # noqa: F401 - register tables

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
