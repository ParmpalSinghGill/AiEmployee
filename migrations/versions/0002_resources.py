"""resources mirror table (optional Postgres catalog)

Revision ID: 0002_resources
Revises: 0001_initial
Create Date: 2026-10-16

Adds the ``resources`` table used to mirror the filesystem resource catalog into
Postgres for cross-task SQL queries. The filesystem catalog remains the source of
truth, so installing without Postgres is fully supported.
"""

from __future__ import annotations

from alembic import op

revision = "0002_resources"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from ai_employee.database.models import Resource

    Resource.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    from ai_employee.database.models import Resource

    Resource.__table__.drop(bind=op.get_bind(), checkfirst=True)
