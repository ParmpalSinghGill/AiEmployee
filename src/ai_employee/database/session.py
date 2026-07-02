"""Async SQLAlchemy engine & session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ai_employee.config import Settings, get_settings
from ai_employee.core.logging import get_logger

log = get_logger("database")

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine
    if _engine is None:
        s = settings or get_settings()
        _engine = create_async_engine(
            s.sqlalchemy_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        log.debug("database.engine_created")
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(settings), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


@asynccontextmanager
async def db_session(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """Transactional session context manager (commits on success, rolls back on error)."""
    maker = get_sessionmaker(settings)
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(settings: Settings | None = None) -> None:
    """Create the pgvector extension and all tables. For dev/bootstrap use;
    production should use Alembic migrations."""
    from sqlalchemy import text

    from ai_employee.database import models  # noqa: F401 - register tables
    from ai_employee.database.base import Base

    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    log.info("database.initialized")


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
