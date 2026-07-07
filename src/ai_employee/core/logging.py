"""Structured logging via structlog.

Console-friendly rendering in development, JSON in production (set LOG_JSON=true).
Every log line can carry structured context (task_id, agent, tool, latency_ms…),
which is what makes reasoning traces and performance metrics queryable later.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False

# Observers receive a copy of every structured log event (as a dict). Used by the
# web UI / task manager to attribute live agent & tool activity to a running task.
_log_observers: list[Any] = []


def add_log_observer(fn: Any) -> None:
    """Register a callable invoked with each structured log event dict."""
    if fn not in _log_observers:
        _log_observers.append(fn)


def remove_log_observer(fn: Any) -> None:
    if fn in _log_observers:
        _log_observers.remove(fn)


def _observer_processor(logger, method_name, event_dict):
    """structlog processor that fans each event out to registered observers.

    Runs before the renderer so observers see the full structured dict (including
    contextvars like ``task_id``). Observer errors are swallowed — telemetry must
    never break logging.
    """
    if _log_observers:
        snapshot = dict(event_dict)
        for obs in list(_log_observers):
            with contextlib.suppress(Exception):
                obs(snapshot)
    return event_dict


# Console verbosity gate. The wrapper lets INFO+ reach the processors (so the UI
# captures everything), and this drops sub-threshold events from being *printed*.
_console_level = logging.INFO
_LEVELS = {"debug": 10, "info": 20, "warning": 30, "warn": 30, "error": 40, "critical": 50}


def _console_gate(logger, method_name, event_dict):
    level_name = str(event_dict.get("level", "info")).lower()
    if _LEVELS.get(level_name, 20) < _console_level:
        raise structlog.DropEvent
    return event_dict


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog + stdlib logging.

    Safe to call repeatedly: an explicit call always re-applies the level/format,
    so a later ``configure_logging(settings.log_level)`` overrides an earlier
    default triggered lazily by the first ``get_logger``.
    """
    global _CONFIGURED, _console_level

    numeric = getattr(logging, level.upper(), logging.INFO)
    _console_level = numeric
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=True)
    )

    # Let INFO+ always flow through the processor chain so observers (the web UI's
    # event capture) see all activity regardless of console verbosity. The console
    # gate below then drops events below the configured level from being *printed*.
    wrapper_level = min(numeric, logging.INFO)

    structlog.configure(
        processors=[*shared_processors, _observer_processor, _console_gate, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(wrapper_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # Don't cache the bound logger so a later reconfigure (e.g. settings-driven
        # level) takes effect for loggers obtained afterwards.
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None, **initial: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, optionally pre-populated with context."""
    if not _CONFIGURED:
        configure_logging()
    logger = structlog.get_logger(name)
    if initial:
        logger = logger.bind(**initial)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind values into the contextvar so they appear on every subsequent log line."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
