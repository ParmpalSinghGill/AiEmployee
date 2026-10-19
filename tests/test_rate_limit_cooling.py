"""Rate limits must pause visibly, not fail a run after a few seconds.

Covers the provider backoff (429 detection, Retry-After/Gemini retryDelay
parsing, capped exponential waits) and the task status that the console renders
as "Cooling down — rate limit".
"""

from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from ai_employee.brains.providers import openai_compatible
from ai_employee.brains.providers.openai_compatible import (
    OpenAICompatibleBrain,
    _is_rate_limit,
    _parse_retry_after,
    _retry_after_seconds,
    _retry_wait,
)
from ai_employee.core.exceptions import BrainError
from ai_employee.observability.tasks import TaskManager, TaskRecord, TaskStatus


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 429,
        headers: dict[str, str] | None = None,
        body: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def json(self) -> dict:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


class _RateLimit(Exception):
    def __init__(
        self,
        *,
        retry_after: str | None = None,
        body: dict | None = None,
        status_code: int = 429,
    ) -> None:
        super().__init__("429 Too Many Requests: quota exceeded")
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        self.response = _FakeResponse(
            status_code=status_code, headers=headers, body=body
        )


class _Connection(Exception):
    pass


def _brain(*, max_retries: int = 3, rate_limit_retries: int = 4) -> OpenAICompatibleBrain:
    """Build a Brain without the OpenAI client — ``_call`` only needs these attrs."""
    brain = OpenAICompatibleBrain.__new__(OpenAICompatibleBrain)
    brain.name = "gemini"
    brain._chat_model = "gemini-test"
    brain._max_retries = max_retries
    brain._rate_limit_retries = max(max_retries, rate_limit_retries)
    brain._rate_limit_base_wait = 5.0
    brain._rate_limit_max_wait = 60.0
    return brain


# ── Classification / header parsing ─────────────────────────────────────────


def test_is_rate_limit_detects_status_and_quota_text() -> None:
    assert _is_rate_limit(_RateLimit())
    assert _is_rate_limit(Exception("RESOURCE_EXHAUSTED: quota exceeded"))
    assert _is_rate_limit(Exception("Error code: 429 - Too Many Requests"))
    assert not _is_rate_limit(_Connection("connection reset"))


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("30", 30.0),
        ("1m30s", 90.0),
        ("250ms", 0.25),
        ("6m0s", 360.0),
        ("", None),
        ("not-a-delay", None),
    ],
)
def test_parse_retry_after(raw: str, seconds: float | None) -> None:
    assert _parse_retry_after(raw) == seconds


def test_retry_after_reads_headers_and_gemini_retry_delay() -> None:
    assert _retry_after_seconds(_RateLimit(retry_after="17")) == 17.0
    exc = _RateLimit(body={"error": {"details": [{"retryDelay": "31s"}]}})
    assert _retry_after_seconds(exc) == 31.0


def test_retry_wait_honors_hint_and_caps_exponential() -> None:
    exc = _RateLimit(retry_after="7")
    assert _retry_wait(exc, 1, rate_limited=True, base=5.0, cap=60.0) == 7.0
    assert _retry_wait(exc, 1, rate_limited=True, base=5.0, cap=3.0) == 3.0
    # No hint for a connection error: 1, 2, 4 … capped at 20.
    assert _retry_wait(_Connection("x"), 3, rate_limited=False, base=1.0, cap=20.0) == 4.0


# ── Provider retry loop ─────────────────────────────────────────────────────


async def test_rate_limit_retries_then_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = _brain(max_retries=3, rate_limit_retries=4)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(openai_compatible, "_RETRYABLE", (_RateLimit,))

    calls = {"n": 0}

    async def fn(**_: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimit(retry_after="3")
        return "ok"

    with capture_logs() as logs:
        result = await brain._call(fn, model="gemini-test")

    assert result == "ok"
    assert calls["n"] == 2
    assert sleeps == [3.0]
    events = [entry["event"] for entry in logs]
    assert "brain.rate_limited" in events
    assert "brain.rate_limit_cleared" in events
    limited = next(e for e in logs if e["event"] == "brain.rate_limited")
    assert limited["attempt"] == 1
    assert limited["max_attempts"] == 4
    assert limited["wait_s"] == 3.0
    assert limited["retry_at"]
    assert limited["cooling_id"]
    cleared = next(e for e in logs if e["event"] == "brain.rate_limit_cleared")
    assert cleared["cooling_id"] == limited["cooling_id"]
    assert cleared["attempts"] == 2


async def test_rate_limit_exhaustion_logs_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain = _brain(max_retries=3, rate_limit_retries=3)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(openai_compatible, "_RETRYABLE", (_RateLimit,))

    async def fn(**_: object) -> str:
        raise _RateLimit(retry_after="1")

    with capture_logs() as logs, pytest.raises(BrainError):
        await brain._call(fn, model="gemini-test")

    assert sleeps == [1.0, 1.0]  # attempts 1 and 2 sleep; attempt 3 is the limit
    events = [entry["event"] for entry in logs]
    assert "brain.rate_limit_exhausted" in events
    assert "brain.rate_limit_cleared" not in events


async def test_other_final_error_clears_a_previous_rate_limit_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain = _brain(max_retries=2, rate_limit_retries=4)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(openai_compatible, "_RETRYABLE", (_RateLimit, _Connection))

    calls = {"n": 0}

    async def fn(**_: object) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimit(retry_after="2")
        raise _Connection("connection reset")

    with capture_logs() as logs, pytest.raises(BrainError):
        await brain._call(fn)

    # The 429 cooldown is 2s; the two connection retries are 1s each and use
    # their own budget even though the total attempt count is higher.
    assert sleeps == [2.0, 1.0]
    events = [entry["event"] for entry in logs]
    assert "brain.rate_limited" in events
    assert "brain.retry" in events
    cleared = next(e for e in logs if e["event"] == "brain.rate_limit_cleared")
    assert cleared["outcome"] == "failed"


async def test_transient_errors_keep_the_short_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain = _brain(max_retries=2, rate_limit_retries=6)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(openai_compatible, "_RETRYABLE", (_Connection,))

    async def fn(**_: object) -> str:
        raise _Connection("connection reset")

    with capture_logs() as logs, pytest.raises(BrainError):
        await brain._call(fn)

    assert sleeps == [1.0]  # only one retry: llm_max_retries == 2
    events = [entry["event"] for entry in logs]
    assert "brain.retry" in events
    assert "brain.rate_limited" not in events


# ── Task status shown by the console ────────────────────────────────────────


def _rate_limit_event(
    task: TaskRecord,
    cooling_id: str,
    *,
    step: int = 1,
    until: str = "2030-01-01T00:00:10+00:00",
) -> dict:
    return {
        "task_id": task.id,
        "event": "brain.rate_limited",
        "level": "warning",
        "cooling_id": cooling_id,
        "provider": "gemini",
        "model": "gemini-test",
        "agent": "research",
        "step": step,
        "attempt": 1,
        "max_attempts": 6,
        "wait_s": 5.0,
        "retry_at": until,
    }


def test_cooling_status_is_visible_and_parallel_cooldowns_are_tracked() -> None:
    tm = TaskManager()
    task = TaskRecord(goal="g", session_id="s")
    tm._tasks[task.id] = task

    tm._route_log(_rate_limit_event(task, "c1", step=1))
    assert task.status is TaskStatus.cooling
    snap = task.snapshot()
    assert snap["cooling"] is not None
    assert snap["cooling"]["until"] == "2030-01-01T00:00:10+00:00"
    assert snap["cooling"]["agent"] == "research"
    assert snap["cooling"]["step"] == 1
    assert snap["cooling"]["count"] == 1

    # A second, longer cooldown keeps the task cooling with a later resume time.
    tm._route_log(
        _rate_limit_event(
            task, "c2", step=2, until="2030-01-01T00:01:00+00:00"
        )
    )
    snap = task.snapshot()
    assert snap["cooling"]["count"] == 2
    assert snap["cooling"]["until"] == "2030-01-01T00:01:00+00:00"

    # Clearing only one cooldown must not pretend the whole run is running.
    tm._route_log(
        {
            "task_id": task.id,
            "event": "brain.rate_limit_cleared",
            "level": "info",
            "cooling_id": "c1",
        }
    )
    assert task.status is TaskStatus.cooling
    assert task.snapshot()["cooling"]["count"] == 1

    tm._route_log(
        {
            "task_id": task.id,
            "event": "brain.rate_limit_cleared",
            "level": "info",
            "cooling_id": "c2",
        }
    )
    assert task.status is TaskStatus.running
    assert task.snapshot()["cooling"] is None


def test_late_rate_limit_event_does_not_reopen_a_finished_task() -> None:
    tm = TaskManager()
    task = TaskRecord(goal="g", session_id="s")
    task.status = TaskStatus.completed
    tm._tasks[task.id] = task

    tm._route_log(_rate_limit_event(task, "late"))

    assert task.status is TaskStatus.completed
    assert task.cooling == {}


def test_exhausted_cooldown_returns_to_running() -> None:
    tm = TaskManager()
    task = TaskRecord(goal="g", session_id="s")
    tm._tasks[task.id] = task

    tm._route_log(_rate_limit_event(task, "c1"))
    assert task.status is TaskStatus.cooling

    tm._route_log(
        {
            "task_id": task.id,
            "event": "brain.rate_limit_exhausted",
            "level": "error",
            "cooling_id": "c1",
        }
    )
    assert task.status is TaskStatus.running
    assert task.snapshot()["cooling"] is None
