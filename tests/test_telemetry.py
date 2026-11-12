"""OpenTelemetry export: off by default, safe no-ops, correct spans and log mapping."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from ai_employee.config import get_settings
from ai_employee.core import telemetry
from ai_employee.core.exceptions import ApprovalRequired


def test_disabled_by_default_and_noops():
    assert get_settings().otel_enabled is False
    assert telemetry.setup_telemetry(get_settings()) is False
    assert telemetry.is_enabled() is False
    # Every helper is safe to call while disabled.
    with telemetry.span("x", a=1) as sp:
        telemetry.set_attrs(sp, b=2)
        telemetry.mark_error(sp, "nope")
    telemetry.count("tool_calls", tool="t", status="ok")
    telemetry.observe("tool_duration", 0.1, tool="t")
    assert telemetry.current_trace_id() is None


def test_middleware_passthrough_when_disabled():
    app = FastAPI()
    app.add_middleware(telemetry.TelemetryMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    assert TestClient(app).get("/ping").json() == {"ok": True}


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry._trace, "get_tracer", lambda name: provider.get_tracer(name))
    return exporter


def test_span_records_errors_but_not_approval_pauses(spans):
    with pytest.raises(ValueError), telemetry.span("boom"):
        raise ValueError("bad")
    with pytest.raises(ApprovalRequired), telemetry.span("paused"):
        raise ApprovalRequired("run_shell")
    by_name = {s.name: s for s in spans.get_finished_spans()}
    assert by_name["boom"].status.status_code is StatusCode.ERROR
    assert by_name["paused"].status.status_code is not StatusCode.ERROR


def test_log_events_map_to_otel_records(monkeypatch):
    emitted: list[dict] = []

    class FakeLogger:
        def emit(self, **kwargs):
            emitted.append(kwargs)

    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_otel_logger", FakeLogger())
    telemetry._export_log_event(
        {
            "event": "tool.execute",
            "level": "warning",
            "timestamp": "2026-01-01T00:00:00",
            "task_id": "t1",
            "agent": "coding",
            "args": {"path": "x"},
            "skip": None,
        }
    )
    (rec,) = emitted
    assert rec["body"] == "tool.execute"
    assert rec["severity_text"] == "WARNING"
    attrs = rec["attributes"]
    assert attrs["task_id"] == "t1" and attrs["agent"] == "coding"
    assert attrs["args"] == str({"path": "x"})  # non-primitive values are stringified
    assert "skip" not in attrs and "timestamp" not in attrs
