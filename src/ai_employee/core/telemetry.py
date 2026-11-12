"""OpenTelemetry export: traces → Tempo, metrics → Prometheus, logs → Loki.

Everything leaves the process over OTLP/HTTP to one OpenTelemetry Collector
(``docker-compose.observability.yml``), which routes each signal to its backend.
Grafana sits on top of all three.

Off by default (``OTEL_ENABLED=false``) so the offline pipeline and the tests
need no network. When off, every helper here is a cheap no-op: spans come from
the OpenTelemetry API's no-op tracer and metric calls return immediately.

What is captured:

* **Traces** — one trace per task (``task.run``) with child spans for each plan
  step, agent turn, LLM call and tool call; plus one span per HTTP request.
* **Metrics** — counters/histograms for tasks, steps, agent runs, LLM calls
  (latency, tokens, errors), tool calls and HTTP requests.
* **Logs** — every structlog event at INFO+ (the same stream the web UI sees),
  with its structured fields as attributes and the active trace/span id, so
  Grafana can jump from a log line to its trace and back.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

from ai_employee.core.logging import add_log_observer, get_logger

log = get_logger("telemetry")

try:  # the API package is tiny and always installed; guard anyway
    from opentelemetry import trace as _trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # noqa: BLE001 - telemetry must never break the app
    _trace = None  # type: ignore[assignment]

_enabled = False
_providers: list[Any] = []
_otel_logger: Any = None
_instruments: dict[str, Any] = {}

# structlog keys that are not useful as log attributes (already mapped elsewhere).
_SKIP_KEYS = {"event", "level", "timestamp", "exc_info", "stack_info"}
_NOT_ERRORS = {"CancelledError", "ApprovalRequired", "GeneratorExit"}
_SEVERITY = {"debug": 5, "info": 9, "warning": 13, "warn": 13, "error": 17, "critical": 21}


def is_enabled() -> bool:
    return _enabled


# ── Setup / shutdown ─────────────────────────────────────────────────────────
def setup_telemetry(settings: Any) -> bool:
    """Start exporting to the collector if ``settings.otel_enabled``.

    Idempotent. Returns True when export is active. Never raises: a missing
    package or bad endpoint only logs a warning.
    """
    global _enabled, _otel_logger
    if _enabled or not getattr(settings, "otel_enabled", False):
        return _enabled
    try:
        from opentelemetry import metrics
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:  # noqa: BLE001
        log.warning("telemetry.unavailable", error=str(exc))
        return False

    endpoint = str(settings.otel_exporter_otlp_endpoint).rstrip("/")
    env = getattr(settings, "environment", "development")
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.1.0",
            "deployment.environment": getattr(env, "value", str(env)),
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    _trace.set_tracer_provider(tracer_provider)

    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
        export_interval_millis=int(settings.otel_metric_export_interval_ms),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{endpoint}/v1/logs"))
    )
    set_logger_provider(logger_provider)
    _otel_logger = logger_provider.get_logger("ai_employee")

    _providers[:] = [tracer_provider, meter_provider, logger_provider]
    _instruments.clear()
    _enabled = True
    add_log_observer(_export_log_event)
    log.info("telemetry.enabled", endpoint=endpoint, service=settings.otel_service_name)
    return True


def shutdown_telemetry() -> None:
    """Flush and stop the exporters (call on app shutdown)."""
    global _enabled
    if not _enabled:
        return
    _enabled = False
    for provider in _providers:
        with contextlib.suppress(Exception):
            provider.shutdown()
    _providers.clear()


# ── Logs → Loki ──────────────────────────────────────────────────────────────
def _attr_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    text = str(value)
    return text if len(text) <= 2000 else text[:2000] + "…"


def _export_log_event(event: dict[str, Any]) -> None:
    """structlog observer: forward one event to the OTel log pipeline."""
    if not _enabled or _otel_logger is None:
        return
    level = str(event.get("level", "info")).lower()
    attributes = {
        f"log.{k}" if k in {"name", "message"} else k: _attr_value(v)
        for k, v in event.items()
        if k not in _SKIP_KEYS and v is not None
    }
    attributes["event"] = str(event.get("event", ""))
    if event.get("exc_info") or event.get("exception"):
        attributes["exception"] = _attr_value(event.get("exception") or event.get("exc_info"))
    from opentelemetry._logs import SeverityNumber

    _otel_logger.emit(
        timestamp=time.time_ns(),
        severity_number=SeverityNumber(_SEVERITY.get(level, 9)),
        severity_text=level.upper(),
        body=str(event.get("event", "")),
        attributes=attributes,
    )


# ── Traces → Tempo ───────────────────────────────────────────────────────────
def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: _attr_value(v) for k, v in attrs.items() if v is not None}


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a span as the current span (works across ``await``).

    Exceptions are recorded on the span and re-raised. When telemetry is off
    this yields a no-op span.
    """
    if _trace is None:
        yield None
        return
    tracer = _trace.get_tracer("ai_employee")
    with tracer.start_as_current_span(
        name, attributes=_clean(attributes), record_exception=False, set_status_on_exception=False
    ) as sp:
        try:
            yield sp
        except BaseException as exc:
            # A user stop or an approval pause is a normal outcome, not an error.
            if type(exc).__name__ not in _NOT_ERRORS:
                sp.record_exception(exc)
                sp.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:300]))
            raise


def set_attrs(sp: Any, **attributes: Any) -> None:
    if sp is not None and _enabled:
        with contextlib.suppress(Exception):
            sp.set_attributes(_clean(attributes))


def mark_error(sp: Any, message: str) -> None:
    if sp is not None and _enabled:
        with contextlib.suppress(Exception):
            sp.set_status(Status(StatusCode.ERROR, message[:300]))


def current_trace_id() -> str | None:
    if _trace is None or not _enabled:
        return None
    ctx = _trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


# ── Metrics → Prometheus ─────────────────────────────────────────────────────
# Prometheus names after the collector's conversion are shown on the right.
_COUNTERS = {
    "tasks": ("ai_employee.tasks", "Finished task runs by mode/status"),  # _tasks_total
    "steps": ("ai_employee.steps", "Executed plan steps by agent/status"),
    "agent_runs": ("ai_employee.agent.runs", "Agent turns by agent/status"),
    "llm_calls": ("ai_employee.llm.calls", "LLM calls by provider/model/status"),
    "llm_tokens": ("ai_employee.llm.tokens", "LLM tokens by provider/model/type"),
    "tool_calls": ("ai_employee.tool.calls", "Tool calls by tool/status"),
    "http_requests": ("ai_employee.http.requests", "HTTP requests by route/method/status"),
}
_HISTOGRAMS = {  # unit "s" → *_seconds_bucket
    "task_duration": ("ai_employee.task.duration", "Task run duration"),
    "step_duration": ("ai_employee.step.duration", "Plan step duration"),
    "agent_duration": ("ai_employee.agent.duration", "Agent turn duration"),
    "llm_duration": ("ai_employee.llm.duration", "LLM call latency"),
    "tool_duration": ("ai_employee.tool.duration", "Tool call latency"),
    "http_duration": ("ai_employee.http.duration", "HTTP request latency"),
}


def _instrument(key: str) -> Any:
    inst = _instruments.get(key)
    if inst is None:
        from opentelemetry import metrics

        meter = metrics.get_meter("ai_employee")
        if key in _COUNTERS:
            name, desc = _COUNTERS[key]
            inst = meter.create_counter(name, description=desc)
        else:
            name, desc = _HISTOGRAMS[key]
            inst = meter.create_histogram(name, unit="s", description=desc)
        _instruments[key] = inst
    return inst


def count(key: str, value: int = 1, **labels: Any) -> None:
    if not _enabled or value <= 0:
        return
    with contextlib.suppress(Exception):
        _instrument(key).add(value, _clean(labels))


def observe(key: str, seconds: float, **labels: Any) -> None:
    if not _enabled:
        return
    with contextlib.suppress(Exception):
        _instrument(key).record(seconds, _clean(labels))


# ── HTTP middleware ──────────────────────────────────────────────────────────
class TelemetryMiddleware:
    """Pure-ASGI middleware: one span + request metrics per HTTP request.

    Labels use the matched route template (``/tasks/{task_id}``), not the raw
    path, so Prometheus label cardinality stays bounded.
    """

    _SKIP_PREFIXES = ("/static", "/favicon")

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if (
            not _enabled
            or scope.get("type") != "http"
            or scope.get("path", "").startswith(self._SKIP_PREFIXES)
        ):
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "GET")
        status = {"code": 500}

        async def _send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 500))
            await send(message)

        started = time.perf_counter()
        with span(f"HTTP {method}", **{"http.method": method, "url.path": scope.get("path")}) as sp:
            try:
                await self.app(scope, receive, _send)
            finally:
                route = getattr(scope.get("route"), "path", None) or "unmatched"
                code = status["code"]
                set_attrs(sp, **{"http.route": route, "http.status_code": code})
                if sp is not None:
                    with contextlib.suppress(Exception):
                        sp.update_name(f"{method} {route}")
                if code >= 500:
                    mark_error(sp, f"HTTP {code}")
                labels = {"route": route, "method": method, "status": str(code)}
                count("http_requests", **labels)
                observe("http_duration", time.perf_counter() - started, **labels)


__all__ = [
    "TelemetryMiddleware",
    "count",
    "current_trace_id",
    "is_enabled",
    "mark_error",
    "observe",
    "set_attrs",
    "setup_telemetry",
    "shutdown_telemetry",
    "span",
]
