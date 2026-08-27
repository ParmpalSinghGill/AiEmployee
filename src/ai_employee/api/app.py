"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai_employee.api.deps import shutdown_employee
from ai_employee.api.routes import router
from ai_employee.api.ui import router as ui_router
from ai_employee.config import get_settings
from ai_employee.core.logging import configure_logging, get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, json=s.log_json)
    log.info("api.startup", provider=s.llm_provider.value, env=s.environment.value)
    # Resume any run left mid-flight by a previous shutdown (e.g. an approval that
    # was queued but never executed) so it isn't silently orphaned in 'pending'.
    try:
        from ai_employee.api.deps import get_employee
        from ai_employee.observability import get_task_manager

        resumed = get_task_manager().resume_interrupted(get_employee())
        if resumed:
            log.info("api.resumed_tasks", count=len(resumed))
    except Exception as exc:  # noqa: BLE001 - startup must not fail on resume issues
        log.warning("api.resume_failed", error=str(exc))
    # Scheduled tasks & watchers (⏰): fire due schedules through the TaskManager.
    runner = None
    try:
        from ai_employee.api.deps import get_employee
        from ai_employee.observability import get_task_manager
        from ai_employee.observability.schedules import SchedulerRunner

        runner = SchedulerRunner(s, get_employee, get_task_manager())
        runner.start()
    except Exception as exc:  # noqa: BLE001
        log.warning("api.scheduler_failed", error=str(exc))
    yield
    if runner is not None:
        await runner.stop()
    await shutdown_employee()
    log.info("api.shutdown")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_name,
        version="0.1.0",
        description="Provider-independent, multi-agent AI Employee framework.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(ui_router)

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/ui")

    @app.get("/api", tags=["system"])
    async def api_info() -> dict:
        return {
            "name": s.app_name,
            "version": "0.1.0",
            "docs": "/docs",
            "ui": "/ui",
            "endpoints": [
                "/chat",
                "/task",
                "/agents",
                "/tools",
                "/memory",
                "/knowledge/ingest",
                "/knowledge/query",
                "/workflows",
                "/config",
            ],
        }

    return app


app = create_app()
