"""FastAPI application exposing the AI Employee over HTTP."""

from ai_employee.api.app import create_app

__all__ = ["create_app"]
