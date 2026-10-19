"""Built-in tools, grouped by category. ``build_builtin_tools`` returns the full
default set, ready to register."""

from __future__ import annotations

from ai_employee.config import Settings
from ai_employee.tools.base import Tool


def build_builtin_tools(settings: Settings) -> list[Tool]:
    from ai_employee.tools.builtins.code import PipInstallTool, PythonExecTool, ShellTool
    from ai_employee.tools.builtins.data import CsvPreviewTool, JsonQueryTool
    from ai_employee.tools.builtins.filesystem import (
        ListDirTool,
        MakeDirTool,
        ReadFileTool,
        WriteFileTool,
    )
    from ai_employee.tools.builtins.google_workspace import build_google_tools
    from ai_employee.tools.builtins.research import build_research_tools
    from ai_employee.tools.builtins.resource_manager import build_resource_manager_tools
    from ai_employee.tools.builtins.resources import build_resource_tools
    from ai_employee.tools.builtins.web import (
        FetchPageTool,
        HttpRequestTool,
        WebSearchTool,
    )

    workspace = settings.workspace_dir
    tools: list[Tool] = [
        ReadFileTool(workspace),
        WriteFileTool(workspace),
        ListDirTool(workspace),
        MakeDirTool(workspace),
        CsvPreviewTool(workspace),
        JsonQueryTool(),
        HttpRequestTool(),
        FetchPageTool(),
        WebSearchTool(settings),
    ]
    tools.extend(build_research_tools(settings))
    tools.extend(build_resource_tools(settings))
    tools.extend(build_resource_manager_tools(settings))
    tools.extend(build_google_tools(settings))  # empty until the OAuth client is set up
    if settings.enable_code_execution:
        tools.append(PythonExecTool(workspace))
        tools.append(ShellTool(workspace))
        if settings.allow_package_install:
            tools.append(PipInstallTool(workspace))
    return tools


__all__ = ["build_builtin_tools"]
