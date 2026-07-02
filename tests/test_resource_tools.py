"""Tests for resource download helpers and tools."""

from __future__ import annotations

import pytest

from ai_employee.tools.builtins.resource_download import detect_source


def test_detect_source_youtube() -> None:
    assert detect_source("https://www.youtube.com/watch?v=abc") == "youtube"
    assert detect_source("https://youtu.be/abc") == "youtube"


def test_detect_source_huggingface() -> None:
    assert detect_source("https://huggingface.co/datasets/foo/bar") == "huggingface"


def test_detect_source_kaggle() -> None:
    assert detect_source("https://www.kaggle.com/datasets/user/ds") == "kaggle"


def test_detect_source_github() -> None:
    assert detect_source("https://github.com/org/repo") == "github"


def test_detect_source_http_fallback() -> None:
    assert detect_source("https://example.com/file.zip") == "http"


@pytest.mark.asyncio
async def test_request_resource_input_requires_approval() -> None:
    from ai_employee.config import get_settings
    from ai_employee.core.exceptions import ApprovalRequired
    from ai_employee.tools import default_registry

    reg = default_registry(get_settings())
    with pytest.raises(ApprovalRequired) as exc:
        await reg.execute(
            "request_resource_input",
            {"resource_description": "mall video", "candidate_urls": ["https://x.com"]},
        )
    assert exc.value.operation == "resource_acquisition"


@pytest.mark.asyncio
async def test_request_resource_input_with_decision() -> None:
    from ai_employee.observability.resource_ctx import (
        bind_resource_decision,
        clear_resource_decision,
    )
    from ai_employee.tools.builtins.resources import RequestResourceInputTool

    bind_resource_decision({"decision": "provide_local_path", "local_path": "./data/v.mp4"})
    try:
        tool = RequestResourceInputTool()
        res = await tool.execute(resource_description="video")
        assert res.ok
        assert res.output["human_decision"]["local_path"] == "./data/v.mp4"
    finally:
        clear_resource_decision()


def test_resource_acquirer_in_registry() -> None:
    from ai_employee.agents import build_agent_registry

    reg = build_agent_registry()
    assert "resource_acquirer" in reg


def test_download_resource_tool_registered() -> None:
    from ai_employee.config import get_settings
    from ai_employee.tools import default_registry

    reg = default_registry(get_settings())
    assert "download_resource" in reg.names()
    assert "hub_search_all" in reg.names()
    assert "youtube_search" in reg.names()
