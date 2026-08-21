"""Resource find vs download workflow — WAITING_FOR_USER status."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_employee.observability.tasks import TaskStatus
from ai_employee.verification.resources import (
    audit_resources,
    build_waiting_for_user_payload,
    format_waiting_for_user_message,
    is_manual_download_error,
)


def test_is_manual_download_error_detects_403() -> None:
    assert is_manual_download_error("HTTP 403 Forbidden")
    assert is_manual_download_error("Cloudflare challenge")
    assert not is_manual_download_error("file not found in workspace")


def test_audit_waiting_for_user_when_download_fails_with_url(tmp_path: Path) -> None:
    url = "https://example.com/data/people.mp4"
    audit = audit_resources(
        task="Find and download mall entrance footage",
        success_criteria="Resource located and saved locally if possible",
        output=f"Found candidate video at {url} but download failed.",
        tool_trace=[
            {
                "tool": "download_resource",
                "ok": False,
                "error": "HTTP 403 Forbidden",
                "args": {"url": url, "destination": "./data/video.mp4"},
            }
        ],
        workspace=tmp_path,
    )
    assert audit["outcome"] == "waiting_for_user"
    assert audit["has_urls"] is True
    assert not audit["has_verified_download"]
    assert not any("no file exists" in i.lower() for i in audit["issues"])


def test_audit_completed_when_file_exists(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    audit = audit_resources(
        task="Download entrance footage",
        success_criteria="File saved locally",
        output=f"Saved to {video.name}",
        tool_trace=[
            {
                "tool": "download_resource",
                "ok": True,
                "args": {"url": "https://example.com/a.mp4"},
                "output": {"path": str(video)},
            }
        ],
        workspace=tmp_path,
    )
    assert audit["outcome"] == "completed"
    assert audit["has_verified_download"] is True


def test_audit_failed_when_claim_without_url(tmp_path: Path) -> None:
    audit = audit_resources(
        task="Find a dataset video",
        success_criteria="A verified URL",
        output="I successfully found and saved the footage.",
        tool_trace=[],
        workspace=tmp_path,
    )
    assert audit["outcome"] == "failed"
    assert audit["issues"]


def test_build_waiting_for_user_payload() -> None:
    audit = {
        "resources": [
            {
                "name": "people.mp4",
                "web_url": "https://x.test/v.mp4",
                "local_path": "NOT DOWNLOADED",
            }
        ],
        "download_failures": [{"url": "https://x.test/v.mp4", "error": "HTTP 403"}],
    }
    payload = build_waiting_for_user_payload(task="mall video", resource_audit=audit)
    assert payload["operation"] == "waiting_for_user"
    assert "WAITING_FOR_USER" in payload["message"]
    assert payload["candidate_urls"]


def test_format_waiting_for_user_message_includes_paths() -> None:
    msg = format_waiting_for_user_message(
        name="People counter sample",
        url="https://example.com/sample.mp4",
        reason="HTTP 403 Forbidden",
    )
    assert "WAITING_FOR_USER" in msg
    assert "C:\\Videos" in msg
    assert "/home/user" in msg


@pytest.mark.asyncio
async def test_request_resource_input_waiting_for_user_with_candidates() -> None:
    from ai_employee.config import get_settings
    from ai_employee.core.exceptions import ApprovalRequired
    from ai_employee.tools import default_registry

    reg = default_registry(get_settings())
    with pytest.raises(ApprovalRequired) as exc:
        await reg.execute(
            "request_resource_input",
            {
                "resource_description": "mall CCTV footage",
                "candidate_urls": ["https://example.com/video.mp4"],
                "download_failure_reason": "HTTP 403 Forbidden",
            },
        )
    assert exc.value.operation == "waiting_for_user"
    assert "WAITING_FOR_USER" in exc.value.payload["message"]


@pytest.mark.asyncio
async def test_request_resource_input_no_candidates_uses_resource_acquisition() -> None:
    from ai_employee.config import get_settings
    from ai_employee.core.exceptions import ApprovalRequired
    from ai_employee.tools import default_registry

    reg = default_registry(get_settings())
    with pytest.raises(ApprovalRequired) as exc:
        await reg.execute(
            "request_resource_input",
            {"resource_description": "mall CCTV footage", "candidate_urls": []},
        )
    assert exc.value.operation == "resource_acquisition"


def test_task_status_includes_waiting_for_user() -> None:
    assert TaskStatus.waiting_for_user.value == "waiting_for_user"


def test_sanitize_url_strips_backticks() -> None:
    from ai_employee.verification.resources import sanitize_url

    assert sanitize_url("`https://www.youtube.com/watch?v=abc`") == (
        "https://www.youtube.com/watch?v=abc"
    )
    assert sanitize_url("https://example.com/v.mp4,") == "https://example.com/v.mp4"


def test_prioritize_download_urls_youtube_first() -> None:
    from ai_employee.verification.resources import prioritize_download_urls

    urls = [
        "https://www.videezy.com/free-video/shop-entrance",
        "https://www.youtube.com/watch?v=gAuJlwnUqMs",
        "https://huggingface.co/Ultralytics/YOLO26",
    ]
    ordered = prioritize_download_urls(urls)
    assert ordered[0].startswith("https://www.youtube.com")
    assert "huggingface.co" in ordered[1]


def test_audit_text_waiting_for_user_without_download_is_found_not_downloaded(
    tmp_path: Path,
) -> None:
    """Agent prose must not trigger waiting_for_user without a real download attempt."""
    url = "https://www.youtube.com/watch?v=gAuJlwnUqMs"
    audit = audit_resources(
        task="Download mall entrance footage",
        success_criteria="Video saved locally",
        output=(f"Found {url}. Status: waiting_for_user because automatic download is blocked."),
        tool_trace=[],
        workspace=tmp_path,
    )
    assert audit["outcome"] == "found_not_downloaded"
    assert audit["download_attempted"] is False
