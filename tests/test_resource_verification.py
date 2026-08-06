"""Resource verification for find/download tasks."""

from __future__ import annotations

from pathlib import Path

from ai_employee.verification.resources import audit_resources, is_resource_task


def test_is_resource_task_detects_video_goal() -> None:
    assert is_resource_task("Find a CCTV video of people at a gate")
    assert not is_resource_task("Summarize async Python")


def test_is_resource_task_ignores_data_cleaning() -> None:
    assert not is_resource_task(
        "Load the dataset, perform thorough data quality checks, clean the data, "
        "and save a cleaned version to the workspace."
    )
    assert not is_resource_task(
        "Clean the dataset: handle missing values and save a cleaned version.",
        "A cleaned dataset file is saved and cleaning steps are documented.",
    )
    assert is_resource_task(
        "Acquire the Superstore Sales Dataset from https://kaggle.com/x and download it."
    )


def test_audit_flags_claim_without_url_or_file(tmp_path: Path) -> None:
    audit = audit_resources(
        task="Find and download a mall entrance video",
        success_criteria="A verified video URL or local file",
        output="I successfully found and saved the requested footage.",
        tool_trace=[],
        workspace=tmp_path,
    )
    assert audit["applies"] is True
    assert audit["issues"]
    assert audit["claims_found"] is True
    assert audit["has_urls"] is False


def test_audit_accepts_existing_local_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    audit = audit_resources(
        task="Download entrance footage",
        success_criteria="File saved locally",
        output=f"Saved video to {video.name}",
        tool_trace=[{"tool": "write_file", "ok": True, "args": {"path": video.name}}],
        workspace=tmp_path,
    )
    assert audit["has_verified_download"] is True
    assert not audit["issues"]


def test_audit_requires_url_verification_when_criteria_say_so(tmp_path: Path) -> None:
    audit = audit_resources(
        task="Find a dataset video",
        success_criteria="URL is identified and verified",
        output="Candidate: https://example.com/data.mp4",
        tool_trace=[{"tool": "web_search", "ok": True, "args": {"query": "dataset"}}],
        workspace=tmp_path,
    )
    assert any("verification" in i.lower() for i in audit["issues"])
