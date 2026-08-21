"""Tests for resource download helpers and tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_employee.tools.builtins.resource_download import (
    build_ytdlp_command,
    detect_source,
    youtube_needs_cookies,
)


def test_detect_source_youtube() -> None:
    assert detect_source("https://www.youtube.com/watch?v=abc") == "youtube"
    assert detect_source("https://youtu.be/abc") == "youtube"


def test_detect_source_huggingface() -> None:
    assert detect_source("https://huggingface.co/datasets/foo/bar") == "huggingface"
    assert detect_source("https://huggingface.co/models/foo/bar") == "huggingface"
    assert detect_source("https://huggingface.co/Ultralytics/YOLO26") == "huggingface"


@pytest.mark.asyncio
async def test_download_huggingface_automated(tmp_path: Path, monkeypatch) -> None:
    from ai_employee.tools.builtins.resource_download import download_huggingface

    captured: dict = {}

    def fake_snapshot(**kwargs: object) -> str:
        captured.update(kwargs)
        dest = Path(str(kwargs["local_dir"]))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.yaml").write_text("ok")
        return str(dest)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    ws = tmp_path / "workspace"
    ws.mkdir()
    dest = ws / "models" / "yolo26"
    info = await download_huggingface(
        "https://huggingface.co/Ultralytics/YOLO26",
        dest,
        workspace=ws,
        token="hf_test",
    )
    assert info["repo_id"] == "Ultralytics/YOLO26"
    assert info["repo_type"] == "model"
    assert captured["repo_id"] == "Ultralytics/YOLO26"
    assert captured["token"] == "hf_test"
    assert dest.is_dir()


def test_parse_huggingface_url() -> None:
    from ai_employee.tools.builtins.resource_download import parse_huggingface_url

    assert parse_huggingface_url("https://huggingface.co/Ultralytics/YOLO26") == (
        "model",
        "Ultralytics/YOLO26",
    )
    assert parse_huggingface_url("https://huggingface.co/datasets/foo/bar") == (
        "dataset",
        "foo/bar",
    )


def test_detect_source_kaggle() -> None:
    assert detect_source("https://www.kaggle.com/datasets/user/ds") == "kaggle"


def test_detect_source_github() -> None:
    assert detect_source("https://github.com/org/repo") == "github"


def test_detect_source_http_fallback() -> None:
    assert detect_source("https://example.com/file.zip") == "http"


def test_sanitize_url() -> None:
    from ai_employee.tools.builtins.resource_download import sanitize_url

    assert sanitize_url("https://youtu.be/abc`") == "https://youtu.be/abc"


def test_prioritize_download_urls() -> None:
    from ai_employee.tools.builtins.resource_download import prioritize_download_urls

    urls = prioritize_download_urls(
        [
            "https://pixabay.com/videos/foo",
            "https://www.youtube.com/watch?v=abc",
            "https://github.com/org/repo",
            "https://www.kaggle.com/datasets/user/ds",
        ]
    )
    assert urls[0].startswith("https://www.youtube.com")
    assert "github.com" in urls[1]
    assert "kaggle.com" in urls[2]


def test_build_ytdlp_command_adds_browser_cookies() -> None:
    cmd = build_ytdlp_command(
        "https://www.youtube.com/watch?v=abc",
        "./out.%(ext)s",
        cookies_from_browser="chrome",
    )
    assert "--cookies-from-browser" in cmd
    assert "chrome" in cmd
    assert cmd[-1].startswith("https://")


def test_build_ytdlp_command_adds_cookies_file() -> None:
    cmd = build_ytdlp_command(
        "https://www.youtube.com/watch?v=abc",
        "./out.%(ext)s",
        cookies_file="/tmp/cookies.txt",
    )
    assert "--cookies" in cmd
    assert "/tmp/cookies.txt" in cmd


def test_youtube_needs_cookies_detects_age_gate() -> None:
    err = "Sign in to confirm your age. Use --cookies-from-browser"
    assert youtube_needs_cookies(err)
    assert not youtube_needs_cookies("file not found")


def test_inspect_youtube_cookies_file_not_logged_in(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.resource_download import inspect_youtube_cookies_file

    cf = tmp_path / "cookies.txt"
    cf.write_text(
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t0\t__Secure-3PSID\tvalue\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tPREF\ten\n"
    )
    status = inspect_youtube_cookies_file(cf)
    assert status["exists"] is True
    assert status["logged_in"] is False


def test_inspect_youtube_cookies_file_logged_in(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.resource_download import inspect_youtube_cookies_file

    cf = tmp_path / "cookies.txt"
    cf.write_text(
        ".youtube.com\tTRUE\t/\tTRUE\t0\tLOGIN_INFO\tvalue\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tvalue\n"
    )
    assert inspect_youtube_cookies_file(cf)["logged_in"] is True


def test_cookie_sources_skips_invalid_file_uses_browser(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.resource_download import _cookie_sources

    cf = tmp_path / "cookies.txt"
    cf.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tPREF\ten\n")
    sources = _cookie_sources("chrome", cf)
    assert ("file", cf) not in [(s, c) for s, c in sources]
    assert ("browser", "chrome") in sources


def test_cookie_sources_uses_logged_in_file(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.resource_download import _cookie_sources

    cf = tmp_path / "cookies.txt"
    cf.write_text(
        ".youtube.com\tTRUE\t/\tTRUE\t0\tLOGIN_INFO\tx\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tx\n"
        ".youtube.com\tTRUE\t/\tFALSE\t0\tSAPISID\tx\n"
    )
    sources = _cookie_sources(None, cf)
    assert sources[0] == ("file", cf)


def test_is_manual_download_error_detects_age_restriction() -> None:
    from ai_employee.verification.resources import is_manual_download_error

    assert is_manual_download_error("Sign in to confirm your age")


def test_safe_dest_strips_duplicate_workspace_prefix(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.resource_download import _safe_dest

    ws = tmp_path / "data" / "workspace"
    ws.mkdir(parents=True)
    dest = _safe_dest(ws, "data/workspace/videos/clip.mp4")
    assert dest == (ws / "videos" / "clip.mp4").resolve()


def test_canonical_artifact_dest_routes_into_three_workspace_buckets(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.resource_download import canonical_artifact_dest

    ws = tmp_path / "data" / "workspace"
    ws.mkdir(parents=True)

    model_dest = canonical_artifact_dest(
        ws,
        "sess-123",
        url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        source_hint="http",
        requested_dest="./models/yolo26n.pt",
    )
    data_dest = canonical_artifact_dest(
        ws,
        "sess-123",
        url="https://www.kaggle.com/datasets/user/pets2009",
        source_hint="kaggle",
        requested_dest="./data/pets2009",
    )
    session_dest = canonical_artifact_dest(
        ws,
        "sess-123",
        url="https://www.youtube.com/watch?v=abc123",
        source_hint="youtube",
        requested_dest="./session/mall.mp4",
    )

    assert model_dest == (ws / "models" / "yolo26n.pt").resolve()
    assert data_dest == (ws / "data" / "pets2009").resolve()
    assert session_dest == (ws / "sessions" / "sess-123" / "mall.mp4").resolve()


@pytest.mark.asyncio
async def test_download_resource_tool_canonicalizes_session_destination(
    tmp_path: Path, monkeypatch
) -> None:
    from ai_employee.config import get_settings
    from ai_employee.observability.resource_ctx import (
        bind_resource_session,
        clear_resource_decision,
    )
    from ai_employee.tools.builtins.resources import DownloadResourceTool

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "workspace"
    settings.workspace_dir.mkdir(parents=True)

    seen: dict[str, Path] = {}

    async def fake_download_http(url: str, dest: Path, *, timeout: int = 120) -> dict[str, object]:
        seen["dest"] = dest
        dest.write_bytes(b"x")  # noqa: ASYNC240 — tiny write in a test fake
        return {"path": str(dest), "bytes": 1, "source": "http"}

    monkeypatch.setattr("ai_employee.tools.builtins.resources.download_http", fake_download_http)

    bind_resource_session("sess-abc")
    try:
        tool = DownloadResourceTool(settings)
        res = await tool.execute(
            url="https://example.com/files/entrance.mp4",
            destination="./session/entrance.mp4",
            source_hint="http",
        )
    finally:
        clear_resource_decision()

    assert res.ok
    expected = settings.workspace_dir / "sessions" / "sess-abc" / "entrance.mp4"
    assert seen["dest"] == expected.resolve()
    assert res.metadata["local_path"] == str(expected.resolve())


@pytest.mark.asyncio
async def test_audit_partial_when_some_downloads_succeed(tmp_path: Path) -> None:
    from ai_employee.verification.resources import audit_resources

    url_ok = "https://www.youtube.com/watch?v=ok"
    url_fail = "https://www.youtube.com/watch?v=fail"
    video = tmp_path / "ok.mp4"
    video.write_bytes(b"x")
    audit = audit_resources(
        task="Download two YouTube clips",
        success_criteria="Files saved locally if possible",
        output=f"Downloaded {url_ok}; {url_fail} failed (age-restricted).",
        tool_trace=[
            {
                "tool": "download_resource",
                "ok": True,
                "args": {"url": url_ok, "destination": "./ok.mp4"},
                "output": {"path": str(video)},
            },
            {
                "tool": "download_resource",
                "ok": False,
                "error": "Sign in to confirm your age",
                "args": {"url": url_fail, "destination": "./fail.mp4"},
            },
        ],
        workspace=tmp_path,
    )
    assert audit["outcome"] == "partial"
    assert audit["has_verified_download"] is True


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
    assert exc.value.operation == "waiting_for_user"


@pytest.mark.asyncio
async def test_request_resource_input_skips_when_preapproved() -> None:
    from ai_employee.config import get_settings
    from ai_employee.observability.resource_ctx import bind_approved_ops
    from ai_employee.tools import default_registry

    bind_approved_ops({"resource_acquisition", "waiting_for_user"})
    reg = default_registry(get_settings())
    result = await reg.execute(
        "request_resource_input",
        {
            "resource_description": "mall video",
            "candidate_urls": ["https://www.youtube.com/watch?v=abc"],
        },
    )
    assert result.ok
    assert result.output["human_decision"] == "proceed_without"


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


def test_research_agent_has_download_tools() -> None:
    from ai_employee.agents import build_agent_registry

    reg = build_agent_registry()
    research = reg.get("research")
    allowed = set(research.allowed_tools or [])
    assert "download_resource" in allowed
    assert "request_resource_input" in allowed
    assert "hub_search_all" in allowed


def test_download_resource_tool_registered() -> None:
    from ai_employee.config import get_settings
    from ai_employee.tools import default_registry

    reg = default_registry(get_settings())
    assert "download_resource" in reg.names()
    assert "hub_search_all" in reg.names()
    assert "youtube_search" in reg.names()


def test_hub_search_all_tabular_skips_video_hubs() -> None:
    from ai_employee.tools.builtins.resources import hubs_for_kind

    tabular = hubs_for_kind("tabular")
    assert "kaggle" in tabular and "huggingface" in tabular
    assert "youtube" not in tabular and "roboflow" not in tabular
    video = hubs_for_kind("video")
    assert "youtube" in video and "kaggle" not in video
    image = hubs_for_kind("image")
    assert "roboflow" in image and "youtube" not in image
    speech = hubs_for_kind("speech")
    assert "huggingface" in speech and "youtube" not in speech and "roboflow" not in speech


@pytest.mark.asyncio
async def test_download_resource_reuses_cached_copy_across_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    """A URL fetched once must never be downloaded again — this is the bandwidth saving."""
    from ai_employee.config import get_settings
    from ai_employee.observability.resource_ctx import bind_resource_session
    from ai_employee.tools.builtins.resources import DownloadResourceTool

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "workspace"
    settings.workspace_dir.mkdir(parents=True)

    calls = {"n": 0}

    async def fake_download_http(url: str, dest: Path, *, timeout: int = 120) -> dict[str, object]:
        calls["n"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"payload")  # noqa: ASYNC240 - test fixture, sync fs is fine
        return {"path": str(dest), "bytes": 7, "source": "http"}

    monkeypatch.setattr("ai_employee.tools.builtins.resources.download_http", fake_download_http)

    url = "https://example.com/files/corpus.zip"
    tool = DownloadResourceTool(settings)

    bind_resource_session("sess-one")
    first = await tool.execute(url=url, destination="./corpus.zip", approved=True)
    assert first.ok
    assert calls["n"] == 1

    # A different session — i.e. a brand-new task — must hit the cache, not the network.
    bind_resource_session("sess-two")
    second = await tool.execute(url=url, destination="./corpus.zip", approved=True)
    assert second.ok
    assert calls["n"] == 1, "second download re-fetched the same URL"
    assert second.output["cached"] is True
    assert second.metadata["cache_hit"] is True
    assert Path(second.output["path"]).exists()  # noqa: ASYNC240 - test fixture, sync fs is fine


@pytest.mark.asyncio
async def test_download_resource_refetches_when_local_copy_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    """The ledger is an index, not the truth — a deleted file must download again."""
    from ai_employee.config import get_settings
    from ai_employee.observability.resource_ctx import bind_resource_session
    from ai_employee.tools.builtins.resources import DownloadResourceTool

    settings = get_settings().model_copy(deep=True)
    settings.workspace_dir = tmp_path / "workspace"
    settings.workspace_dir.mkdir(parents=True)

    calls = {"n": 0}

    async def fake_download_http(url: str, dest: Path, *, timeout: int = 120) -> dict[str, object]:
        calls["n"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"payload")  # noqa: ASYNC240 - test fixture, sync fs is fine
        return {"path": str(dest), "bytes": 7, "source": "http"}

    monkeypatch.setattr("ai_employee.tools.builtins.resources.download_http", fake_download_http)

    url = "https://example.com/files/corpus.zip"
    tool = DownloadResourceTool(settings)
    bind_resource_session("sess-one")

    first = await tool.execute(url=url, destination="./corpus.zip", approved=True)
    assert first.ok and calls["n"] == 1

    Path(first.output["path"]).unlink()  # noqa: ASYNC240 - test fixture, sync fs is fine

    again = await tool.execute(url=url, destination="./corpus.zip", approved=True)
    assert again.ok
    assert calls["n"] == 2, "deleted file should have been re-downloaded"
