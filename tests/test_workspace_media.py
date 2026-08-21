"""Workspace file serving + media sampling for the dashboard."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from ai_employee.api.workspace_files import (
    is_dataset_content,
    resolve_workspace_path,
    sample_media,
)


def test_resolve_workspace_path_relative_and_absolute(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    target = ws / "sessions" / "s1" / "clip.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"video")

    assert resolve_workspace_path(ws, "sessions/s1/clip.mp4") == target.resolve()
    assert resolve_workspace_path(ws, str(target)) == target.resolve()
    assert resolve_workspace_path(ws, "data/workspace/sessions/s1/clip.mp4") == target.resolve()


def test_resolve_workspace_path_rejects_escape(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    with pytest.raises(ValueError):
        resolve_workspace_path(ws, str(outside))
    with pytest.raises(ValueError):
        resolve_workspace_path(ws, "../secret.txt")


def test_sample_media_caps_when_more_than_five(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    folder = ws / "data" / "set"
    folder.mkdir(parents=True)
    for i in range(8):
        (folder / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    (folder / "clip.mp4").write_bytes(b"vid")

    out = sample_media(ws, [str(folder)])
    assert out["total"] == 9
    assert out["sampled"] is True
    assert out["shown"] == 3
    assert len(out["videos"]) + len(out["images"]) == 3


def test_sample_media_shows_all_when_few(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    vid = ws / "sessions" / "a" / "a.mp4"
    img = ws / "sessions" / "a" / "b.png"
    vid.parent.mkdir(parents=True)
    vid.write_bytes(b"v")
    img.write_bytes(b"i")

    out = sample_media(ws, [str(vid), str(img)])
    assert out["total"] == 2
    assert out["sampled"] is False
    assert len(out["videos"]) == 1
    assert len(out["images"]) == 1
    assert out["videos"][0]["url"].startswith("/ui/api/workspace/file?path=")


@pytest.fixture
async def client(tmp_path, monkeypatch):
    from ai_employee.api.app import create_app
    from ai_employee.config import get_settings

    ws = tmp_path / "workspace"
    ws.mkdir()
    s = get_settings()
    monkeypatch.setattr(s, "workspace_dir", ws)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, ws


async def test_workspace_file_endpoint(client) -> None:
    c, ws = client
    f = ws / "sessions" / "t" / "demo.mp4"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"fake-mp4")

    r = await c.get("/ui/api/workspace/file", params={"path": "sessions/t/demo.mp4"})
    assert r.status_code == 200
    assert r.content == b"fake-mp4"
    assert "video" in (r.headers.get("content-type") or "")


async def test_workspace_file_rejects_escape(client) -> None:
    c, ws = client
    outside = ws.parent / "outside.bin"
    outside.write_bytes(b"x")
    r = await c.get("/ui/api/workspace/file", params={"path": str(outside)})
    assert r.status_code == 400


async def test_workspace_media_endpoint_samples(client) -> None:
    c, ws = client
    folder = ws / "data" / "imgs"
    folder.mkdir(parents=True)
    for i in range(6):
        (folder / f"n{i}.png").write_bytes(b"png")

    r = await c.get(
        "/ui/api/workspace/media",
        params=[("paths", str(folder))],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 6
    assert body["sampled"] is True
    assert body["shown"] == 3
    assert len(body["images"]) == 3


def test_is_dataset_content_flags_download_buckets_and_splits() -> None:
    # Downloaded corpora live under the canonical buckets.
    assert is_dataset_content(("data", "coco", "cat.jpg")) is True
    assert is_dataset_content(("models", "yolo", "preview.png")) is True
    # Split directories mark dataset content wherever they are nested.
    assert is_dataset_content(("sessions", "s1", "repo", "train", "0001.jpg")) is True
    assert is_dataset_content(("sessions", "s1", "images", "0001.jpg")) is True
    # Generated output: workspace root, or a session dir that isn't a split.
    assert is_dataset_content(("accuracy_by_class.png",)) is False
    assert is_dataset_content(("sessions", "s1", "confusion_matrix.png")) is False


def test_sample_media_excludes_dataset_content(tmp_path: Path) -> None:
    """The unnamed sweep must show generated figures, never dataset samples."""
    ws = tmp_path / "workspace"
    # A downloaded image dataset: exactly what used to flood the Result tab.
    dataset = ws / "data" / "defects" / "train"
    dataset.mkdir(parents=True)
    for i in range(40):
        (dataset / f"sample{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")
    # Model-bucket art is also corpus content, not a result.
    (ws / "models" / "yolo").mkdir(parents=True)
    (ws / "models" / "yolo" / "arch.png").write_bytes(b"i")
    # What the run actually produced.
    ws.mkdir(exist_ok=True)
    (ws / "confusion_matrix.png").write_bytes(b"i")
    (ws / "accuracy_by_class.png").write_bytes(b"i")

    out = sample_media(ws, ["."], sample_when_over=60, sample_size=60, exclude_dataset_content=True)

    names = sorted(i["name"] for i in out["images"])
    assert names == ["accuracy_by_class.png", "confusion_matrix.png"]
    assert out["total"] == 2
    assert out["sampled"] is False


def test_sample_media_keeps_dataset_content_when_not_excluded(tmp_path: Path) -> None:
    """Explicitly-named paths still resolve — the filter is opt-in."""
    ws = tmp_path / "workspace"
    dataset = ws / "data" / "defects"
    dataset.mkdir(parents=True)
    (dataset / "sample.jpg").write_bytes(b"\xff\xd8\xff")

    out = sample_media(ws, ["data/defects"])
    assert out["total"] == 1
    assert out["images"][0]["name"] == "sample.jpg"


def test_deliverables_exclude_dataset_annotations(tmp_path: Path) -> None:
    """A detection dataset ships thousands of per-image .xml annotations.

    Those are inputs the run consumed; listing them as "files produced" buried the
    actual deliverables (the training script, README and metrics).
    """
    from ai_employee.api.workspace_files import collect_deliverables

    ws = tmp_path / "workspace"
    ann = ws / "data" / "metal-surface-defects" / "NEU-DET" / "validation" / "annotations"
    ann.mkdir(parents=True)
    for i in range(280, 340):
        (ann / f"scratches_{i}.xml").write_text("<annotation/>")
    # Model-bucket config files are corpus content too.
    (ws / "models" / "yolo").mkdir(parents=True)
    (ws / "models" / "yolo" / "config.json").write_text("{}")
    # What the run actually produced.
    (ws / "train_defect_model.py").write_text("print('train')")
    (ws / "README.md").write_text("# How to run")

    out = collect_deliverables(ws)

    assert sorted(f["path"] for f in out["files"]) == ["README.md", "train_defect_model.py"]
    assert out["total"] == 2


def test_deliverables_hide_internal_ledger(tmp_path: Path) -> None:
    """assets.json is framework bookkeeping, not a user-facing deliverable."""
    from ai_employee.api.workspace_files import collect_deliverables

    ws = tmp_path / "workspace"
    ws.mkdir(parents=True)
    (ws / "assets.json").write_text("{}")
    (ws / "report.md").write_text("# report")

    out = collect_deliverables(ws)
    assert [f["path"] for f in out["files"]] == ["report.md"]
