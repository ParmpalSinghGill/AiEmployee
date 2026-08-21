"""Workspace asset registry: reuse local models/datasets instead of re-downloading."""

from __future__ import annotations

from pathlib import Path

from ai_employee.observability.asset_registry import (
    AssetRegistry,
    build_local_inventory_context,
    classify_asset,
    derive_identifier,
)


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def test_classify_asset_uses_bucket_then_extension() -> None:
    assert classify_asset("models/bert/model.safetensors", "huggingface") == "model"
    assert classify_asset("data/coco/train.zip", "kaggle") == "dataset"
    assert classify_asset("sessions/s1/repo", "github") == "repo"
    assert classify_asset("sessions/s1/clip.mp4", "youtube") == "media"
    assert classify_asset("notes.txt", "http") == "file"


def test_derive_identifier_prefers_hub_repo_id() -> None:
    assert derive_identifier("https://huggingface.co/google/vit-base", "models/vit") == (
        "google/vit-base"
    )
    assert (
        derive_identifier("https://huggingface.co/datasets/detection/defects", "data/defects")
        == "detection/defects"
    )
    assert derive_identifier("https://github.com/ultralytics/yolov5", "x") == ("ultralytics/yolov5")
    # No recognizable hub → fall back to the filename.
    assert derive_identifier("https://example.com/a/b/weights.pt", "models/weights.pt") == (
        "weights.pt"
    )


def test_record_and_list_roundtrip(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "models").mkdir()
    weights = ws / "models" / "model.safetensors"
    weights.write_bytes(b"x" * 2048)

    reg = AssetRegistry(ws)
    rec = reg.record(path=weights, source="huggingface", url="https://huggingface.co/o/m")
    assert rec is not None
    assert rec.kind == "model"
    assert rec.bytes == 2048

    # A separate instance reads the same ledger — this is what makes it cross-task.
    listed = AssetRegistry(ws).list_assets()
    assert [a.path for a in listed] == ["models/model.safetensors"]
    assert listed[0].identifier == "o/m"


def test_find_by_url_is_normalized_and_verifies_disk(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    f = ws / "data.csv"
    f.write_bytes(b"a,b\n")
    reg = AssetRegistry(ws)
    reg.record(path=f, source="http", url="https://example.com/data.csv")

    # Trailing slash / scheme / www differences must still hit the cache.
    assert reg.find_by_url("http://example.com/data.csv/") is not None
    assert reg.find_by_url("https://www.example.com/data.csv") is not None
    assert reg.find_by_url("https://example.com/other.csv") is None

    # Deleted outside the app → reported missing, never handed to an agent.
    f.unlink()
    assert reg.find_by_url("https://example.com/data.csv") is None


def test_find_matches_on_terms(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "models").mkdir()
    a = ws / "models" / "defect-detector.pt"
    b = ws / "models" / "speech.pt"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    reg = AssetRegistry(ws)
    reg.record(path=a, source="huggingface", url="https://huggingface.co/acme/defect-detector")
    reg.record(path=b, source="huggingface", url="https://huggingface.co/acme/speech")

    hits = reg.find("defect detection model")
    assert hits and hits[0].path == "models/defect-detector.pt"
    assert reg.find("zzzz-nothing-matches") == []


def test_forget_missing_prunes_deleted_files(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    keep = ws / "keep.csv"
    drop = ws / "drop.csv"
    keep.write_bytes(b"x")
    drop.write_bytes(b"x")
    reg = AssetRegistry(ws)
    reg.record(path=keep, source="http", url="https://e.com/keep.csv")
    reg.record(path=drop, source="http", url="https://e.com/drop.csv")

    drop.unlink()
    assert reg.forget_missing() == 1
    assert [a.path for a in reg.list_assets(verify=False)] == ["keep.csv"]


def test_record_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    outside = tmp_path / "elsewhere.bin"
    outside.write_bytes(b"x")
    assert AssetRegistry(ws).record(path=outside, source="http") is None


def test_ledger_survives_corruption(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "assets.json").write_text("{not json")
    reg = AssetRegistry(ws)
    assert reg.list_assets() == []  # degrades to empty rather than raising

    f = ws / "new.csv"
    f.write_bytes(b"x")
    assert reg.record(path=f, source="http", url="https://e.com/new.csv") is not None
    assert len(reg.list_assets()) == 1


def test_inventory_context_instructs_reuse(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "models").mkdir()
    m = ws / "models" / "vit.safetensors"
    m.write_bytes(b"x" * 1024)
    reg = AssetRegistry(ws)
    reg.record(path=m, source="huggingface", url="https://huggingface.co/google/vit-base")

    ctx = build_local_inventory_context(reg)
    assert "google/vit-base" in ctx
    assert "models/vit.safetensors" in ctx
    # The instruction is the point — a bare list gets ignored by the model.
    assert "BEFORE searching" in ctx
    assert "do NOT download it again" in ctx


def test_inventory_context_empty_when_nothing_local(tmp_path: Path) -> None:
    assert build_local_inventory_context(AssetRegistry(_ws(tmp_path))) == ""


def test_record_accepts_cwd_relative_path(tmp_path: Path, monkeypatch) -> None:
    """Download helpers return cwd-relative paths like ``data/workspace/models/x``.

    Joining those onto the workspace doubled the prefix and silently dropped the
    asset from the ledger, so every task re-downloaded it.
    """
    monkeypatch.chdir(tmp_path)
    ws = Path("workspace")
    (ws / "models").mkdir(parents=True)
    m = ws / "models" / "vit.safetensors"
    m.write_bytes(b"x" * 16)

    reg = AssetRegistry(ws)
    rec = reg.record(path=m, source="huggingface", url="https://huggingface.co/a/vit")

    assert rec is not None
    assert rec.path == "models/vit.safetensors"
    assert [a.path for a in reg.list_assets()] == ["models/vit.safetensors"]


def test_record_accepts_workspace_relative_path(tmp_path: Path) -> None:
    """The other real caller shape: a path already relative to the workspace."""
    ws = _ws(tmp_path)
    (ws / "models").mkdir()
    (ws / "models" / "vit.safetensors").write_bytes(b"x")

    reg = AssetRegistry(ws)
    rec = reg.record(path="models/vit.safetensors", source="huggingface")
    assert rec is not None
    assert rec.path == "models/vit.safetensors"
