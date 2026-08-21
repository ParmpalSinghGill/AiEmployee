"""Tool registry, sandbox and approval-gating tests."""

from __future__ import annotations

import pytest

from ai_employee.core.exceptions import ApprovalRequired


async def test_registry_lists_builtins(registry):
    names = registry.names()
    assert {"read_file", "write_file", "python_exec", "web_search"} <= set(names)


async def test_write_requires_approval(registry):
    with pytest.raises(ApprovalRequired):
        await registry.execute("write_file", {"path": "x.txt", "content": "hi"})


async def test_write_then_read_roundtrip(registry):
    await registry.execute("write_file", {"path": "rt.txt", "content": "payload"}, approved=True)
    res = await registry.execute("read_file", {"path": "rt.txt"})
    assert res.ok and res.output == "payload"


async def test_sandbox_blocks_escape(registry):
    res = await registry.execute("read_file", {"path": "../../../etc/passwd"})
    assert not res.ok
    assert "sandbox" in (res.error or "")


async def test_python_exec(registry):
    res = await registry.execute("python_exec", {"code": "print(6*7)"})
    assert res.ok
    assert res.output["stdout"].strip() == "42"


async def test_shell_read_only_not_gated(registry):
    res = await registry.execute("shell", {"command": "echo hi"})
    assert res.ok and res.output["stdout"].strip() == "hi"


async def test_shell_write_is_gated(registry):
    with pytest.raises(ApprovalRequired):
        await registry.execute("shell", {"command": "rm -rf something"})


async def test_json_query(registry):
    res = await registry.execute("json_query", {"data": {"a": [1, 2, 3]}, "path": "a[2]"})
    assert res.output == 3


async def test_csv_preview_reports_non_utf8_encoding(registry, settings):
    ws = settings.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    path = ws / "nbsp.csv"
    path.write_bytes(b"name,product\nA,Conference\xa0phone\n")
    res = await registry.execute("csv_preview", {"path": "nbsp.csv", "rows": 2})
    assert res.ok
    assert res.output["encoding"] in {"cp1252", "latin-1"}
    assert res.output["row_count"] == 1
    assert "phone" in res.output["rows"][0]["product"]


async def test_csv_preview_reports_utf8_encoding(registry, settings):
    ws = settings.workspace_dir
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "plain.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    res = await registry.execute("csv_preview", {"path": "plain.csv"})
    assert res.ok
    assert res.output["encoding"] in {"utf-8", "utf-8-sig"}


async def test_python_exec_multiline_and_same_interpreter(registry):
    res = await registry.execute(
        "python_exec",
        {"code": "import sys\nx = 1\nprint(sys.executable)\nprint(x + 41)"},
    )
    assert res.ok
    lines = [ln for ln in res.output["stdout"].splitlines() if ln.strip()]
    assert lines[-1] == "42"


async def test_python_exec_recovers_complete_raw_json(registry):
    res = await registry.execute(
        "python_exec",
        {"_raw": '{"code": "print(6*7)"}'},
    )
    assert res.ok
    assert res.output["stdout"].strip() == "42"


async def test_python_exec_truncated_raw_is_invalid_tool_args(registry):
    res = await registry.execute(
        "python_exec",
        {"_raw": '{"code": "import pandas as pd\\ndf = pd.read_csv(\'x.csv\''},
    )
    assert not res.ok
    assert "InvalidToolArgs" in (res.error or "")


def test_tool_spec_shape(registry):
    spec = registry.get("read_file").to_spec()
    assert spec.name == "read_file"
    assert spec.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_list_dir_recursive_replaces_level_by_level_walking(tmp_path) -> None:
    """Agents walked deep dataset trees one level at a time, tripping the duplicate
    guard. One recursive call now returns the whole tree."""
    from ai_employee.tools.builtins.filesystem import ListDirTool

    for split in ("train", "validation"):
        for cls in ("crazing", "scratches"):
            d = tmp_path / "NEU-DET" / split / "images" / cls
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"{cls}_{i}.jpg").write_bytes(b"x")

    tool = ListDirTool(tmp_path)

    flat = await tool.execute(path="NEU-DET")
    assert sorted(e["name"] for e in flat.output) == ["train", "validation"]

    rec = await tool.execute(path="NEU-DET", recursive=True)
    assert rec.metadata["count"] == 12
    assert all(e["type"] == "file" for e in rec.output)
    assert any(e["name"].startswith("train/images/crazing/") for e in rec.output)


@pytest.mark.asyncio
async def test_list_dir_truncation_points_at_python_exec(tmp_path) -> None:
    from ai_employee.tools.builtins.filesystem import ListDirTool

    d = tmp_path / "big"
    d.mkdir()
    for i in range(20):
        (d / f"f{i}.txt").write_bytes(b"x")

    out = await ListDirTool(tmp_path).execute(path="big", recursive=True, max_entries=5)
    assert len(out.output) == 5
    assert out.metadata["count"] == 20
    assert out.metadata["truncated"] is True
    # The agent must be told to compute, not to list again.
    assert "python_exec" in out.metadata["note"]


def test_duplicate_message_names_the_next_step() -> None:
    """A bare 'duplicate' left the model retrying near-identical calls."""
    from ai_employee.tools.base import ToolResult
    from ai_employee.tools.invocation_history import duplicate_payload

    prev = ToolResult(ok=True, output="listing")
    generic = duplicate_payload(prev)
    assert "materially different" in generic["message"]

    listing = duplicate_payload(prev, "list_dir")
    assert "python_exec" in listing["message"]
    assert "recursive=true" in listing["message"]

    hub = duplicate_payload(prev, "hf_search")
    assert "kaggle_search" in hub["message"]
