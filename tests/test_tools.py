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


def test_tool_spec_shape(registry):
    spec = registry.get("read_file").to_spec()
    assert spec.name == "read_file"
    assert spec.parameters["type"] == "object"
