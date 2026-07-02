"""Credit / balance checks for managed LLM keys."""

from __future__ import annotations

import pytest

from ai_employee.config.credits_checker import check_brain_credits


@pytest.mark.asyncio
async def test_brain_credits_skips_local_and_mock() -> None:
    from ai_employee.config import get_settings

    s = get_settings()
    for prov in ("local", "mock"):
        res = await check_brain_credits(s, prov)
        assert res["supported"] is False


@pytest.mark.asyncio
async def test_brain_credits_no_key() -> None:
    from ai_employee.config import get_settings

    s = get_settings()
    s.deepseek_api_key = ""
    res = await check_brain_credits(s, "deepseek")
    assert res["ok"] is False
    assert "No key" in res["message"]
