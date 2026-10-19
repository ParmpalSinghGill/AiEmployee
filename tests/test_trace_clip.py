"""Explain-trace truncation is configurable (EXPLAIN_MAX_CHARS / EXPLAIN_TOOL_MAX_CHARS)."""

from __future__ import annotations

from ai_employee.config import get_settings
from ai_employee.core import trace


def test_clip_default_keeps_everything() -> None:
    # Out of the box: no truncation. The whole point — a step task or a long
    # LLM answer must be readable in full in the Explain view.
    assert get_settings().explain_max_chars == 0
    assert get_settings().explain_tool_max_chars == 0
    long = "x" * 200_000
    assert trace.clip(long) == long
    assert "more chars" not in trace.clip(long)


def test_clip_truncates_over_limit_with_marker(monkeypatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "explain_max_chars", 10, raising=False)
    out = trace.clip("abcdefghijklmnop")
    assert out.startswith("abcdefghij")
    assert "more chars" in out
    assert "EXPLAIN_MAX_CHARS" in out  # tells the user how to see more


def test_clip_zero_disables_truncation(monkeypatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "explain_max_chars", 0, raising=False)
    text = "y" * 200_000
    assert trace.clip(text) == text


def test_explicit_limit_still_wins() -> None:
    assert trace.clip("abcdef", 3) == "abc… (+3 more chars — raise EXPLAIN_MAX_CHARS to see more)"


def test_tool_clip_uses_tool_limit(monkeypatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "explain_tool_max_chars", 5, raising=False)
    out = trace.tool_clip("abcdefghij")
    assert out.startswith("abcde")
    assert "more chars" in out


def test_clip_obj_recurses_and_caps_lists(monkeypatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "explain_max_chars", 4, raising=False)
    out = trace.clip_obj({"a": "abcdefgh", "b": ["12345678", "ok"]})
    assert out["a"].startswith("abcd")
    assert out["b"][0].startswith("1234")
    assert out["b"][1] == "ok"
