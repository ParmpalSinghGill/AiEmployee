"""UI preference persistence for brain and coding defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_employee.config import get_settings
from ai_employee.config.ui_preferences import (
    apply_ui_preferences,
    read_preferences,
    save_brain_default,
    save_coding_default,
)


@pytest.fixture
def prefs_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ui_preferences.json"
    s = get_settings()
    monkeypatch.setattr(s, "ui_preferences_file", path)
    return path


def test_save_and_apply_brain_default(prefs_file: Path) -> None:
    s = get_settings()
    save_brain_default(s, "mock", "mock-1")
    assert prefs_file.exists()
    saved = read_preferences(s)
    assert saved["brain"] == {"provider": "mock", "model": "mock-1"}

    s.llm_provider = "deepseek"  # type: ignore[assignment]
    apply_ui_preferences(s)
    assert s.llm_provider.value == "mock"


def test_save_coding_default(prefs_file: Path) -> None:
    s = get_settings()
    save_coding_default(s, "cursor")
    saved = read_preferences(s)
    assert saved["coding_backend"] == "cursor"
    s.coding_backend = "brain"
    apply_ui_preferences(s)
    assert s.coding_backend == "cursor"
