"""Session follow-up context from persisted prior runs."""

from __future__ import annotations

from ai_employee.observability.session_context import (
    build_prior_session_context,
    extract_paths,
    extract_urls,
    urls_from_result,
)
from ai_employee.observability.tasks import TaskManager, TaskRecord, TaskStatus


def test_extract_urls_strips_trailing_punctuation():
    text = "See https://www.youtube.com/watch?v=abc123)."
    assert extract_urls(text) == ["https://www.youtube.com/watch?v=abc123"]


def test_extract_paths_finds_workspace_style_paths():
    text = "Saved ./session/mall.mp4 and ./models/yolo26n.pt for later."
    assert "./session/mall.mp4" in extract_paths(text)
    assert "./models/yolo26n.pt" in extract_paths(text)


def test_urls_from_result_collects_answer_steps_and_audit():
    result = {
        "answer": "Clip: https://www.youtube.com/watch?v=oLvM7rXBNRg",
        "steps": [
            {
                "output": "backup https://example.com/foo",
                "resource_audit": {
                    "resources": [
                        {"web_url": "https://www.youtube.com/watch?v=erzPAxE9X58`"},
                    ]
                },
            }
        ],
    }
    urls = urls_from_result(result)
    assert "https://www.youtube.com/watch?v=oLvM7rXBNRg" in urls
    assert "https://www.youtube.com/watch?v=erzPAxE9X58" in urls
    assert "https://example.com/foo" in urls


def test_build_prior_session_context_includes_goals_and_urls():
    prior = [
        {
            "goal": "find cctv footage with smoking",
            "result": {
                "answer": "YouTube clip at https://www.youtube.com/watch?v=oLvM7rXBNRg",
                "steps": [],
            },
        }
    ]
    ctx = build_prior_session_context(prior, workspace=None)
    assert "find cctv footage with smoking" in ctx
    assert "https://www.youtube.com/watch?v=oLvM7rXBNRg" in ctx
    assert "do not ask the user to repeat links" in ctx.lower()


def test_taskmanager_builds_context_from_prior_run():
    tm = TaskManager(None)
    from ai_employee.observability.tasks import _Session

    session = _Session("cctv smoking")
    tm._sessions[session.id] = session

    first = TaskRecord(goal="find cctv smoking footage", session_id=session.id)
    first.id = "run1"
    first.status = TaskStatus.completed
    first.result = {
        "answer": "Found https://www.youtube.com/watch?v=oLvM7rXBNRg",
        "steps": [],
    }
    session.task_ids.append(first.id)
    tm._tasks[first.id] = first

    follow = TaskRecord(
        goal="download that youtube link",
        session_id=session.id,
        parent_id=first.id,
    )
    follow.id = "run2"
    session.task_ids.append(follow.id)
    tm._tasks[follow.id] = follow

    ctx = tm._build_session_context(follow)
    assert "find cctv smoking footage" in ctx
    assert "https://www.youtube.com/watch?v=oLvM7rXBNRg" in ctx


def test_build_prior_session_context_from_real_cctv_session_snapshot():
    """Regression: user's two-tab session (research then download follow-up)."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "sessions" / "5f33e4caea35" / "runs"
    first_path = root / "3322c1c08367.json"
    if not first_path.exists():
        return
    first = json.loads(first_path.read_text(encoding="utf-8"))
    ctx = build_prior_session_context([first])
    assert "find the cctv fotage where it show people do smoking" in ctx
    assert "https://www.youtube.com/watch?v=oLvM7rXBNRg" in ctx


def test_build_prior_session_context_marks_missing_prior_downloads(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    existing = workspace / "sessions" / "sess-1" / "clip.mp4"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"x")

    prior = [
        {
            "goal": "download mall videos",
            "result": {
                "answer": "Downloaded ./sessions/sess-1/clip.mp4 and ./sessions/sess-1/missing.mp4",
                "steps": [
                    {
                        "output": "Saved mall clip",
                        "resource_audit": {
                            "resources": [
                                {
                                    "web_url": "https://www.youtube.com/watch?v=abc123",
                                    "local_path": str(existing),
                                },
                                {
                                    "web_url": "https://www.youtube.com/watch?v=def456",
                                    "local_path": str(workspace / "sessions" / "sess-1" / "missing.mp4"),
                                },
                            ]
                        },
                    }
                ],
            },
        }
    ]
    ctx = build_prior_session_context(prior, workspace=workspace)
    assert "Existing local resources" in ctx
    assert str(existing) in ctx
    assert "Missing prior downloads" in ctx
    assert "https://www.youtube.com/watch?v=def456" in ctx
    assert "re-download" in ctx.lower()
