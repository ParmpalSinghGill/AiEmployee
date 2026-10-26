"""Fixes from the mall people-tracking research run (2026-10-23).

1. The verifier must not reject real work (paths, bare names, downloads, retries).
2. A Hugging Face link to one file downloads one file; huge repos are refused.
3. Report goals don't download; estimate/compare steps go to research.
4. The "you must download" push fires only on explicit requests.
5. One failing search source doesn't end a research step.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# ── 1. verifier ground truth ────────────────────────────────────────────────


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_relative_path_is_not_read_as_a_root_path(tmp_path: Path) -> None:
    from ai_employee.verification.filesystem import audit_filesystem

    _touch(tmp_path / "data/citypersons/anno_train.mat")
    out = (
        "Saved data/citypersons/anno_train.mat (verified on disk). Sources: "
        "kaggle.com/datasets/agusginting/mall-twlpa; MOT17/MOT20/CrowdHuman labels."
    )
    audit = audit_filesystem(output=out, workspace=tmp_path)
    assert audit["issues"] == []
    checked = {c["path"]: c["exists"] for c in audit["checked"]}
    assert checked.get("data/citypersons/anno_train.mat") is True
    assert not any(p.startswith("/citypersons") or p.startswith("/datasets") for p in checked)


def test_bare_file_name_is_found_in_a_subfolder(tmp_path: Path) -> None:
    from ai_employee.verification.artifacts import audit_artifacts
    from ai_employee.verification.filesystem import audit_filesystem

    _touch(tmp_path / "reports/capacity_grid.csv", b"a,b\n1,2\n")
    out = "Figures trace to capacity_grid.csv, which was saved."
    assert audit_artifacts(output=out, workspace=tmp_path)["issues"] == []
    assert audit_filesystem(output=out, workspace=tmp_path)["issues"] == []


def test_numbers_stay_with_their_own_file(tmp_path: Path) -> None:
    from ai_employee.verification.artifacts import audit_artifacts

    _touch(tmp_path / "reports/grid.csv", b"h\n" + b"r\n" * 168)  # 169 lines
    _touch(tmp_path / "reports/summary.csv", b"h\n" + b"r\n" * 6)  # 7 lines
    out = (
        "Replace ~46 KB with 616 lines / 66,099 B; rewrite provenance to cite "
        "`reports/grid.csv`.\n- `grid.csv` (168 rows), `summary.csv`, saved."
    )
    # 616 belongs to the report, 168 rows = 169 lines minus the header.
    assert audit_artifacts(output=out, workspace=tmp_path)["issues"] == []


def test_wrong_row_count_is_still_caught(tmp_path: Path) -> None:
    from ai_employee.verification.artifacts import audit_artifacts

    _touch(tmp_path / "grid.csv", b"h\n" + b"r\n" * 168)
    issues = audit_artifacts(output="Saved grid.csv with 500 rows.", workspace=tmp_path)["issues"]
    assert issues and "500" in issues[0]


def test_downloads_count_and_span_all_attempts() -> None:
    from ai_employee.verification.evidence import summarize_tool_evidence

    first = [
        {
            "tool": "download_resource",
            "args": {"url": "u", "destination": "./data/MOT17Labels.zip"},
            "ok": True,
            "output": "{'path': '/ws/data/MOT17Labels.zip', 'bytes': 10107022}",
        }
    ]
    retry = [{"tool": "read_file", "args": {"path": "x"}, "ok": True}]
    ev = summarize_tool_evidence(first + retry)
    assert ev["downloads"] == ["/ws/data/MOT17Labels.zip"]


def test_files_changed_since_lists_new_files_and_new_clones(tmp_path: Path) -> None:
    from ai_employee.verification.locate import files_changed_since

    old_repo = tmp_path / "django"
    _touch(old_repo / ".git/config")
    _touch(old_repo / "big.py")
    past = time.time() - 3600
    for p in (old_repo / ".git/config", old_repo / "big.py"):
        os.utime(p, (past, past))
    since = time.time() - 5
    _touch(tmp_path / "data/MOT17Labels.zip", b"z" * 10)
    _touch(tmp_path / "data/OpenTraj/.git/config")  # cloned during the step
    _touch(tmp_path / "django/edited.py")  # inside an old repo: audited via git

    changed = {c["path"]: c for c in files_changed_since(tmp_path, since)}
    assert changed["data/MOT17Labels.zip"]["bytes"] == 10
    assert changed["data/OpenTraj/"]["kind"] == "repo"
    assert not any(p.startswith("django") for p in changed)


class _CaptureBrain:
    def __init__(self) -> None:
        self.messages: list = []

    async def structured_output(self, messages, schema, **kwargs):
        self.messages = messages
        return schema(passed=True, confidence=0.9, issues=[])


async def test_verifier_sees_downloads_and_disk_changes() -> None:
    from ai_employee.verification.verifier import Verifier

    brain = _CaptureBrain()
    await Verifier(brain).verify(  # type: ignore[arg-type]
        task="Find datasets",
        output="Saved data/MOT17Labels.zip",
        tool_evidence={"write_calls": 0, "test_runs": [], "downloads": ["data/MOT17Labels.zip"]},
        disk_changes=[{"path": "data/MOT17Labels.zip", "bytes": 10107022}],
    )
    system, user = brain.messages[0].content, brain.messages[1].content
    assert "never call it fabricated" in system
    assert "successful downloads: 1" in user
    assert "data/MOT17Labels.zip (10,107,022 bytes)" in user


# ── 2. Hugging Face downloads ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main/annotation_val.odgt",
            ("dataset", "sshao0516/CrowdHuman", "main", "annotation_val.odgt"),
        ),
        (
            "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
            ("model", "bert-base-uncased", "main", "config.json"),
        ),
        (
            "https://huggingface.co/org/model/blob/v1/sub/dir/w.safetensors",
            ("model", "org/model", "v1", "sub/dir/w.safetensors"),
        ),
        ("https://huggingface.co/datasets/sshao0516/CrowdHuman", None),
    ],
)
def test_single_file_links_are_recognised(url, expected) -> None:
    from ai_employee.tools.builtins.resource_download import parse_huggingface_file

    assert parse_huggingface_file(url) == expected


async def test_single_file_link_downloads_only_that_file(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    from ai_employee.tools.builtins import resource_download as rd

    def fake_hf_hub_download(*, repo_id, filename, repo_type, revision, cache_dir, token):
        assert (repo_id, filename, repo_type) == ("org/ds", "ann.odgt", "dataset")
        return str(_touch(Path(cache_dir) / "blob", b"12345"))

    def no_snapshot(**kwargs):
        raise AssertionError("whole-repo download must not run for a file link")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_snapshot)
    info = await rd.download_huggingface(
        "https://huggingface.co/datasets/org/ds/resolve/main/ann.odgt",
        tmp_path / "data/ds/ann.odgt",
        workspace=tmp_path,
    )
    assert info["bytes"] == 5
    assert Path(info["path"]) == tmp_path / "data/ds/ann.odgt"


async def test_oversized_repo_is_refused(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    from ai_employee.tools.builtins import resource_download as rd

    monkeypatch.setattr(rd, "_hf_repo_bytes", lambda *a: 13 * 1024**3)

    def no_snapshot(**kwargs):
        raise AssertionError("must not start a 13 GB download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_snapshot)
    with pytest.raises(RuntimeError, match="DOWNLOAD_MAX_GB"):
        await rd.download_huggingface(
            "https://huggingface.co/datasets/sshao0516/CrowdHuman",
            tmp_path / "data/crowdhuman",
            workspace=tmp_path,
            max_bytes=5 * 1024**3,
        )


# ── 3 + 4. download intent and routing ──────────────────────────────────────

MALL_GOAL = (
    "Research AI Models for People Tracking in Shopping Malls. Research and identify the "
    "most suitable AI models, datasets ... Datasets: download/access links and "
    "restrictions. Evaluate Technical Solutions: compare approaches based on: accuracy. "
    "Produce a structured research report."
)


@pytest.mark.parametrize(
    ("goal", "withheld"),
    [
        (MALL_GOAL, True),
        ("Research which detector is best for retail and compare licences", True),
        ("Find a cats and dogs image dataset. Save it in the workspace.", False),
        ("Find 3 CCTV videos of a mall entrance", False),
        ("Compare YOLO models and evaluate them on the COCO dataset", False),
    ],
)
def test_report_goals_withhold_downloads(goal, withheld) -> None:
    from ai_employee.planning.intent import goal_withholds_downloads

    assert goal_withholds_downloads(goal) is withheld


@pytest.mark.parametrize(
    ("text", "asks"),
    [
        ("Check whether public pretrained weights are actually downloadable", False),
        ("Record the direct download or access-request URL", False),
        ("List datasets with download/access links", False),
        ("Download the MOT17 annotations", True),
        ("Save the clips into the workspace", True),
    ],
)
def test_download_intent_is_explicit(text, asks) -> None:
    from ai_employee.planning.intent import asks_to_acquire

    assert asks_to_acquire(text) is asks


def test_own_description_ignores_pasted_prior_outputs() -> None:
    from ai_employee.planning.intent import PRIOR_MARKER, asks_to_acquire, own_description

    task = f"Summarize the models.\n\n{PRIOR_MARKER} — step 3 said: Download the dataset"
    assert not asks_to_acquire(own_description(task))


class _StaticPlanBrain:
    def __init__(self, plan) -> None:
        self._plan = plan

    async def structured_output(self, messages, schema, **kwargs):
        return schema.model_validate(self._plan.model_dump())


async def test_estimate_only_coding_step_moves_to_research() -> None:
    from ai_employee.planning.planner import Plan, Planner, PlanStep

    catalog = [
        {"name": n, "role": n, "description": n}
        for n in ("research", "coding", "testing", "documentation")
    ]
    plan = Plan(
        goal="g",
        steps=[
            PlanStep(
                id=1,
                agent="coding",
                description=(
                    "Estimate hardware and deployment requirements and produce a technical "
                    "comparison: GPU/VRAM needs, achievable FPS and streams-per-GPU."
                ),
            ),
            PlanStep(id=2, agent="coding", description="Write an evaluation script for the model"),
            PlanStep(id=3, agent="coding", description="Benchmark inference FPS on this machine"),
        ],
    )
    out = await Planner(_StaticPlanBrain(plan)).make_plan("g", agent_catalog=catalog)
    assert [s.agent for s in out.steps] == ["research", "coding", "coding"]


# ── 5. one failing source does not end a research step ──────────────────────


class _GitHubThenWeb:
    def __init__(self) -> None:
        self.turn = 0
        self.saw_final = False

    async def chat(self, messages, **kwargs):
        from ai_employee.brains.schemas import ChatResult, ToolCall, Usage

        self.turn += 1
        plan = ["github_search", "github_search", "github_search", "web_search"]
        if self.turn <= len(plan):
            name = plan[self.turn - 1]
            return ChatResult(
                content="",
                tool_calls=[
                    ToolCall(id=str(self.turn), name=name, arguments={"query": f"q{self.turn}"})
                ],
                usage=Usage(),
            )
        self.saw_final = True
        return ChatResult(content="Findings from web search with links.", usage=Usage())


async def test_rate_limited_source_is_switched_off_not_fatal(settings) -> None:
    from pydantic import BaseModel

    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.tools.base import Tool, ToolResult
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry

    calls: list[str] = []

    class _GitHub(Tool):
        name = "github_search"
        description = "gh"
        category = "research"

        class Args(BaseModel):
            query: str

        async def _run(self, args) -> ToolResult:
            calls.append("github")
            return ToolResult(
                ok=False, error="github unavailable: Client error '403 rate limit exceeded'"
            )

    class _Web(Tool):
        name = "web_search"
        description = "web"
        category = "web"

        class Args(BaseModel):
            query: str

        async def _run(self, args) -> ToolResult:
            calls.append("web")
            return ToolResult(ok=True, output=[{"title": "t", "url": "https://x"}])

    class _Agent(ToolUsingAgent):
        name = "research"
        allowed_tools = ["github_search", "web_search"]
        max_iterations = 8

    reg = ToolRegistry(settings)
    reg.register(_GitHub())
    reg.register(_Web())
    brain = _GitHubThenWeb()
    begin_task_tool_history()
    try:
        result = await _Agent().run("Research trackers", AgentContext(brain=brain, tools=reg))  # type: ignore[arg-type]
    finally:
        clear_task_tool_history()

    assert calls == ["github", "github", "web"]  # 3rd GitHub call refused, web still ran
    assert brain.saw_final
    assert "aborted" not in str(result.metadata.get("note", ""))
    assert result.output.startswith("Findings")
