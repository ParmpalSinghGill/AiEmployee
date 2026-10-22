"""Regression tests for prompt routing, ground-truth audits and runtime-state guards.

Covers:
- planner/coding/git system-prompt rules;
- URL and git-diff-header safe path extraction;
- git working-tree facts + diff-vs-disk consistency in the filesystem audit;
- blocking reads of AIEmployee's own runtime state from shell/file tools.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ai_employee.agents.specs import spec_by_name
from ai_employee.observability.session_context import extract_paths
from ai_employee.verification.filesystem import ZERO_MODIFICATIONS_ISSUE, audit_filesystem

_GIT = shutil.which("git")
_needs_git = pytest.mark.skipif(_GIT is None, reason="git is not installed")

_PLANNER_CATALOG = [
    {"name": "research", "role": "Research Analyst", "description": "investigates"},
    {"name": "coding", "role": "Software Engineer", "description": "writes code"},
    {"name": "git", "role": "Version Control Engineer", "description": "runs git"},
    {"name": "testing", "role": "Test Engineer", "description": "writes tests"},
]


class _StaticPlanBrain:
    """Structured-output stub that returns a fixed plan regardless of the prompt."""

    def __init__(self, plan) -> None:
        self._plan = plan

    async def structured_output(self, messages, schema, **kwargs):
        return self._plan


# ── prompts ──────────────────────────────────────────────────────────────────


def test_coding_prompt_keeps_fixes_physically_on_disk() -> None:
    prompt = spec_by_name()["coding"].system_prompt
    assert "physically saved on disk" in prompt
    assert "do NOT revert your fix" in prompt
    assert "temporary scratch scripts" in prompt
    assert "Never leave the workspace in a broken state" not in prompt


def test_git_prompt_stays_on_requested_repository_operations() -> None:
    prompt = spec_by_name()["git"].system_prompt
    assert "Focus strictly on the requested git operations" in prompt
    assert "Do not inspect internal application logs" in prompt
    assert "unrelated parent directories" in prompt


def test_git_prompt_resets_existing_clone_before_checkout() -> None:
    prompt = spec_by_name()["git"].system_prompt
    assert "git fetch --all" in prompt
    assert "git reset --hard <target_commit>" in prompt
    assert "git clean -fd" in prompt
    assert "leftover edits or different commits" in prompt


def test_skill_extraction_prompt_bans_task_specific_skills() -> None:
    # The retired `learning` agent never ran; skills are distilled by the reflect
    # step with this prompt, so the rules must live here.
    from ai_employee.memory.skill_policy import SKILL_EXTRACTION_SYSTEM as prompt

    assert "NEVER create a skill for a specific task instance" in prompt
    assert "django__django-11555" in prompt
    assert "should_save=false" in prompt
    assert "general-purpose, reusable workflows" in prompt


def test_planner_prompt_gives_each_kind_of_work_one_owner() -> None:
    import inspect

    from ai_employee.planning.planner import Planner

    src = inspect.getsource(Planner.make_plan)
    assert "exactly ONE owner" in src
    assert "never reads, edits or runs local project code" in src
    assert "NO web access" in src
    assert "SINGLE `coding` step" in src
    assert "root-cause diagnosis" in src
    for retired in ("scout", "resource_acquirer", "devops", "hardware_analyst"):
        assert f"`{retired}`" not in src


def test_planner_prompt_forbids_guessing_file_paths() -> None:
    import inspect

    from ai_employee.planning.planner import Planner

    src = inspect.getsource(Planner.make_plan)
    assert "NEVER guess, invent, or hardcode specific source" in src
    assert "root-cause files itself" in src
    assert "actual behavior and" in src
    assert "stack trace" in src


async def test_planner_maps_retired_agent_names_to_successors() -> None:
    """An LLM that remembers the old roster still produces a runnable plan."""
    from ai_employee.planning.planner import Plan, Planner, PlanStep

    plan = Plan(
        goal="find a model, size it, write the training code",
        steps=[
            PlanStep(id=1, description="Search Hugging Face for a model", agent="scout"),
            PlanStep(id=2, description="Estimate VRAM for it", agent="hardware_analyst"),
            PlanStep(id=3, description="Write a Dockerfile for training", agent="devops"),
            PlanStep(id=4, description="Validate the result", agent="qa"),
        ],
    )

    out = await Planner(_StaticPlanBrain(plan)).make_plan(
        "find a model", agent_catalog=_PLANNER_CATALOG
    )

    assert [step.agent for step in out.steps] == ["research", "research", "coding", "testing"]


async def test_planner_collapses_inspection_then_fix_coding_steps() -> None:
    from ai_employee.planning.planner import Plan, Planner, PlanStep

    plan = Plan(
        goal="fix the bug",
        steps=[
            PlanStep(
                id=1,
                description="Locate the root-cause files and inspect the failing module",
                agent="coding",
            ),
            PlanStep(
                id=2,
                description="Implement the minimal fix",
                agent="coding",
                depends_on=[1],
            ),
            PlanStep(
                id=3,
                description="Add a regression test for the fix",
                agent="testing",
                depends_on=[2],
            ),
        ],
    )

    out = await Planner(_StaticPlanBrain(plan)).make_plan(
        "fix the bug", agent_catalog=_PLANNER_CATALOG
    )

    assert [step.agent for step in out.steps] == ["coding", "testing"]
    assert [step.id for step in out.steps] == [1, 2]
    assert "Locate the root-cause files" in out.steps[0].description
    assert "Implement the minimal fix" in out.steps[0].description
    assert out.steps[1].depends_on == [1]


async def test_planner_normalization_does_not_override_a_pinned_run() -> None:
    from ai_employee.planning.planner import Plan, Planner, PlanStep

    plan = Plan(
        goal="stay pinned",
        steps=[
            PlanStep(
                id=1,
                description="Clone the repository and check out the requested commit",
                agent="coding",
            )
        ],
    )

    out = await Planner(_StaticPlanBrain(plan)).make_plan(
        "stay pinned", agent_catalog=_PLANNER_CATALOG, base_agent="research"
    )

    assert out.steps[0].agent == "research"


# ── path extraction ──────────────────────────────────────────────────────────


def test_urls_are_not_audited_as_local_paths(tmp_path: Path) -> None:
    output = (
        "Clone https://github.com/org/repo.git or git://example.com/org/repo.git "
        "and git@github.com:org/repo.git"
    )
    audit = audit_filesystem(output=output, workspace=tmp_path)
    assert audit["applies"] is False
    assert audit["checked"] == []
    assert audit["issues"] == []


def test_diff_headers_do_not_become_a_or_b_directories(tmp_path: Path) -> None:
    output = (
        "Implemented the fix.\n\n"
        "```diff\n"
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )
    audit = audit_filesystem(output=output, workspace=tmp_path)
    assert audit["applies"] is True
    for entry in audit["checked"]:
        assert not entry["path"].startswith(("a/", "b/"))
    assert not any("a/src/app.py" in issue or "b/src/app.py" in issue for issue in audit["issues"])


def test_session_path_extraction_ignores_urls_and_diff_headers() -> None:
    text = (
        "Source: https://github.com/org/repo.git\n"
        "```diff\n"
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "```\n"
    )
    assert extract_paths(text) == []


# ── git working-tree audit ───────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    assert _GIT is not None
    subprocess.run(
        [_GIT, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("old\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _diff_output(repo: Path) -> str:
    return (
        f"Modified {repo}/app.py\n\n"
        "```diff\n"
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )


@_needs_git
def test_git_facts_are_measured_and_reported(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "app.py").write_text("new\n", encoding="utf-8")

    audit = audit_filesystem(output=f"Repository at {repo}", workspace=tmp_path)
    entry = next(c for c in audit["checked"] if Path(c["path"]).resolve() == repo.resolve())
    assert entry["git_status"] == "1 tracked change(s)"
    assert entry["git_diff_stat"] != "none"
    assert entry["git_head"]


@_needs_git
def test_diff_claim_on_clean_tree_is_flagged(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # Remove reflog history so no commit can be mistaken for one made by this step.
    _git(repo, "reflog", "expire", "--expire=now", "--all")

    audit = audit_filesystem(output=_diff_output(repo), workspace=tmp_path)
    assert ZERO_MODIFICATIONS_ISSUE in audit["issues"]


@_needs_git
def test_diff_claim_with_uncommitted_edit_passes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "app.py").write_text("new\n", encoding="utf-8")

    audit = audit_filesystem(output=_diff_output(repo), workspace=tmp_path)
    assert ZERO_MODIFICATIONS_ISSUE not in audit["issues"]


@_needs_git
def test_diff_claim_with_new_commit_passes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "app.py").write_text("new\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "fix")

    import time

    audit = audit_filesystem(output=_diff_output(repo), workspace=tmp_path, since=time.time() - 30)
    assert ZERO_MODIFICATIONS_ISSUE not in audit["issues"]


def test_verifier_describes_git_status_and_diff() -> None:
    from ai_employee.verification.verifier import _describe_fs

    text = _describe_fs(
        {
            "exists": True,
            "kind": "dir",
            "git_head": "abc123",
            "git_status": "1 tracked change(s)",
            "git_diff_stat": "1 file changed, 1 insertion(+)",
        }
    )
    assert "HEAD=abc123" in text
    assert "git status: 1 tracked change(s)" in text
    assert "git diff: 1 file changed, 1 insertion(+)" in text


# ── internal runtime-state sandbox ───────────────────────────────────────────


def _workspace(settings) -> Path:
    ws = Path(settings.workspace_dir).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _configured(settings, field: str) -> Path:
    return Path(getattr(settings, field)).resolve()


def _write(path: Path, content: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


async def test_shell_blocks_internal_audit_log(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    secret = _write(_configured(settings, "audit_log_file"), "{}\n")

    result = await ShellTool(ws).execute(command=f"cat ../{secret.name}")
    assert not result.ok
    assert "internal platform state" in (result.error or "").lower()


async def test_shell_blocks_sessions_directory(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    sessions = _mkdir(_configured(settings, "sessions_dir"))
    _write(sessions / "sess-1.json")

    result = await ShellTool(ws).execute(command=f"cat ../{sessions.name}/sess-1.json")
    assert not result.ok
    assert "internal platform state" in (result.error or "").lower()


async def test_shell_blocks_recursive_grep_of_data_root(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)

    result = await ShellTool(ws).execute(command="grep -r token ..")
    assert not result.ok
    assert "do not search parent application directories" in (result.error or "").lower()
    assert str(ws) in (result.error or "")


async def test_shell_allows_recursive_search_inside_workspace(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    _write(ws / "notes.txt", "needle\n")

    result = await ShellTool(ws).execute(command="grep -rn needle .")
    assert result.ok
    assert "notes.txt" in result.output["stdout"]


async def test_shell_allows_find_inside_workspace(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    _write(ws / "notes.txt", "ok\n")

    result = await ShellTool(ws).execute(command=f"find {ws} -maxdepth 2 -type f")
    assert result.ok
    assert "notes.txt" in result.output["stdout"]


async def test_shell_blocks_parent_app_and_data_searches(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    data_root = _configured(settings, "audit_log_file").parent
    app_root = data_root.parent

    for target in (data_root, app_root):
        result = await ShellTool(ws).execute(command=f"grep -rn token {target}")
        assert not result.ok, f"search of {target} was not blocked"
        error = (result.error or "").lower()
        assert "do not search parent application directories" in error
        assert str(ws) in (result.error or "")


async def test_shell_blocks_find_of_data_root_even_with_maxdepth(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    data_root = _configured(settings, "audit_log_file").parent

    result = await ShellTool(ws).execute(command=f"find {data_root} -maxdepth 2 -type f")
    assert not result.ok
    assert "do not search parent application directories" in (result.error or "").lower()


async def test_shell_blocks_recursive_read_after_cd_to_parent(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    app_root = _configured(settings, "audit_log_file").parent.parent

    result = await ShellTool(ws).execute(command=f"cd {app_root}; grep -rn token .")
    assert not result.ok
    assert "do not search parent application directories" in (result.error or "").lower()


async def test_shell_blocks_absolute_internal_path(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    secret = _write(_configured(settings, "audit_log_file"), "{}\n")

    result = await ShellTool(ws).execute(command=f"head -n 5 {secret}")
    assert not result.ok
    assert "internal platform state" in (result.error or "").lower()


async def test_shell_blocks_inline_python_read_of_internal_state(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    _write(_configured(settings, "audit_log_file"), "{}\n")

    result = await ShellTool(ws).execute(
        command="python3 -c \"print(open('../test_audit_log.jsonl').read())\""
    )
    assert not result.ok
    assert "internal platform state" in (result.error or "").lower()


async def test_shell_still_allows_workspace_files(settings) -> None:
    from ai_employee.tools.builtins.code import ShellTool

    ws = _workspace(settings)
    _write(ws / "notes.txt", "ok\n")

    result = await ShellTool(ws).execute(command="cat notes.txt")
    assert result.ok
    assert result.output["stdout"].strip() == "ok"


async def test_read_file_blocks_internal_secrets(settings) -> None:
    from ai_employee.tools.builtins.filesystem import ReadFileTool

    ws = _workspace(settings)
    secret = _write(_configured(settings, "secrets_file"))

    result = await ReadFileTool(ws).execute(path=f"../{secret.name}")
    assert not result.ok
    assert "internal platform state" in (result.error or "").lower()


async def test_read_file_allows_workspace_file_named_secrets(settings) -> None:
    from ai_employee.tools.builtins.filesystem import ReadFileTool

    ws = _workspace(settings)
    _write(ws / "secrets.json", "workspace-value")

    result = await ReadFileTool(ws).execute(path="secrets.json")
    assert result.ok
    assert result.output == "workspace-value"


async def test_list_dir_blocks_internal_sessions(settings) -> None:
    from ai_employee.tools.builtins.filesystem import ListDirTool

    ws = _workspace(settings)
    sessions = _mkdir(_configured(settings, "sessions_dir"))

    result = await ListDirTool(ws).execute(path=f"../{sessions.name}")
    assert not result.ok
    assert "internal platform state" in (result.error or "").lower()


# ── learned-skill guard (task-specific skills must never be saved/injected) ──


def _reason(name="", description="", steps=None, content=""):
    from ai_employee.memory.skill_policy import skill_rejection_reason

    return skill_rejection_reason(name=name, description=description, steps=steps, content=content)


def test_skill_guard_rejects_instance_ids() -> None:
    assert _reason("# Task: scikit-learn__scikit-learn-14053") is not None
    assert _reason("Work on SWE-bench task django__django-11555") is not None
    assert _reason("sphinx-doc__sphinx-7454") is not None
    assert _reason("sympy__sympy-21612") is not None


def test_skill_guard_rejects_commit_sha() -> None:
    assert (
        _reason(name="fix", steps=["checkout 6ab8c86c383dd847a1be7103ad115f174fe23ffd"]) is not None
    )
    assert (
        _reason(name="fix", content="base commit 8dd5877f58f84f2b11126afbd0813e24545919ed")
        is not None
    )


def test_skill_guard_rejects_raw_task_dump_headers() -> None:
    assert _reason(name="x", description="## 1. Repository Information") is not None
    assert _reason(name="x", description="### Workspace Setup") is not None
    assert _reason(name="x", description="#### Steps/Code to Reproduce") is not None


def test_skill_guard_rejects_question_titles_and_raw_goal() -> None:
    assert _reason(name="do you update the actual code?") is not None
    assert _reason(name="x", description="Reusable approach for: do the thing") is not None


def test_skill_guard_allows_general_skills() -> None:
    assert (
        _reason(
            "Run the Django test suite",
            "how to run tests in a Django repo",
            ["python manage.py test"],
        )
        is None
    )
    assert (
        _reason("Tabular EDA with pandas", "profile a CSV", ["load csv", "summarize columns"])
        is None
    )
    assert _reason("TypeScript project conventions", "", ["npm run lint", "npm test"]) is None


def test_skill_sanitizer_rejects_and_caps() -> None:
    from ai_employee.memory.skill_policy import sanitize_skill

    assert sanitize_skill("# Task: django__django-11555", "x", ["y"]) is None

    long_desc = "word " * 400
    clean = sanitize_skill("general workflow", long_desc, ["one step"])
    assert clean is not None
    words = len((clean["description"] + " " + " ".join(clean["steps"])).split())
    assert words <= 150


def test_skill_guard_filters_serialized_record() -> None:
    from ai_employee.memory.skill_policy import skill_rejection_reason_from_dict

    bad = {
        "namespace": "skill",
        "content": "SKILL: # Task: scikit-learn__scikit-learn-14053\nReusable approach for: x\nSTEPS:\n- a",
        "metadata": {"name": "# Task: scikit-learn__scikit-learn-14053", "steps": ["a"]},
    }
    good = {
        "namespace": "skill",
        "content": "SKILL: Run Django tests\nhow to run tests\nSTEPS:\n- python manage.py test",
        "metadata": {"name": "Run Django tests", "steps": ["python manage.py test"]},
    }
    assert skill_rejection_reason_from_dict(bad) is not None
    assert skill_rejection_reason_from_dict(good) is None


def test_skill_guard_rejects_conversational_titles_without_question_mark() -> None:
    # The exact legacy title that leaked into prompts had no trailing "?".
    assert _reason(name="do you update the acutal code") is not None
    assert _reason(name="Can you fix it") is not None
    assert _reason(name="did we push the branch") is not None


def test_skill_guard_rejects_issue_refs_and_specific_bug_titles() -> None:
    assert _reason(name="Ticket #30557 fix") is not None
    assert _reason(name="Parser cleanup", description="See issue 1234") is not None
    assert _reason(name="x", steps=["read github.com/o/r/pull/88"]) is not None
    assert _reason(name="Fix IndexError in export_text when tree has one feature") is not None


def test_skill_guard_allows_general_titles_near_the_new_rules() -> None:
    assert _reason("Debugging Python ImportError in virtualenvs", "check sys.path") is None
    assert _reason("Fix flaky pytest tests", "rerun, isolate state, seed RNG") is None
    assert _reason("Matplotlib chart styling", "", ["use #333333 for axis text"]) is None
    assert _reason("JSONL dataset generation", "write one JSON object per line") is None
    assert _reason("Run SymPy tests", "", ["bin/test sympy/core"]) is None


def test_task_specific_reason_for_long_term_text() -> None:
    from ai_employee.memory.skill_policy import task_specific_reason

    assert task_specific_reason("GOAL: # Task: django__django-11555\nRESULT: ok") is not None
    assert (
        task_specific_reason("GOAL: checkout 8dd5877f58f84f2b11126afbd0813e24545919ed") is not None
    )
    assert task_specific_reason("GOAL: x\n## 1. Repository Information\n- repo") is not None
    assert task_specific_reason("GOAL: summarize sales.csv\nRESULT: revenue up 4%") is None
    # The agent's own report may use numbered headings; only the goal is checked.
    assert task_specific_reason("GOAL: add /health\nRESULT: done\n### 1. Changes") is None


async def test_long_term_recall_and_import_skip_task_dumps(brain) -> None:
    from ai_employee.memory import MemoryNamespace
    from ai_employee.memory.in_memory import InMemoryVectorStore
    from ai_employee.memory.manager import MemoryManager

    memory = MemoryManager(InMemoryVectorStore(), brain)
    memory.import_records(
        [
            {
                "id": "dump",
                "namespace": "long_term",
                "content": "GOAL: # Task: scikit-learn__scikit-learn-14053\n## 1. Repository Information",
                "metadata": {},
            },
            {"id": "note", "namespace": "long_term", "content": "GOAL: tidy csv", "metadata": {}},
        ]
    )
    assert {r["id"] for r in memory.export_records()} == {"note"}

    # A dump recorded live (pre-guard) is still never surfaced, and purge removes it.
    await memory.record(
        "GOAL: Work on SWE-bench task django__django-11555\nRESULT: done",
        namespace=MemoryNamespace.long_term,
        embed=False,
    )
    good = await memory.record(
        "GOAL: run django tests\nRESULT: use manage.py test", namespace=MemoryNamespace.long_term
    )
    hits = await memory.find_knowledge("django task", k=5)
    assert [h.id for h in hits] == [good.id]
    assert await memory.purge_task_specific_memories() == 1
    assert {r["id"] for r in memory.export_records()} == {"note", good.id}
