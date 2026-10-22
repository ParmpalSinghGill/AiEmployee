"""Regression tests for the SWE-bench run failures (sympy-21612, django-15732).

Covers:
- guessed source paths in plan steps are flagged, not trusted;
- code-change steps are nudged to write, and fail when no repo file changed;
- the out-of-budget prompt is a platform notice, not a "user instruction";
- the verifier sees real test results and write counts;
- shell/python tools put a referenced repo checkout on PYTHONPATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ai_employee.core.workspace_repos import find_repos, pythonpath_for, references_repo
from ai_employee.verification.evidence import (
    expects_code_change,
    is_mutating_call,
    no_change_issue,
    snapshot_for_change_check,
    summarize_tool_evidence,
)

_GIT = shutil.which("git")
_needs_git = pytest.mark.skipif(_GIT is None, reason="git is not installed")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [_GIT or "git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )


def _make_repo(ws: Path, name: str = "proj") -> Path:
    repo = ws / name
    (repo / name).mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / name / "__init__.py").write_text("VALUE = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


# ── step classification & write detection ───────────────────────────────────


def test_expects_code_change_uses_the_leading_verb() -> None:
    assert expects_code_change("Implement fix in BaseDatabaseSchemaEditor")
    assert expects_code_change("Fix the printer so nested fractions keep brackets")
    assert expects_code_change("Modify the schema editor\n\nPRIOR: ...")
    assert not expects_code_change("Run the new test and the full test suites")
    assert not expects_code_change("Clean up scratch scripts; keep only the minimal fix")
    assert not expects_code_change("Locate the files relevant to the issue")


def test_is_mutating_call_detects_real_writes_only() -> None:
    assert is_mutating_call("write_file", {"path": "a.py", "content": "x"})
    assert is_mutating_call("python_exec", {"code": "open('a.py', 'w').write(s)"})
    assert is_mutating_call("python_exec", {"code": "Path('a').write_text(s)"})
    assert is_mutating_call("shell", {"command": "sed -i 's/a/b/' f.py"})
    assert is_mutating_call("shell", {"command": "echo hi > notes.txt"})
    # Reads, comparisons and stream plumbing are not writes.
    assert not is_mutating_call("python_exec", {"code": "if a > b:\n    print(open('f').read())"})
    assert not is_mutating_call("shell", {"command": "python t.py 2>&1 | tail -5"})
    assert not is_mutating_call("shell", {"command": "grep -n x f.py 2>/dev/null"})
    assert not is_mutating_call("read_file", {"path": "a.py"})


def test_tool_evidence_keeps_test_summary_tail() -> None:
    trace = [
        {"tool": "read_file", "args": {"path": "x"}, "ok": True},
        {"tool": "python_exec", "args": {"code": "open('f.py','w').write('x')"}, "ok": True},
        {
            "tool": "shell",
            "args": {"command": "cd django && python tests/runtests.py schema"},
            "ok": True,
            "output": "(head clipped)",
            "result_tail": "Ran 179 tests in 1.3s\n\nOK (skipped=28)",
        },
    ]
    ev = summarize_tool_evidence(trace)
    assert ev["write_calls"] == 1
    assert len(ev["test_runs"]) == 1
    assert "Ran 179 tests" in ev["test_runs"][0]["tail"]


# ── repo references & PYTHONPATH ────────────────────────────────────────────


def test_references_repo_requires_a_path_like_reference(tmp_path: Path) -> None:
    repo = tmp_path / "django"
    repo.mkdir()
    for text in (
        "cd django && python tests/runtests.py",
        "python django/tests/runtests.py schema",
        "git -C django status",
        'subprocess.run(["python", "x.py"], cwd="django")',
        f"ls {repo}/tests",
    ):
        assert references_repo(text, repo, workspace=tmp_path), text
    for text in ("import django", "pip show django", "the django project"):
        assert not references_repo(text, repo, workspace=tmp_path), text


def test_pythonpath_for_only_python_repos(tmp_path: Path) -> None:
    py = tmp_path / "proj"
    (py / ".git").mkdir(parents=True)
    (py / "pyproject.toml").write_text("")
    (py / "src").mkdir()
    other = tmp_path / "webapp"
    (other / ".git").mkdir(parents=True)

    assert {p.name for p in find_repos(tmp_path)} == {"proj", "webapp"}
    assert pythonpath_for("cd proj && pytest", tmp_path) == [
        str(py.resolve()),
        str((py / "src").resolve()),
    ]
    assert pythonpath_for("cd webapp && npm test", tmp_path) == []
    assert pythonpath_for("python -c 'import proj'", tmp_path) == []


async def test_shell_and_python_exec_import_the_checkout(tmp_path: Path) -> None:
    from ai_employee.tools.builtins.code import PythonExecTool, ShellTool

    repo = tmp_path / "proj"
    (repo / "proj").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    (repo / "proj" / "__init__.py").write_text("WHERE = 'checkout'\n")

    shell = ShellTool(tmp_path)
    res = await shell._run(
        ShellTool.Args(command="cd proj/.. && python -c 'import proj; print(proj.WHERE)' # proj/")
    )
    assert res.ok, res.error
    assert "checkout" in res.output["stdout"]
    assert res.output["pythonpath_added"] == [str(repo.resolve())]

    py = PythonExecTool(tmp_path)
    res = await py._run(
        PythonExecTool.Args(
            code=(
                "import subprocess, sys\n"
                "r = subprocess.run([sys.executable, '-c', 'import proj; print(proj.WHERE)'],"
                " cwd='proj', capture_output=True, text=True)\n"
                "print(r.stdout)"
            )
        )
    )
    assert res.ok, res.error
    assert "checkout" in res.output["stdout"]


# ── "did the code-change step change anything?" ─────────────────────────────


@_needs_git
def test_no_change_issue_flags_an_untouched_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    before = snapshot_for_change_check(tmp_path)
    issue = no_change_issue(
        before=before, workspace=tmp_path, scope_text="Implement fix in proj/proj/__init__.py"
    )
    assert issue and "proj" in issue

    (repo / "proj" / "__init__.py").write_text("VALUE = 2\n")
    assert (
        no_change_issue(before=before, workspace=tmp_path, scope_text="fix proj/proj/__init__.py")
        is None
    )


@_needs_git
def test_no_change_issue_counts_new_files_and_skips_out_of_scope(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    before = snapshot_for_change_check(tmp_path)
    (repo / "tests_new.py").write_text("def test(): pass\n")
    assert no_change_issue(before=before, workspace=tmp_path, scope_text="edit proj/x") is None
    # A step that never references the repo is not judged by it.
    before = snapshot_for_change_check(tmp_path)
    assert (
        no_change_issue(before=before, workspace=tmp_path, scope_text="update the README") is None
    )


@_needs_git
def test_no_change_issue_skips_repo_dirty_at_start(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "proj" / "__init__.py").write_text("VALUE = 3\n")  # earlier step's edit
    before = snapshot_for_change_check(tmp_path)
    assert no_change_issue(before=before, workspace=tmp_path, scope_text="fix proj/x") is None


# ── planner: guessed file paths ─────────────────────────────────────────────


def test_planner_flags_paths_the_goal_never_named() -> None:
    from ai_employee.planning.planner import Plan, PlanStep, flag_guessed_paths

    goal = "Latex parsing of fractions yields wrong expression due to missing brackets"
    plan = Plan(
        goal=goal,
        steps=[
            PlanStep(
                id=1,
                description="Implement the fix in `sympy/parsing/latex/_parse_latex_antlr.py`.",
                agent="coding",
            ),
            PlanStep(id=2, description="Run sympy/parsing/tests/test_latex.py", agent="testing"),
        ],
    )
    flag_guessed_paths(plan, goal)
    fix, run = plan.steps
    assert "_parse_latex_antlr.py` was named by the planner" in fix.description
    assert "ROOT CAUSE" in fix.description
    assert "named by the planner" not in run.description  # running tests: no note
    # Idempotent.
    flag_guessed_paths(plan, goal)
    assert fix.description.count("named by the planner") == 1


def test_planner_keeps_paths_the_user_named() -> None:
    from ai_employee.planning.planner import Plan, PlanStep, flag_guessed_paths

    goal = "Fix the bug in utils/dates.py where parse() drops timezones"
    plan = Plan(
        goal=goal,
        steps=[PlanStep(id=1, description="Fix parse() in utils/dates.py", agent="coding")],
    )
    flag_guessed_paths(plan, goal)
    assert "named by the planner" not in plan.steps[0].description


def test_coding_prompt_requires_root_cause_and_applied_edits() -> None:
    from ai_employee.agents.specs import AGENT_SPECS

    prompt = next(s for s in AGENT_SPECS if s.name == "coding").system_prompt
    assert "ROOT CAUSE FIRST" in prompt
    assert "APPLY, DON'T DESCRIBE" in prompt


# ── verifier ────────────────────────────────────────────────────────────────


class _CapturingBrain:
    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.messages: list = []

    async def structured_output(self, messages, schema, **kwargs):
        self.messages = messages
        return schema(passed=self.passed, confidence=0.9, issues=[])


async def test_verifier_sees_tool_evidence() -> None:
    from ai_employee.verification.verifier import Verifier

    brain = _CapturingBrain()
    evidence = {
        "write_calls": 1,
        "test_runs": [
            {"command": "python tests/runtests.py schema", "ok": True, "tail": "Ran 179 tests OK"}
        ],
    }
    verdict = await Verifier(brain).verify(  # type: ignore[arg-type]
        task="Run the schema tests", output="Blocked: could not run tests", tool_evidence=evidence
    )
    assert verdict.passed
    system, user = brain.messages[0].content, brain.messages[1].content
    assert "do NOT fail the step for 'tests not run'" in system
    assert "Ran 179 tests OK" in user
    assert "successful file-writing calls: 1" in user


async def test_verifier_change_issue_forces_failure() -> None:
    from ai_employee.verification.verifier import Verifier

    verdict = await Verifier(_CapturingBrain(passed=True)).verify(  # type: ignore[arg-type]
        task="Implement the fix",
        output="Applied the fix to schema.py",
        change_issue="No file in django changed during this step",
    )
    assert not verdict.passed
    assert verdict.confidence <= 0.2
    assert any("No file in django changed" in i for i in verdict.issues)


# ── agent loop: write nudge & out-of-budget notice ──────────────────────────


class _ReadOnlyBrain:
    """Keeps making distinct read-only calls; answers in text when tools are off."""

    def __init__(self) -> None:
        self.turn = 0
        self.seen: list[list] = []

    async def chat(self, messages, **kwargs):
        from ai_employee.brains.schemas import ChatResult, ToolCall, Usage

        self.seen.append(list(messages))
        if not kwargs.get("tools"):
            return ChatResult(
                content="Final: proposed change only, nothing applied.", usage=Usage()
            )
        self.turn += 1
        return ChatResult(
            content="",
            tool_calls=[
                ToolCall(id=f"c{self.turn}", name="echo", arguments={"query": f"q{self.turn}"})
            ],
            usage=Usage(),
        )


async def _run_agent(settings, monkeypatch, task: str, **overrides):
    import ai_employee.agents.base as base_mod
    from ai_employee.agents.base import AgentContext, ToolUsingAgent
    from ai_employee.tools.invocation_history import (
        begin_task_tool_history,
        clear_task_tool_history,
    )
    from ai_employee.tools.registry import ToolRegistry
    from tests.test_tool_duplicate import EchoTool

    s = settings.model_copy(update={"agent_loop_check_enabled": False, **overrides})
    monkeypatch.setattr(base_mod, "get_settings", lambda: s)

    class _Agent(ToolUsingAgent):
        name = "coder"
        allowed_tools = ["echo"]
        max_iterations = 6

    reg = ToolRegistry(s)
    reg.register(EchoTool())
    brain = _ReadOnlyBrain()
    begin_task_tool_history()
    try:
        result = await _Agent().run(task, AgentContext(brain=brain, tools=reg))  # type: ignore[arg-type]
    finally:
        clear_task_tool_history()
    return brain, result


def _all_text(brain: _ReadOnlyBrain) -> str:
    return "\n".join(str(m.content or "") for call in brain.seen for m in call)


async def test_code_change_step_is_nudged_to_write(settings, monkeypatch) -> None:
    brain, _ = await _run_agent(
        settings, monkeypatch, "Implement the fix in the parser", agent_write_nudge_after=2
    )
    text = _all_text(brain)
    assert "no file has been changed" in text
    assert "tool round(s) remain" in text  # final-window warning


async def test_read_only_step_is_not_nudged(settings, monkeypatch) -> None:
    brain, _ = await _run_agent(
        settings, monkeypatch, "Locate the files for the issue", agent_write_nudge_after=2
    )
    assert "[PLATFORM NOTICE] " not in _all_text(brain).split("tool-call budget")[0]


async def test_out_of_budget_prompt_is_a_platform_notice(settings, monkeypatch) -> None:
    brain, result = await _run_agent(settings, monkeypatch, "Locate the files for the issue")
    final_prompt = str(brain.seen[-1][-1].content)
    assert "not from the user" in final_prompt
    assert "tool-call budget is used up" in final_prompt
    assert "Stop using tools now" not in final_prompt
    assert "max_iterations" in result.metadata["note"]


# ── cancelling a running shell/python tool ──────────────────────────────────


async def test_cancelled_shell_command_kills_its_child(tmp_path: Path) -> None:
    import asyncio
    import os

    from ai_employee.tools.builtins.code import ShellTool

    marker = tmp_path / "pid"
    task = asyncio.create_task(
        ShellTool(tmp_path)._run(
            ShellTool.Args(command=f"echo $$ > {marker}; exec sleep 30", timeout=60)
        )
    )
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        await asyncio.sleep(0.05)
    pid = int(marker.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the child was killed, not orphaned


async def test_cancel_during_spawn_never_hangs(tmp_path: Path) -> None:
    """A Stop that lands while the subprocess is still being spawned must finish."""
    import asyncio

    from ai_employee.tools.builtins.code import PythonExecTool

    tool = PythonExecTool(tmp_path)
    for _ in range(10):
        task = asyncio.create_task(tool._run(PythonExecTool.Args(code="print(1)")))
        await asyncio.sleep(0)  # let the spawn start, then cancel mid-flight
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
