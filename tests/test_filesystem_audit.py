"""Ground-truth filesystem checks: cross-checking an agent's path claims against disk.

Regression coverage for the real django-ticket failure observed in a live run: the
agent reported a checkout's path and HEAD SHA in prose, and the LLM verifier — which
only reads prose — rejected the (correct) result because it could not confirm the
location or the commit. audit_filesystem measures existence, kind, directory
contents and the resolved git HEAD directly so the verifier gets facts instead of
trusting the agent's self-report.
"""

from __future__ import annotations

from pathlib import Path

from ai_employee.verification.filesystem import audit_filesystem


def _write_git_repo(root: Path, head_sha: str) -> None:
    git = root / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    refs = git / "refs" / "heads"
    refs.mkdir(parents=True)
    (refs / "main").write_text(head_sha + "\n", encoding="utf-8")


def test_no_paths_mentioned_does_not_apply() -> None:
    audit = audit_filesystem(output="The result looks good.", workspace=Path("."))
    assert audit["applies"] is False
    assert audit["issues"] == []
    assert audit["checked"] == []


def test_existing_directory_is_reported_with_git_head(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "manage.py").write_text("", encoding="utf-8")
    sha = "8dd5877f58f84f2b11126afbd0813e24545919ed"
    _write_git_repo(repo, sha)

    output = f"Checked out {repo} — git rev-parse HEAD prints {sha}."
    audit = audit_filesystem(output=output, workspace=tmp_path)

    assert audit["applies"] is True
    entry = next(c for c in audit["checked"] if c["path"] == str(repo))
    assert entry["exists"] is True
    assert entry["kind"] == "dir"
    assert entry["git_head"] == sha
    assert "manage.py" in entry["entries"]
    # The quoted SHA matches the measured HEAD, so no issue.
    assert audit["issues"] == []


def test_claimed_but_missing_path_is_flagged(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "clone"
    output = f"Created {missing} with the full Django checkout."
    audit = audit_filesystem(output=output, workspace=tmp_path)
    assert audit["applies"] is True
    assert any("does not exist" in issue for issue in audit["issues"])


def test_negated_existence_is_not_flagged(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "clone"
    output = f"Confirmed {missing} does not exist, so I created it elsewhere."
    audit = audit_filesystem(output=output, workspace=tmp_path)
    assert audit["applies"] is True
    assert audit["issues"] == []


def test_wrong_quoted_head_sha_is_flagged(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    real = "8dd5877f58f84f2b11126afbd0813e24545919ed"
    wrong = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    _write_git_repo(repo, real)

    output = f"Checked out {repo} at commit {wrong}."
    audit = audit_filesystem(output=output, workspace=tmp_path)
    assert any(real in issue for issue in audit["issues"])
    assert any(wrong in issue for issue in audit["issues"])


def test_relative_parent_path_resolves_and_reports(tmp_path: Path) -> None:
    # A "../"-relative path resolves relative to the workspace (like the agent's
    # sandbox) — the audit reports where it actually lands, not where the agent
    # meant it to land.
    sibling = (tmp_path / ".." / "outside").resolve()
    sibling.mkdir(parents=True, exist_ok=True)
    try:
        output = "Created ../outside."
        audit = audit_filesystem(output=output, workspace=tmp_path)
        assert audit["applies"] is True
        assert any(c.get("exists") and c.get("kind") == "dir" for c in audit["checked"])
    finally:
        sibling.rmdir()
