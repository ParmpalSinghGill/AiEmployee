"""Output verification via independent self-review."""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip

log = get_logger("verifier")


def _describe_checked(entry: dict) -> str:
    parts = [f"{entry['actual_bytes']:,} bytes"]
    if "actual_lines" in entry:
        parts.append(f"{entry['actual_lines']:,} lines")
    if "actual_dimensions" in entry:
        w, h = entry["actual_dimensions"]
        parts.append(f"{w}x{h}px")
    return ", ".join(parts)


def _describe_fs(entry: dict) -> str:
    if not entry.get("exists"):
        return "does not exist"
    parts = [str(entry.get("kind"))]
    if entry.get("git_head"):
        parts.append(f"HEAD={entry['git_head']}")
    if entry.get("git_status"):
        parts.append(f"git status: {entry['git_status']}")
    if entry.get("git_diff_stat"):
        parts.append(f"git diff: {entry['git_diff_stat']}")
    if entry.get("git_staged_stat"):
        parts.append(f"git staged: {entry['git_staged_stat']}")
    if entry.get("kind") == "dir":
        parts.append(f"{entry.get('entry_count', 0)} entries")
        names = entry.get("entries") or []
        if names:
            parts.append("e.g. " + ", ".join(names[:8]))
    elif entry.get("bytes") is not None:
        parts.append(f"{entry['bytes']:,} bytes")
    return ", ".join(parts)


class Verdict(BaseModel):
    passed: bool = Field(description="True if the output satisfies the requirement.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict.")
    issues: list[str] = Field(default_factory=list, description="Concrete problems found.")
    suggestion: str = Field(default="", description="How to fix it if it failed.")


class Verifier:
    """Adversarial-ish reviewer: tries to find what is wrong before accepting."""

    def __init__(self, brain: Brain) -> None:
        self._brain = brain

    async def verify(
        self,
        *,
        task: str,
        output: str,
        success_criteria: str = "",
        resource_audit: dict | None = None,
        artifact_audit: dict | None = None,
        fs_audit: dict | None = None,
        tool_evidence: dict | None = None,
        change_issue: str | None = None,
    ) -> Verdict:
        criteria = success_criteria or "The output fully and correctly addresses the task."
        system = (
            "You are a strict reviewer. Judge whether the OUTPUT satisfies the TASK and "
            "SUCCESS CRITERIA. Actively look for errors, omissions, and unsupported "
            "claims. Be skeptical: only pass output that is genuinely correct and "
            "complete. Return your verdict as the required JSON."
        )
        if artifact_audit and artifact_audit.get("applies"):
            system += (
                "\n\nGROUND TRUTH: the facts below about files the output mentions were "
                "measured directly from disk (actual byte size / line count / pixel "
                "dimensions) — they are not the agent's self-report and cannot be wrong. "
                "If the output's own prose contradicts a ground-truth fact for a file's "
                "FINAL state, that is a real defect. But agents sometimes redo a file "
                "mid-task (e.g. regenerate a chart) and their prose can quote an earlier, "
                "superseded version alongside the final one — that is sloppy writing, not "
                "a defect in the deliverable, since ground truth always reflects the file "
                "as it exists now. Judge which case this is; don't fail solely because the "
                "prose is internally inconsistent if the ground truth itself checks out.\n"
                "A date-range shortfall (task asked for N days, the file covers far fewer) "
                "IS a real defect even if the agent's prose is internally consistent and "
                "even if the shortfall has a legitimate cause (e.g. a data provider's "
                "intraday history limit) — the deliverable still doesn't cover what was "
                "asked. Only pass such a step if the output itself clearly discloses the "
                "shortfall and the reason for it; fail it if the output presents a partial "
                "range as if it fully satisfies the request."
            )
        if resource_audit and resource_audit.get("applies"):
            outcome = resource_audit.get("outcome", "")
            system += (
                "\n\nThis step involves finding/obtaining a resource (video, dataset, file, "
                "URL). Distinguish three outcomes:\n"
                "- COMPLETED: a concrete web URL AND/OR an existing local file was verified.\n"
                "- WAITING_FOR_USER: a suitable resource URL was found and automatic download "
                "was attempted or is impossible (403, login, CAPTCHA, etc.) — manual download "
                "is acceptable; do NOT fail for missing local file alone.\n"
                "- FAILED / NOT_FOUND: no suitable resource exists, only broken URLs, or the "
                "output falsely claims success without URLs.\n"
                "Search snippets alone are NOT proof. Never treat 'resource not found' as true "
                "when concrete candidate URLs are listed."
            )
            if outcome == "waiting_for_user":
                system += (
                    "\n\nThe programmatic audit marks this as WAITING_FOR_USER: pass if a "
                    "suitable URL is documented even when the file is not in the workspace yet."
                )
        if fs_audit and fs_audit.get("applies"):
            system += (
                "\n\nFILESYSTEM GROUND TRUTH: the facts below about paths the output "
                "mentions were measured directly from disk — whether each path exists, "
                "its kind, its directory contents, and its resolved git HEAD — they are "
                "not the agent's self-report and cannot be wrong. If the output claims a "
                "path exists, is located somewhere, or was created/cloned/checked out "
                "somewhere that ground truth contradicts, that is a real defect. If the "
                "output quotes a git commit SHA for a directory, compare it to the "
                "measured HEAD and flag any mismatch. A path the output correctly says "
                "does NOT exist is not a defect."
            )
        runs = (tool_evidence or {}).get("test_runs") or []
        if tool_evidence is not None:
            system += (
                "\n\nTOOL EXECUTION EVIDENCE: the platform recorded what this step's "
                "tools actually did — the test commands it ran with their real results, "
                "and how many file-writing calls succeeded. This is not the agent's "
                "self-report. Judge test outcomes from it: if the evidence shows the "
                "requested tests ran and passed, do NOT fail the step for 'tests not "
                "run' even if the prose says they were blocked. If the prose claims "
                "tests passed but the evidence shows failures or no test run, that is a "
                "real defect. If the task requires changing code and there were 0 "
                "file-writing calls, claims that changes were applied are unsupported."
            )
        user = f"TASK:\n{task}\n\nSUCCESS CRITERIA:\n{criteria}\n\nOUTPUT TO REVIEW:\n{output}"
        if tool_evidence is not None:
            user += (
                "\n\nTOOL EXECUTION EVIDENCE (recorded by the platform):\n"
                f"- successful file-writing calls: {tool_evidence.get('write_calls', 0)}\n"
            )
            if runs:
                user += "- test commands run (oldest first):\n" + "\n".join(
                    f"  * [{'ok' if r.get('ok') else 'FAILED'}] {r.get('command', '')[:200]}\n"
                    f"    result tail: {r.get('tail', '')}"
                    for r in runs
                )
            else:
                user += "- no test commands were run in this step\n"
        if resource_audit and resource_audit.get("issues"):
            user += "\n\nPROGRAMMATIC RESOURCE AUDIT ISSUES:\n" + "\n".join(
                f"- {issue}" for issue in resource_audit["issues"]
            )
        if artifact_audit and artifact_audit.get("checked"):
            user += "\n\nGROUND-TRUTH FILE FACTS (measured directly from disk):\n" + "\n".join(
                f"- {c['file']}: {_describe_checked(c)}" for c in artifact_audit["checked"]
            )
        if fs_audit and fs_audit.get("checked"):
            user += (
                "\n\nGROUND-TRUTH FILESYSTEM FACTS (measured directly from disk):\n"
                + "\n".join(f"- {c['path']}: {_describe_fs(c)}" for c in fs_audit["checked"])
            )
        if fs_audit and fs_audit.get("issues"):
            user += "\n\nPROGRAMMATIC FILESYSTEM AUDIT — prose vs. ground truth:\n" + "\n".join(
                f"- {issue}" for issue in fs_audit["issues"]
            )
        if artifact_audit and artifact_audit.get("issues"):
            user += "\n\nPROGRAMMATIC ARTIFACT AUDIT — prose vs. ground truth:\n" + "\n".join(
                f"- {issue}" for issue in artifact_audit["issues"]
            )
        # The verifier's *input*: what it was asked to judge and against what.
        log.info(
            "verifier.input",
            task=clip(task),
            criteria=clip(criteria),
            output=clip(output),
        )
        try:
            # phase=verify lets the trace attribute this LLM call to the verifier
            # rather than to the agent whose context is currently bound.
            with structlog.contextvars.bound_contextvars(phase="verify"):
                verdict = await self._brain.structured_output(
                    [Message.system(system), Message.user(user)], Verdict
                )
        except Exception as exc:  # noqa: BLE001 - never let verification crash the run
            log.warning("verifier.failed", error=str(exc))
            verdict = Verdict(passed=True, confidence=0.5, issues=["verifier unavailable"])
        if resource_audit and resource_audit.get("issues"):
            outcome = resource_audit.get("outcome", "")
            if outcome == "waiting_for_user":
                verdict = Verdict(
                    passed=True,
                    confidence=max(verdict.confidence, 0.75),
                    issues=verdict.issues,
                    suggestion=verdict.suggestion
                    or "Resource found; waiting for the user to supply a local file path.",
                )
            else:
                verdict = Verdict(
                    passed=False,
                    confidence=min(verdict.confidence, 0.25),
                    issues=[*verdict.issues, *resource_audit["issues"]],
                    suggestion=verdict.suggestion
                    or (
                        "Only report resources you verified. For each resource include: "
                        "Web URL, Local path (or NOT DOWNLOADED), and the model used."
                    ),
                )
        # Deterministic: a code-change step that left its repo untouched failed,
        # whatever the prose says.
        if change_issue:
            verdict = Verdict(
                passed=False,
                confidence=min(verdict.confidence, 0.2),
                issues=[*verdict.issues, change_issue],
                suggestion=verdict.suggestion
                or "Apply the change to the files on disk, then verify it with tests.",
            )
        # The verifier's *output*: verdict plus the concrete reasoning behind it.
        log.info(
            "verifier.verdict",
            passed=verdict.passed,
            confidence=verdict.confidence,
            issues=[clip(i) for i in verdict.issues],
            suggestion=clip(verdict.suggestion),
            artifact_audit_issues=(artifact_audit or {}).get("issues") or [],
            fs_audit_issues=(fs_audit or {}).get("issues") or [],
            change_issue=change_issue or "",
            test_runs=len(runs),
            write_calls=(tool_evidence or {}).get("write_calls"),
        )
        return verdict
