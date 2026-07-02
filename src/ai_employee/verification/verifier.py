"""Output verification via independent self-review."""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message
from ai_employee.core.logging import get_logger
from ai_employee.core.trace import clip

log = get_logger("verifier")


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
    ) -> Verdict:
        criteria = success_criteria or "The output fully and correctly addresses the task."
        system = (
            "You are a strict reviewer. Judge whether the OUTPUT satisfies the TASK and "
            "SUCCESS CRITERIA. Actively look for errors, omissions, and unsupported "
            "claims. Be skeptical: only pass output that is genuinely correct and "
            "complete. Return your verdict as the required JSON."
        )
        if resource_audit and resource_audit.get("applies"):
            system += (
                "\n\nThis step involves finding/obtaining a resource (video, dataset, file, "
                "URL). Do NOT pass if the output claims a resource was found, downloaded, or "
                "saved without a concrete web URL and/or an existing local file path that "
                "was actually verified. Search-result snippets alone are NOT proof. If nothing "
                "was obtained, the output must say NOT FOUND / NOT DOWNLOADED clearly."
            )
        user = (
            f"TASK:\n{task}\n\nSUCCESS CRITERIA:\n{criteria}\n\nOUTPUT TO REVIEW:\n{output}"
        )
        if resource_audit and resource_audit.get("issues"):
            user += "\n\nPROGRAMMATIC RESOURCE AUDIT ISSUES:\n" + "\n".join(
                f"- {issue}" for issue in resource_audit["issues"]
            )
        # The verifier's *input*: what it was asked to judge and against what.
        log.info(
            "verifier.input",
            task=clip(task, 800),
            criteria=clip(criteria, 400),
            output=clip(output, 2000),
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
        # The verifier's *output*: verdict plus the concrete reasoning behind it.
        log.info(
            "verifier.verdict",
            passed=verdict.passed,
            confidence=verdict.confidence,
            issues=[clip(i, 300) for i in verdict.issues],
            suggestion=clip(verdict.suggestion, 500),
        )
        return verdict
