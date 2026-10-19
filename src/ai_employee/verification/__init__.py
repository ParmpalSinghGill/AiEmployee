"""Verification layer.

Never trust a single generated answer. The :class:`Verifier` runs a self-review
pass over an agent's output against the step's success criteria, returns a verdict
with a confidence score, and tells the orchestrator whether to accept or retry.
"""

from ai_employee.verification.artifacts import audit_artifacts, requested_days_from_task
from ai_employee.verification.filesystem import audit_filesystem
from ai_employee.verification.resources import (
    audit_resources,
    build_waiting_for_user_payload,
    format_resource_block,
    is_resource_task,
)
from ai_employee.verification.verifier import Verdict, Verifier

__all__ = [
    "Verifier",
    "Verdict",
    "audit_artifacts",
    "audit_filesystem",
    "audit_resources",
    "build_waiting_for_user_payload",
    "format_resource_block",
    "is_resource_task",
    "requested_days_from_task",
]
