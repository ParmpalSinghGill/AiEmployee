"""Verification layer.

Never trust a single generated answer. The :class:`Verifier` runs a self-review
pass over an agent's output against the step's success criteria, returns a verdict
with a confidence score, and tells the orchestrator whether to accept or retry.
"""

from ai_employee.verification.resources import (
    audit_resources,
    format_resource_block,
    is_resource_task,
)
from ai_employee.verification.verifier import Verdict, Verifier

__all__ = ["Verifier", "Verdict", "audit_resources", "format_resource_block", "is_resource_task"]
