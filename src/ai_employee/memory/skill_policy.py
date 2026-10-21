"""Guard and sanitizer for learned skills.

Skills are injected into future prompts under "Relevant prior knowledge & learned
skills". A skill that merely echoes a specific task — a benchmark instance id, a
git commit SHA, a raw issue dump, or a conversational question — wastes tokens and
contaminates later runs with stale ids, SHAs and file paths.

This module holds the deterministic policy so correctness does not depend on the
LLM alone. It is used both when *saving* a skill (reject/normalize) and when
*loading* skills for prompt injection (skip task-specific records).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

#: ``owner__repo-1234`` style benchmark/task instance ids.
_INSTANCE_ID_RE = re.compile(r"\b[a-z0-9][a-z0-9_.-]*__[a-z0-9_.-]+-\d+\b", re.I)

#: 40-character hex git commit SHA (surrounded by non-hex so we don't split one).
_SHA_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", re.I)

#: Raw task-dump markdown headings (``## 1. Repository Information``, ``### Workspace
#: Setup``, ``#### Steps/Code to Reproduce``, ``## Tested on`` …).
_RAW_HEADER_RE = re.compile(
    r"(?m)^\s*#{1,6}\s+(?:"
    r"\d+\.\s+|"
    r".*\b(?:Repository|Workspace|Setup|Issue|Ticket|Reproduce|Environment|"
    r"Instance|Patch|Commit|SHA|Tested\s+on|Steps)\b.*"
    r")",
    re.I,
)

#: A title that is just the pasted raw task prompt.
_RAW_TITLE_RE = re.compile(r"^#?\s*task\s*:", re.I)

#: The reflect node's old "reusable approach for: <goal>" passthrough.
_RAW_GOAL_PREFIX = "reusable approach for:"

#: Interrogative openers, used only with a trailing ``?`` (one-off questions).
_QUESTION_RE = re.compile(
    r"^(?:do|does|can|could|how|what|why|when|where|is|are|should|would|did|will|"
    r"has|have|was|were|which|who|whose|am)\b",
    re.I,
)

#: Skills must be concise; anything longer is almost certainly a pasted plan/task.
MAX_SKILL_WORDS = 150


class SkillRejectedError(ValueError):
    """A candidate skill is task-specific and must not be saved."""


class SkillDraft(BaseModel):
    """LLM-produced distillation of a completed task into a reusable skill."""

    should_save: bool = Field(
        default=True,
        description="True only when the task yielded a genuinely reusable, general workflow.",
    )
    name: str = Field(
        default="",
        description="Short general name at the language/framework/dataset/tooling level.",
    )
    description: str = Field(
        default="",
        description="When to use + what to do, 1-3 sentences, no instance ids or SHAs.",
    )
    steps: list[str] = Field(
        default_factory=list, description="3-6 concise, generalizable action steps."
    )
    tags: list[str] = Field(default_factory=list, description="Optional scoping tags.")


SKILL_EXTRACTION_SYSTEM = (
    "You distill a completed task into a reusable skill for future tasks.\n"
    "STRICT RULES:\n"
    "- NEVER create a skill for a specific task instance, bug ticket, commit hash, "
    "or one-off user question.\n"
    "- FORBIDDEN in a skill: task/benchmark instance IDs (e.g. django__django-11555, "
    "scikit-learn__scikit-learn-14053, sphinx-doc__sphinx-7454), git commit SHAs "
    "(40 hex characters), issue/PR numbers or bug descriptions, the raw user goal/"
    "prompt pasted as the title or body, and conversational follow-up questions "
    '(e.g. "do you update the actual code?").\n'
    "- ALLOWED: general-purpose, reusable workflows scoped to a programming language "
    "(e.g. Python, TypeScript), a framework/repository family (e.g. how to run "
    "Django or SymPy tests generally), a dataset/modality type (e.g. tabular EDA, "
    "JSONL generation), or a tooling pattern.\n"
    "- If the task only applied existing knowledge to fix a specific bug and yielded "
    "no new generalizable workflow, set should_save=false.\n"
    "Write the name and description at the repository/language/dataset level — never "
    "with instance IDs, SHAs, or file paths from this specific run."
)


def skill_slug(name: str) -> str:
    """A canonical, case-insensitive key for deduplicating skills by title."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def skill_fields_from_content(content: str) -> tuple[str, str, list[str]]:
    """Parse ``SKILL: name`` / description / ``STEPS:`` from stored content."""
    lines = (content or "").splitlines()
    name = ""
    description_lines: list[str] = []
    steps: list[str] = []
    in_steps = False
    for index, line in enumerate(lines):
        if index == 0 and line.strip().upper().startswith("SKILL:"):
            name = line.split(":", 1)[1].strip()
            continue
        if line.strip().upper() == "STEPS:":
            in_steps = True
            continue
        if in_steps:
            step = line.strip()
            if step.startswith("- "):
                step = step[2:]
            if step:
                steps.append(step.strip())
        else:
            description_lines.append(line)
    return name, "\n".join(description_lines).strip(), steps


def skill_name_from_content(content: str) -> str:
    return skill_fields_from_content(content)[0]


def skill_rejection_reason(
    name: str = "",
    description: str = "",
    steps: list[str] | None = None,
    content: str = "",
) -> str | None:
    """Return a human reason a skill should be rejected, or ``None`` if it is OK."""
    name = (name or "").strip()
    description = (description or "").strip()
    steps = [str(s) for s in (steps or []) if str(s).strip()]
    combined = "\n".join([name, description, *steps, content or ""])

    if _INSTANCE_ID_RE.search(combined):
        return "contains a task/benchmark instance id"
    if _SHA_RE.search(combined):
        return "contains a git commit SHA"
    if _RAW_HEADER_RE.search(combined):
        return "contains raw task markdown headers"
    if _RAW_TITLE_RE.search(name):
        return "title pastes the raw task prompt"

    title = name.splitlines()[0].strip() if name else ""
    if title.rstrip().endswith("?") and _QUESTION_RE.search(title):
        return "looks like a one-off question, not a reusable skill"

    if description.lower().startswith(_RAW_GOAL_PREFIX):
        return "description echoes the raw task goal"
    return None


def skill_rejection_reason_from_dict(data: dict[str, Any]) -> str | None:
    """Rejection check for a serialized memory record (JSONL / pgvector row)."""
    metadata = data.get("metadata") or {}
    content = str(data.get("content") or "")
    meta_name = str(metadata.get("name") or "").strip()
    _name, description, steps_from_content = skill_fields_from_content(content)
    steps = metadata.get("steps")
    if not isinstance(steps, list):
        steps = steps_from_content
    return skill_rejection_reason(
        name=meta_name or _name,
        description=description,
        steps=[str(s) for s in steps],
        content=content,
    )


def skill_rejection_reason_from_record(record: Any) -> str | None:
    """Rejection check for a :class:`MemoryRecord` (used on load)."""
    try:
        return skill_rejection_reason_from_dict(record.model_dump(mode="json"))
    except Exception:  # noqa: BLE001 - never break recall over a bad record
        metadata = getattr(record, "metadata", None) or {}
        content = getattr(record, "content", "") or ""
        name = str(metadata.get("name") or "")
        return skill_rejection_reason(name=name, content=content)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _cap_length(
    name: str, description: str, steps: list[str], max_words: int
) -> tuple[str, str, list[str]]:
    total = _word_count(name) + _word_count(description) + sum(_word_count(s) for s in steps)
    if total <= max_words:
        return name, description, steps

    # Drop steps from the end first, then trim the description to the remaining
    # word budget. This keeps the most actionable, general part of the skill.
    while steps and total > max_words:
        total -= _word_count(steps[-1])
        steps.pop()
    if total <= max_words:
        return name, description, steps

    budget = max_words - _word_count(name) - sum(_word_count(s) for s in steps)
    description = "" if budget <= 0 else " ".join(description.split()[:budget]).rstrip()
    return name, description, steps


def sanitize_skill(
    name: str,
    description: str,
    steps: list[str] | None = None,
    tags: list[str] | None = None,
    *,
    max_words: int = MAX_SKILL_WORDS,
) -> dict[str, Any] | None:
    """Normalize a candidate skill, or return ``None`` when it must be rejected."""
    name = (name or "").strip()
    description = (description or "").strip()
    steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

    if skill_rejection_reason(name, description, steps):
        return None

    name, description, steps = _cap_length(name, description, steps, max_words)
    if not name:
        return None
    return {"name": name, "description": description, "steps": steps, "tags": tags}
