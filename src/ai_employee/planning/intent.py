"""What a goal or a step actually asks for, read from its wording.

Two questions drive download behaviour:

* Does this STEP ask to obtain files? Only then is the research agent pushed to
  call ``download_resource`` before it finishes.
* Is this GOAL a research/report ask that never requested files? Then research
  steps run without download tools, however the planner phrased them.

Both match explicit requests only — "whether weights are downloadable" or
"download/access links" describe a resource, they don't ask to fetch it.
"""

from __future__ import annotations

import re

# "download X", "acquire", "obtain", "save … to the workspace", "into the workspace".
# Not: "downloadable", "downloaded", "download links", "download/access URL",
# "available for download".
_ACQUIRE_RE = re.compile(
    r"(?<!for )(?<!available )\b(?:download|acquire|obtain)\b"
    r"(?!\s*(?:/|or\b|links?\b|urls?\b|pages?\b|sizes?\b|counts?\b|status\b))"
    r"|\bsave\b[^.\n]{0,60}\b(?:workspace|disk|locally|local (?:folder|directory))\b"
    r"|\b(?:into|in|to) the workspace\b",
    re.I,
)

# Train / evaluate / run something ON data: needs the data locally.
_RUN_ON_DATA_RE = re.compile(
    r"\b(?:train|fine-?tune|run inference|evaluate|benchmark|test)\b[^.\n]{0,80}\bon\b"
    r"[^.\n]{0,60}\b(?:dataset|data|images|videos|footage|samples|clips)\b",
    re.I,
)

# A goal whose deliverable is knowledge: research, compare, recommend, report.
_REPORT_GOAL_RE = re.compile(
    r"\b(?:research|report|compare|comparison|recommend(?:ation)?s?|identify|"
    r"investigate|survey|shortlist|which (?:model|dataset|tool|library|approach)|"
    r"best (?:model|approach|option|tool))\b",
    re.I,
)

#: Where earlier steps' outputs are appended to a step's task text.
PRIOR_MARKER = "PREREQUISITE STEPS ALREADY COMPLETED"


def own_description(task: str) -> str:
    """A step's own instructions, without earlier steps' outputs pasted after it."""
    text = task or ""
    cut = text.find(PRIOR_MARKER)
    return text[:cut] if cut >= 0 else text


def asks_to_acquire(text: str) -> bool:
    """Does ``text`` explicitly ask to download or save resources?"""
    return bool(_ACQUIRE_RE.search(text or ""))


def goal_withholds_downloads(goal: str) -> bool:
    """True for a research/report goal that never asks to obtain files or to run
    anything on data — its research steps should record links, not download."""
    g = goal or ""
    if not _REPORT_GOAL_RE.search(g):
        return False
    return not (asks_to_acquire(g) or _RUN_ON_DATA_RE.search(g))


__all__ = [
    "PRIOR_MARKER",
    "asks_to_acquire",
    "goal_withholds_downloads",
    "own_description",
]
