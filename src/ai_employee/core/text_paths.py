"""Text helpers for extracting real local-path claims from agent output.

Agent answers routinely quote URLs and git diffs. A naive path regex sees
``https://github.com/org/repo.git`` as the local path ``/github.com/org/repo.git``
and ``a/src/app.py`` from a diff header as a real ``a/`` directory, which makes a
filesystem audit report phantom missing files. These helpers remove those regions
before extraction so audits compare only genuine local claims.
"""

from __future__ import annotations

import re

# URLs whose path component looks like an absolute Unix path. ``git://`` and
# ``ssh://`` are included because clone commands use them too.
_URL_RE = re.compile(r"\b(?:https?|git|ssh)://[^\s<>\"'`\)\]]+", re.I)

# scp-style git remotes: git@github.com:org/repo.git
_SCP_GIT_RE = re.compile(r"\bgit@[\w.-]+:[\w./@~-]+")

# Git diff header lines. Their ``a/`` / ``b/`` prefixes are diff notation, not
# real directories on disk.
_DIFF_HEADER_RE = re.compile(
    r"(?m)^(?:"
    r"diff --git\b.*|"
    r"---\s+[ab]/.*|"
    r"\+\+\+\s+[ab]/.*|"
    r"rename from\b.*|"
    r"rename to\b.*|"
    r"index\s+[0-9a-f]+\.\.[0-9a-f]+(?:\s+\d+)?"
    r")$"
)

# A fenced diff/patch block, or a bare ``diff --git`` header.
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)\b.*?```", re.S | re.I)
_BARE_DIFF_RE = re.compile(r"(?m)^diff --git\s", re.I)


def strip_urls(text: str) -> str:
    """Replace URLs (including scp-style git remotes) with whitespace."""
    if not text:
        return text or ""
    return _SCP_GIT_RE.sub(" ", _URL_RE.sub(" ", text))


def strip_git_diff_headers(text: str) -> str:
    """Remove git diff header lines so ``a/`` and ``b/`` are not read as paths."""
    return _DIFF_HEADER_RE.sub("", text or "")


def clean_for_path_extraction(text: str) -> str:
    """Return ``text`` with URLs and diff headers removed."""
    return strip_git_diff_headers(strip_urls(text))


def has_diff_claim(text: str) -> bool:
    """True when the output presents a git diff (fenced or bare header)."""
    if not text:
        return False
    return bool(_DIFF_FENCE_RE.search(text) or _BARE_DIFF_RE.search(text))
