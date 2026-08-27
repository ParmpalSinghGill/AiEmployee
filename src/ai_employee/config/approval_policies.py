"""Per-operation approval policies — rule-based auto-approval with quotas.

The blanket choices today are "gate everything" or "auto-approve everything".
Policies sit in between: for one approval key (``email_send``,
``calendar_write``, ``mcp_tool``, …) the operator can say *auto-approve, but
only to these recipients, and at most N times a day*. Rules live in a JSON file
managed from the console and are enforced in ``ToolRegistry.execute`` — outside
the model, so a prompt can't argue past them.

Policy shape (all fields optional)::

    {"email_send": {
        "mode": "auto",                      # "gate" (default) | "auto"
        "allow_recipients": ["*@my.co", "me@gmail.com"],  # for to/attendees args
        "daily_limit": 20                    # successful actions/day, from audit
    }}
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Any

from ai_employee.core.logging import get_logger

log = get_logger("approval_policy")

#: Argument names that carry recipient-like addresses worth matching.
_RECIPIENT_ARGS = ("to", "cc", "bcc", "attendees", "recipient", "recipients")


def load_policies(settings: Any) -> dict[str, dict[str, Any]]:
    p = Path(settings.approval_policies_file)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a bad file means "no policies"
        log.warning("approval_policy.load_failed", error=str(exc))
        return {}


def save_policies(settings: Any, policies: dict[str, dict[str, Any]]) -> None:
    p = Path(settings.approval_policies_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(policies, indent=2))


def _addresses(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in _addresses(v)]
    if isinstance(value, str):
        return [a.strip() for a in re.split(r"[,;\s]+", value) if "@" in a]
    return []


def _recipients_allowed(patterns: list[str], arguments: dict[str, Any]) -> bool:
    """Every recipient-like address in the args must match some pattern."""
    addresses = [
        addr for name in _RECIPIENT_ARGS for addr in _addresses(arguments.get(name))
    ]
    if not addresses:  # nothing recipient-shaped to check — patterns don't apply
        return True
    return all(
        any(fnmatch.fnmatch(addr.lower(), pat.lower()) for pat in patterns)
        for addr in addresses
    )


def auto_allows(settings: Any, approval_key: str, arguments: dict[str, Any]) -> bool:
    """True when the operator's policy grants this call without a human click."""
    policy = load_policies(settings).get(approval_key)
    if not policy or policy.get("mode") != "auto":
        return False
    patterns = [p for p in policy.get("allow_recipients") or [] if str(p).strip()]
    if patterns and not _recipients_allowed([str(p) for p in patterns], arguments):
        log.info("approval_policy.recipient_blocked", key=approval_key)
        return False
    limit = policy.get("daily_limit")
    if limit:
        from ai_employee.observability.audit import count_today

        used = count_today(settings, approval_key)
        if used >= int(limit):
            log.info("approval_policy.quota_exhausted", key=approval_key, used=used)
            return False
    return True


__all__ = ["auto_allows", "load_policies", "save_policies"]
