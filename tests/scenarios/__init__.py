"""Realtime acceptance-test scenarios for the AI Employee.

Each scenario pairs a natural-language goal with the agent routing and result
properties a *good* run must exhibit, then runs it through the real pipeline and
scores the outcome. See ``docs/test-scenarios.md`` for the human-readable spec.

Run it with::

    python scripts/run_scenarios.py --brain mock --all
    python scripts/run_scenarios.py --brain real --scenario s2_sales_analysis

The package is importable as ``scenarios`` (add ``tests/`` to ``sys.path``) and is
exercised by ``tests/test_scenarios.py``.
"""

from __future__ import annotations

from .definitions import SCENARIOS, scenario_by_id
from .harness import render_report, run_all, run_scenario
from .models import Check, Evidence, Scenario

__all__ = [
    "Check",
    "Evidence",
    "Scenario",
    "SCENARIOS",
    "scenario_by_id",
    "run_scenario",
    "run_all",
    "render_report",
]
