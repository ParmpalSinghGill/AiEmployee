#!/usr/bin/env python3
"""Run the AI Employee realtime acceptance-test scenarios and grade them.

Examples
--------
Offline smoke run (deterministic mock brain — proves the harness end to end)::

    python scripts/run_scenarios.py --brain mock --all

Realtime run against the configured provider in ``.env`` (DeepSeek/OpenAI/local…)::

    python scripts/run_scenarios.py --brain real --all --approve-all

A single scenario, writing a Markdown report::

    python scripts/run_scenarios.py --brain real --scenario s2_sales_analysis \
        --report data/scenario_reports/s2.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make ``scenarios`` (under tests/) and the src package importable when run directly.
for extra in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scenarios.definitions import SCENARIOS, scenario_by_id  # noqa: E402
from scenarios.harness import render_report, run_all  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AI Employee acceptance-test scenarios")
    p.add_argument("--brain", choices=["mock", "real"], default="mock",
                   help="mock = offline deterministic brain; real = the provider configured in .env")
    p.add_argument("--all", action="store_true", help="Run every scenario")
    p.add_argument("--scenario", action="append", default=[], metavar="ID",
                   help="Run a specific scenario id (repeatable)")
    p.add_argument("--approve-all", action="store_true",
                   help="Pre-grant every approval key so runs complete unattended")
    p.add_argument("--root", default=str(REPO_ROOT / "data" / "scenario_runs"),
                   help="Directory for per-scenario workspaces")
    p.add_argument("--report", default=None, help="Write the Markdown report here")
    p.add_argument("--timeout", type=float, default=900.0, help="Per-scenario timeout (s)")
    return p


def _selected(args) -> list:
    if args.all or not args.scenario:
        return list(SCENARIOS)
    return [scenario_by_id(s) for s in args.scenario]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    scenarios = _selected(args)

    provider = "mock"
    if args.brain == "real":
        from ai_employee.config import get_settings

        provider = get_settings().llm_provider.value

    print(f"Running {len(scenarios)} scenario(s) with brain={args.brain} (provider={provider})")
    reports = asyncio.run(
        run_all(
            root=Path(args.root),
            scenarios=scenarios,
            brain=args.brain,
            approve_all=args.approve_all,
            timeout=args.timeout,
        )
    )

    report_md = render_report(
        reports,
        meta={"brain": args.brain, "provider": provider,
              "note": "mock brain does not reason; use --brain real for true routing."},
    )
    print(report_md)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_md, encoding="utf-8")
        print(f"\nReport written to {out}")

    # Non-zero only on a hard error (crash), not on graded failures.
    return 1 if any(r["status"] == "error" for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
