"""Offline tests for the realtime scenario suite.

These keep ``make test`` green and prove three things without any network or keys:

1. the five scenarios are well-formed and reference real checks;
2. the evaluator PASSES crafted "good" evidence and FAILS crafted "bad" evidence
   for every scenario (i.e. the checks actually catch the failure modes);
3. the harness runs the real pipeline end to end against the mock brain.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Make ``scenarios`` importable the same way ``scripts/run_scenarios.py`` does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scenarios import fixtures  # noqa: E402
from scenarios.checks import CHECKS  # noqa: E402
from scenarios.definitions import SCENARIOS, scenario_by_id  # noqa: E402
from scenarios.harness import evaluate_scenario, render_report, run_scenario, snapshot  # noqa: E402
from scenarios.models import Evidence  # noqa: E402

ALL_IDS = [s.id for s in SCENARIOS]


# ── helpers ──────────────────────────────────────────────────────────────────
def _write(root: pathlib.Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _result(planned: list[str], answer: str, tools: dict | None = None, needs=None) -> dict:
    steps = [{"id": i + 1, "description": "", "agent": a} for i, a in enumerate(planned)]
    return {
        "answer": answer,
        "plan": {"goal": "", "steps": steps},
        "steps": list(steps),
        "agents_used": [
            {"agent": a, "steps": 1, "tools": [], "ok": 0, "failed": 0} for a in planned
        ],
        "tools_used": list((tools or {}).values()),
        "needs_approval": needs,
    }


def _ev(
    ws: pathlib.Path,
    sid: str,
    answer: str,
    planned: list[str],
    *,
    seed_files: dict[str, str] | None = None,
    new_files: dict[str, str] | None = None,
    mutate=None,
    tools: dict | None = None,
    needs=None,
) -> Evidence:
    for rel, content in (seed_files or {}).items():
        _write(ws, rel, content)
    before = snapshot(ws)
    for rel, content in (new_files or {}).items():
        _write(ws, rel, content)
    if mutate is not None:
        mutate(ws)
    after = snapshot(ws)
    created = sorted(set(after) - set(before))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    return Evidence(
        scenario_id=sid,
        goal="",
        mode="task",
        result=_result(planned, answer, tools, needs),
        workspace=str(ws),
        files_before=before,
        files_after=after,
        created=created,
        changed=changed,
    )


def _check(report: dict, name: str) -> dict:
    for c in report["checks"]:
        if c["name"] == name:
            return c
    raise AssertionError(f"check {name!r} not found in {[c['name'] for c in report['checks']]}")


# ── 1. definitions well-formed ───────────────────────────────────────────────
def test_five_scenarios_defined_and_unique() -> None:
    assert len(SCENARIOS) == 5
    assert len(set(ALL_IDS)) == 5
    for s in SCENARIOS:
        assert s.goal.strip(), s.id
        assert s.mode in {"task", "research", "download"}
        for name in s.checks:
            assert name in CHECKS, f"{s.id} references unknown check {name!r}"


def test_scenario_by_id_roundtrips() -> None:
    for s in SCENARIOS:
        assert scenario_by_id(s.id) is s
    with pytest.raises(KeyError):
        scenario_by_id("nope")


# ── 2. evaluator: good vs bad evidence per scenario ──────────────────────────
def test_s1_good_vs_false_download(tmp_path: pathlib.Path) -> None:
    s = scenario_by_id("s1_cats_dogs_dataset")
    dl_tool = {"tool": "download_resource", "calls": 1, "ok": 1, "failed": 0}

    good = _ev(
        tmp_path / "good",
        s.id,
        "Source: https://huggingface.co/datasets/x/cats-vs-dogs\n"
        "**Local path:** ./data/cats_dogs.zip\n**Status:** downloaded",
        ["resource_acquirer"],
        new_files={"data/cats_dogs.zip": "PK..."},
        tools={"download_resource": dl_tool},
    )
    rep = evaluate_scenario(s, good)
    assert rep["status"] == "pass", rep
    assert _check(rep, "s1_no_false_download")["ok"] is True

    # Claims a download but nothing is on disk and no download tool ran.
    bad = _ev(
        tmp_path / "bad",
        s.id,
        "Found a great dataset.\n**Status:** downloaded\n**Local path:** ./data/cats_dogs.zip",
        ["resource_acquirer"],
    )
    rep_bad = evaluate_scenario(s, bad)
    assert _check(rep_bad, "s1_no_false_download")["ok"] is False
    assert rep_bad["status"] != "pass"


def test_s2_totals_and_chart(tmp_path: pathlib.Path) -> None:
    s = scenario_by_id("s2_sales_analysis")
    good = _ev(
        tmp_path / "good",
        s.id,
        "Total sales by region: North 310, South 630, East 450, West 180.\n"
        "Strongest month: March. Weakest month: January.\n"
        "Chart: monthly_sales.png",
        ["data_analysis"],
        seed_files={"data/sales.csv": fixtures.SALES_CSV},
        new_files={"monthly_sales.png": "PNG"},
    )
    rep = evaluate_scenario(s, good)
    assert rep["status"] == "pass", rep
    assert _check(rep, "s2_region_totals")["ok"] is True
    assert _check(rep, "s2_chart_saved_and_referenced")["ok"] is True

    bad = _ev(
        tmp_path / "bad",
        s.id,
        "Total sales were about 1000. The best month was November.",
        ["data_analysis"],
        seed_files={"data/sales.csv": fixtures.SALES_CSV},
    )
    rep_bad = evaluate_scenario(s, bad)
    assert _check(rep_bad, "s2_region_totals")["ok"] is False
    assert _check(rep_bad, "s2_strongest_weakest_month")["ok"] is False
    assert _check(rep_bad, "s2_chart_saved_and_referenced")["ok"] is False


def test_s3_endpoint_tests_readme(tmp_path: pathlib.Path) -> None:
    s = scenario_by_id("s3_health_endpoint")
    app_with_health = (
        "from fastapi import FastAPI\napp = FastAPI()\n"
        "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n"
    )
    good = _ev(
        tmp_path / "good",
        s.id,
        "Added GET /health to sampleapp/app.py, created sampleapp/tests/test_health.py "
        "(2 tests pass), and updated README with a curl example.",
        ["coding", "testing", "documentation"],
        seed_files={"sampleapp/__init__.py": "", "sampleapp/app.py": "from fastapi import FastAPI\n"},
        new_files={
            "sampleapp/app.py": app_with_health,
            "sampleapp/tests/test_health.py": "def test_health():\n    assert '/health'\n",
            "sampleapp/README.md": "## GET /health\ncurl localhost:8000/health -> 200 {\"status\":\"ok\"}\n",
        },
    )
    rep = evaluate_scenario(s, good)
    assert rep["status"] == "pass", rep

    bad = _ev(
        tmp_path / "bad",
        s.id,
        "I looked at the project.",
        ["coding", "testing", "documentation"],
        seed_files={"sampleapp/app.py": "from fastapi import FastAPI\n"},
    )
    rep_bad = evaluate_scenario(s, bad)
    assert _check(rep_bad, "s3_endpoint_present")["ok"] is False
    assert _check(rep_bad, "s3_tests_present")["ok"] is False


def test_s4_script_and_did_not_run(tmp_path: pathlib.Path) -> None:
    s = scenario_by_id("s4_flower_model_eval")
    good = _ev(
        tmp_path / "good",
        s.id,
        "Model: https://huggingface.co/google/vit-base-patch16-224\n"
        "Dataset: https://huggingface.co/datasets/timm/oxford-iiit-pet\n"
        "Inference needs ~0.4 GB VRAM (fp16).\n"
        "Script written to research/eval_flowers.py. The evaluation was not run.",
        ["scout", "hardware_analyst", "codesmith"],
        new_files={
            "research/eval_flowers.py": "from transformers import AutoModelForImageClassification\n",
        },
    )
    rep = evaluate_scenario(s, good)
    assert rep["status"] == "pass", rep
    assert _check(rep, "s4_did_not_run_disclosed")["ok"] is True

    bad = _ev(
        tmp_path / "bad",
        s.id,
        "I ran the evaluation. Accuracy: 0.94.",
        ["scout", "hardware_analyst", "codesmith"],
    )
    rep_bad = evaluate_scenario(s, bad)
    assert _check(rep_bad, "s4_did_not_run_disclosed")["ok"] is False


def test_s5_review_only(tmp_path: pathlib.Path) -> None:
    s = scenario_by_id("s5_deploy_review")
    seed = {
        "deploy_review/Dockerfile": "FROM python:latest\nENV DB_PASSWORD=secret\n",
        "deploy_review/docker-compose.yml": "services:\n  api:\n    privileged: true\n",
        "deploy_review/.github/workflows/ci.yml": "test:\n  continue-on-error: true\n",
        "deploy_review/README.md": "# review\n",
    }
    answer = (
        "Security review (review-only). Critical: hardcoded DB_PASSWORD in Dockerfile. "
        "High: privileged: true in docker-compose.yml. CI: continue-on-error in ci.yml. "
        "Recommendations: pin the base image, add a USER, move secrets out. "
        "No files were modified."
    )
    good = _ev(tmp_path / "good", s.id, answer, ["devops", "security"], seed_files=seed)
    rep = evaluate_scenario(s, good)
    assert rep["status"] == "pass", rep
    assert _check(rep, "s5_review_only")["ok"] is True

    def _edit(ws: pathlib.Path) -> None:
        _write(ws, "deploy_review/Dockerfile", "FROM python:3.12-slim\n")

    bad = _ev(
        tmp_path / "bad",
        s.id,
        "I edited the Dockerfile to fix the issues.",
        ["devops", "security"],
        seed_files=seed,
        mutate=_edit,
    )
    rep_bad = evaluate_scenario(s, bad)
    assert _check(rep_bad, "s5_review_only")["ok"] is False
    assert _check(rep_bad, "s5_states_no_changes")["ok"] is False


def test_routing_check_fails_when_planner_picks_wrong_agent(tmp_path: pathlib.Path) -> None:
    s = scenario_by_id("s2_sales_analysis")
    ev = _ev(
        tmp_path / "ws",
        s.id,
        "done",
        ["coding"],  # wrong agent
        seed_files={"data/sales.csv": fixtures.SALES_CSV},
    )
    rep = evaluate_scenario(s, ev)
    assert _check(rep, "routes_to_expected_agents")["ok"] is False


# ── 3. harness end-to-end (offline) ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_harness_runs_review_scenario_offline(tmp_path: pathlib.Path) -> None:
    """The whole harness (seed → run → snapshot → grade) works with the mock brain."""
    s = scenario_by_id("s5_deploy_review")
    ev = await run_scenario(s, root=tmp_path / "runs", brain="mock", timeout=120)
    assert ev.error is None, ev.error
    # It ran a real plan and produced an answer.
    assert ev.plan_steps, "planner produced no steps"
    assert ev.answer, "no answer produced"
    rep = evaluate_scenario(s, ev)
    assert rep["status"] in {"pass", "partial", "fail"}  # not an error
    # Mock never writes, so the review-only invariant must hold.
    assert _check(rep, "s5_review_only")["ok"] is True

    md = render_report([rep], meta={"brain": "mock", "provider": "mock"})
    assert "s5_deploy_review" in md and "Status" in md
