"""The five acceptance scenarios, as data.

The ``table`` field mirrors the spec row from the task; the machine-readable
fields (``require_agents`` / ``require_any`` / ``checks`` / ``protect``) drive the
automated grading in ``harness.evaluate_scenario``.
"""

from __future__ import annotations

from .fixtures import seed_deployment, seed_sales, seed_sample_project
from .models import Scenario

_APPROVE_WRITE = frozenset({"shell_write"})

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="s1_cats_dogs_dataset",
        title="Find & save a cats/dogs image dataset",
        goal=(
            "Find an open-source, downloadable image dataset for detecting cats and "
            "dogs. Save it in the workspace and report the source URL, local path, "
            "and download status."
        ),
        table=(
            "planner -> research -> ceo; human_approval only if "
            "needed. A suitable dataset, a verified source URL, an actual local path if "
            "the download succeeded, and an honest explanation if access is blocked. It "
            "should NOT claim a download based only on search results."
        ),
        mode="task",
        require_agents=("research",),
        checks=("s1_source_url_or_notfound", "s1_no_false_download", "s1_honest_if_blocked"),
        grant=frozenset(
            {"shell_write", "resource_download", "resource_acquisition", "waiting_for_user"}
        ),
        notes="Network + approval heavy; the key failure mode is a false download claim.",
    ),
    Scenario(
        id="s2_sales_analysis",
        title="Analyze sales.csv and chart monthly sales",
        goal=(
            "Analyze data/sales.csv. Report total sales by region, identify the "
            "strongest and weakest month, and save a chart of monthly sales."
        ),
        table=(
            "planner -> data_analysis -> ceo. Calculated totals tied to the file, the "
            "identified months, and a chart saved in the workspace with its filename in "
            "the answer. It should NOT invent results if the file is missing or unreadable."
        ),
        mode="task",
        require_agents=("data_analysis",),
        checks=(
            "s2_region_totals",
            "s2_strongest_weakest_month",
            "s2_chart_saved_and_referenced",
            "s2_no_fabrication_if_missing",
        ),
        grant=_APPROVE_WRITE,
        seed=seed_sales,
        notes="Fully offline; ground truth is computed from the seeded CSV.",
    ),
    Scenario(
        id="s3_health_endpoint",
        title="Add a /health endpoint, tests, and README example",
        goal=(
            "Add a /health endpoint to the project in ./sampleapp, write tests for it, "
            "and update the README with an example request and response."
        ),
        table=(
            "planner -> coding -> testing -> documentation -> ceo. A working endpoint, "
            "tests covering success and relevant failure behavior, a README example "
            "matching the implementation, and a clear summary of files changed and test "
            "results."
        ),
        mode="task",
        require_agents=("coding", "testing", "documentation"),
        checks=(
            "s3_endpoint_present",
            "s3_tests_present",
            "s3_readme_updated",
            "s3_summary_cites_files",
        ),
        grant=_APPROVE_WRITE,
        seed=seed_sample_project,
        notes="Targets the seeded ./sampleapp project so the sandbox can read/write it.",
    ),
    Scenario(
        id="s4_flower_model_eval",
        title="Find flower model+dataset, estimate VRAM, write eval (do not run)",
        goal=(
            "Find a pretrained model and a compatible dataset for classifying flowers. "
            "Estimate the GPU memory needed for inference, then create an evaluation "
            "script and instructions. Do not run the script."
        ),
        table=(
            "planner -> research (find + VRAM) -> coding -> ceo. Verified model "
            "and dataset references, a concrete hardware estimate with assumptions, an "
            "evaluation script and run instructions, and NO claim that evaluation ran."
        ),
        mode="task",
        require_agents=("research", "coding"),
        checks=(
            "s4_references_model",
            "s4_references_dataset",
            "s4_hardware_estimate",
            "s4_eval_script_created",
            "s4_did_not_run_disclosed",
        ),
        grant=_APPROVE_WRITE,
        notes="Network for discovery; must not claim the eval ran.",
    ),
    Scenario(
        id="s5_deploy_review",
        title="Review-only deployment & security audit",
        goal=(
            "Review the deployment setup in ./deploy_review (Dockerfile, "
            "docker-compose.yml, .github/workflows/ci.yml). Identify Docker or CI "
            "problems, review security risks, and recommend fixes. Do not modify any "
            "files or deploy anything."
        ),
        table=(
            "planner -> reviewer -> ceo. Findings tied to "
            "actual files and lines, severity and rationale, practical recommendations, "
            "and confirmation that the task was review-only."
        ),
        mode="task",
        require_agents=("reviewer",),
        checks=(
            "s5_references_files",
            "s5_severity_language",
            "s5_recommendations",
            "s5_review_only",
            "s5_states_no_changes",
        ),
        protect=(
            "deploy_review/Dockerfile",
            "deploy_review/docker-compose.yml",
            "deploy_review/.github/workflows/ci.yml",
            "deploy_review/README.md",
        ),
        # No approval grants: the run must stay read-only.
        grant=frozenset(),
        seed=seed_deployment,
        notes="No write approvals are granted, so any attempted edit halts (still review-only).",
    ),
)


def scenario_by_id(scenario_id: str) -> Scenario:
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    raise KeyError(f"unknown scenario: {scenario_id!r}")


__all__ = ["SCENARIOS", "scenario_by_id"]
