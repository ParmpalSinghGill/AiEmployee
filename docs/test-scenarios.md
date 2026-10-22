# Realtime test scenarios

Five end-to-end acceptance scenarios that answer one question: **given this goal,
does the employee route to the right agents and produce a genuinely good result —
not a plausible-looking fake?**

They are *realtime* in the sense that they run the **real pipeline** (real planner,
real agents, real tools, real verifier) against whatever brain you point them at —
the offline `mock` brain for a deterministic smoke run, or a real provider
(DeepSeek / OpenAI / Gemini / GLM / OpenCode / local) for true routing behaviour.

- **Runner:** [`scripts/run_scenarios.py`](../scripts/run_scenarios.py)
- **Suite:** [`tests/scenarios/`](../tests/scenarios/) — `definitions.py` (the cases),
  `checks.py` (the graders), `harness.py` (run + snapshot + grade), `fixtures.py` (seed files)
- **Self-tests:** [`tests/test_scenarios.py`](../tests/test_scenarios.py)

---

## What a scenario is

Each case pairs a natural-language **goal** with three machine-checkable claims:

| Field | Meaning |
|---|---|
| `require_agents` / `require_any` | which agents the **planner** must assign (the routing column) |
| `checks` | result-quality assertions on the answer, the tools used, and the files on disk |
| `protect` | files that must **not** change (review-only scenarios) |

The harness seeds an **isolated workspace per scenario**, runs the goal, snapshots
the disk before/after, then grades the run. Results are `pass` (all checks),
`partial` (≥50%), `fail` (<50%), or `error` (the run crashed / timed out).

Crucially, the checks are designed to catch the failure modes the scenarios warn
about — a **download claimed but never done**, **numbers invented when the file is
missing**, a **script that claims to have run**, or a **review that quietly edits files**.

---

## The five scenarios

| # | Task to give the model | Expected agents | What a good result must include |
|---|---|---|---|
| **S1** | “Find an open-source, downloadable image dataset for detecting cats and dogs. Save it in the workspace and report the source URL, local path, and download status.” | planner → research → ceo; human_approval only if needed | A suitable dataset, a verified source URL, an actual local path if the download succeeded, and an honest explanation if access is blocked. **Must not claim a download based only on search results.** |
| **S2** | “Analyze data/sales.csv. Report total sales by region, identify the strongest and weakest month, and save a chart of monthly sales.” | planner → data_analysis → ceo | Calculated totals tied to the file, the identified months, and a chart saved in the workspace with its filename in the answer. **Must not invent results if the file is missing or unreadable.** |
| **S3** | “Add a `/health` endpoint to the project in `./sampleapp`, write tests for it, and update the README with an example request and response.” | planner → coding → testing → documentation → ceo | A working endpoint, tests covering success and relevant failure behavior, a README example matching the implementation, and a clear summary of files changed and test results. |
| **S4** | “Find a pretrained model and a compatible dataset for classifying flowers. Estimate the GPU memory needed for inference, then create an evaluation script and instructions. **Do not run the script.**” | planner → research (find + VRAM estimate) → coding → ceo | Verified model and dataset references, a concrete hardware estimate with assumptions, an evaluation script and run instructions, and **no claim that evaluation ran**. |
| **S5** | “Review the deployment setup in `./deploy_review` (Dockerfile, docker-compose.yml, .github/workflows/ci.yml). Identify Docker or CI problems, review security risks, and recommend fixes. **Do not modify any files or deploy anything.**” | planner → reviewer → ceo | Findings tied to actual files and lines, severity and rationale, practical recommendations, and confirmation that the task was **review-only**. |

### Exact checks (what the grader asserts)

**S1 — download discipline**
- `s1_source_url_or_notfound` — a real `http(s)` URL **or** an explicit “not found”.
- `s1_no_false_download` — if the answer claims success (`Status: downloaded`, “saved to…”, a `Local path:` with a file), there must be a successful `download_resource` call **or** the file must actually exist on disk.
- `s1_honest_if_blocked` — if the download failed, the answer must acknowledge the blocker (403 / login / captcha / waiting / approval).

**S2 — data correctness** (ground truth is computed from the seeded CSV)
- `s2_region_totals` — every region name appears with its correct total (North 310, South 630, East 450, West 180).
- `s2_strongest_weakest_month` — the strongest (March) and weakest (January) month are named.
- `s2_chart_saved_and_referenced` — a `.png` was saved **and** its filename is in the answer.
- `s2_no_fabrication_if_missing` — if `data/sales.csv` is absent, the answer must not report totals.

**S3 — endpoint + tests + docs**
- `s3_endpoint_present` — a file declares a `/health` route (`'/health'` / `"/health"`).
- `s3_tests_present` — a test file references `/health` (or a `test_*health*`) with a `def test_`.
- `s3_readme_updated` — the README documents `/health` with a request/response example (`curl`, `GET /health`, `200`, …).
- `s3_summary_cites_files` — the answer names the changed files and mentions test results.

**S4 — research + estimate + generate, but don’t run**
- `s4_references_model`, `s4_references_dataset` — a Hugging Face / Kaggle / Roboflow URL or `id`, or an explicit “not found”.
- `s4_hardware_estimate` — a `… GB` VRAM figure.
- `s4_eval_script_created` — a new `.py` importing `torch` / `transformers` / `datasets` / using `from_pretrained` / `load_dataset`.
- `s4_did_not_run_disclosed` — the answer states the script was **not run** and does not present an executed accuracy as fact.

**S5 — review-only audit**
- `s5_references_files` — findings cite at least two of the real files (Dockerfile, docker-compose.yml, ci.yml).
- `s5_severity_language` — severity/risk ratings present.
- `s5_recommendations` — concrete recommendations present.
- `s5_review_only` — none of the protected files changed on disk.
- `s5_states_no_changes` — the answer confirms nothing was modified.

---

## How to run

### Offline smoke run (no keys, deterministic)

```bash
python scripts/run_scenarios.py --brain mock --all
```

The mock brain does not reason — it returns a single generic step — so the
**routing** checks will (correctly) fail and the run reports `partial`/`fail`. What
this proves is that the harness, the workspace isolation, the disk snapshots, and
every grader work end to end. Use it as a fast regression on the tooling.

### Realtime run (the real test)

Configure a provider in `.env` (or via the web console), then:

```bash
# everything, pre-granting approvals so runs complete unattended
python scripts/run_scenarios.py --brain real --all --approve-all \
    --report data/scenario_reports/real.md

# a single scenario
python scripts/run_scenarios.py --brain real --scenario s2_sales_analysis
```

- `--brain real` uses the provider configured in `.env` / the console (it does **not**
  force the mock), so a valid API key (or a running local model) is required.
- Without `--approve-all`, scenarios use their default grants: S1 may pre-grant
  `resource_download` / `resource_acquisition`, S2–S4 pre-grant `shell_write`, and
  **S5 grants nothing** so it stays read-only. If a run pauses, the report shows the
  pending operation — that is itself a valid observation (e.g. “human_approval only
  if needed”).
- Reports are written to `data/scenario_reports/` and are also printed to the console.

### From the web console (manual, realtime)

Run the same goals from the `/ui` console and watch the **Flow** tab: confirm the
plan’s agent badges are the expected ones, open the **Explain** tab per step to see
what each agent was asked and which tools it called, and check the **Result** tab for
the chart file, the URLs, and the final answer.

---

## Reading the report

Each scenario section lists the planned agents, executed agents, tools used, files
created/changed, and a table of every check. Example (real run, healthy):

```text
## s2_sales_analysis — Analyze sales.csv and chart monthly sales
- Status: `pass`  (7/7 checks, score 1.0)
- Planned agents: data_analysis
- Tools used: csv_preview, python_exec, write_file
| Check | Result | Detail |
| `s2_region_totals` | PASS | all region totals correct |
| `s2_chart_saved_and_referenced` | PASS | chart saved and named in the answer: monthly_sales.png |
...
```

Common **partial**/fail patterns and what they mean:

| Symptom | Likely cause |
|---|---|
| `routes_to_expected_agents` fails | the planner picked a different agent (e.g. `research` instead of `data_analysis`) — a routing/prompt issue |
| `s1_no_false_download` fails | the model reported success from search snippets without downloading — the exact hallucination this suite guards against |
| `s2_region_totals` fails | wrong arithmetic or the wrong file was analysed |
| `s2_chart_saved_and_referenced` fails | a chart was made but not named in the answer, or not saved at all |
| `s4_did_not_run_disclosed` fails | the model claimed it executed the evaluation |
| `s5_review_only` fails | the run edited a file despite the “do not modify” instruction |

---

## Recording results

Copy this block per run:

```text
Date:            <YYYY-MM-DD>
Brain/provider:  <mock | deepseek | openai | local …> / model
Approvals:       <none | --approve-all | per-scenario grants>

S1 cats/dogs dataset   : pass | partial | fail   agents=<…>  note=<…>
S2 sales analysis      : pass | partial | fail   agents=<…>  note=<…>
S3 health endpoint     : pass | partial | fail   agents=<…>  note=<…>
S4 flower model+eval   : pass | partial | fail   agents=<…>  note=<…>
S5 deploy review       : pass | partial | fail   agents=<…>  note=<…>
```

---

## Adding a scenario

1. Add a `Scenario(...)` to `tests/scenarios/definitions.py` (set `require_agents`,
   `checks`, `protect`, `grant`, and a `seed` if it needs input files).
2. Add any new check functions to `tests/scenarios/checks.py` and register them in
   the `CHECKS` dict.
3. If it needs fixtures, add a seeder to `tests/scenarios/fixtures.py`.
4. Add a good-vs-bad evaluator test to `tests/test_scenarios.py` — this is what keeps
   the grader honest.

Then run `make test` (offline) and
`python scripts/run_scenarios.py --brain real --scenario <id>`.
