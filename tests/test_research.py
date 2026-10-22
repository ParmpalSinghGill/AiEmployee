"""Research-assistant tools and decision-tree workflow (offline)."""

from __future__ import annotations

import pytest

from ai_employee.workflows.research_graph import (
    RUN_APPROVAL_KEY,
    DatasetHit,
    Discovery,
    ModelHit,
    ResearchOrchestrator,
)


# ── tools ────────────────────────────────────────────────────────────────────
async def test_research_tools_registered(registry):
    names = set(registry.names())
    assert {
        "hf_search",
        "paperswithcode_search",
        "arxiv_search",
        "github_search",
        "kaggle_search",
        "roboflow_search",
        "estimate_hardware",
    } <= names


async def test_estimate_hardware_from_params(registry):
    res = await registry.execute("estimate_hardware", {"params_billions": 7.0, "dtype": "fp16"})
    assert res.ok
    # 7B fp16 ≈ 14GB weights → ~16.8 with overhead; training ≈ 104GB.
    assert 14 < res.output["inference_vram_gb"] < 20
    assert res.output["full_finetune_vram_gb"] > res.output["inference_vram_gb"]
    assert "recommendation" in res.output


async def test_estimate_hardware_needs_input(registry):
    res = await registry.execute("estimate_hardware", {})
    assert not res.ok


async def test_discovery_tools_degrade_offline(registry):
    # No network in tests: tools must return cleanly, never raise.
    for tool, args in [
        ("hf_search", {"query": "pneumonia", "limit": 3}),
        ("arxiv_search", {"query": "pneumonia", "limit": 2}),
        ("paperswithcode_search", {"query": "pneumonia"}),
        ("github_search", {"query": "pneumonia"}),
    ]:
        res = await registry.execute(tool, args)
        assert res.output is not None  # ok or graceful empty, but no crash


async def test_kaggle_without_creds_is_graceful(registry):
    res = await registry.execute("kaggle_search", {"query": "x"})
    assert res.ok and res.output == []


async def test_roboflow_without_key_is_graceful(registry):
    res = await registry.execute("roboflow_search", {"query": "helmet detection"})
    assert res.ok and res.output == []


# ── decision tree ────────────────────────────────────────────────────────────
def _orch(employee) -> ResearchOrchestrator:
    return employee.research_orchestrator


@pytest.mark.parametrize(
    "models,datasets,expected",
    [
        ([ModelHit(id="m")], [DatasetHit(id="d")], "both"),
        ([ModelHit(id="m")], [], "model_only"),
        ([], [DatasetHit(id="d")], "data_only"),
        ([], [], "neither"),
    ],
)
def test_branch_decision(employee, models, datasets, expected):
    orch = _orch(employee)
    state = {"discovery": Discovery(models=models, datasets=datasets)}
    out = orch._decide_node(state)
    assert out["branch"] == expected


async def test_research_runs_end_to_end(employee):
    out = await employee.research("classify chest x-rays for pneumonia")
    r = out["result"]
    assert r["branch"] in {"both", "model_only", "data_only", "neither"}
    assert "found_model" in r and "found_dataset" in r and "papers" in r
    assert r["generated"]["instructions"]
    assert {t.get("node") for t in out["trace"]} >= {"discover", "decide", "generate"}


async def test_run_is_approval_gated(employee):
    # auto_run requested but not approved → surfaces an approval request, does not run.
    out = await employee.research("x", auto_run=True)
    assert out["needs_approval"]
    assert out["needs_approval"]["operation"] == RUN_APPROVAL_KEY
    assert out["result"]["accuracy"] is None


# ── token accounting ─────────────────────────────────────────────────────────
def test_token_meter_records_and_buckets():
    from ai_employee.brains.schemas import Usage
    from ai_employee.core.usage import TokenMeter

    m = TokenMeter()
    m.record(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15), "agent:a")
    m.record(Usage(prompt_tokens=20, completion_tokens=0, total_tokens=20), "agent:a")
    m.record(Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2), "agent:b")
    snap = m.snapshot()
    assert snap["total_tokens"] == 37
    assert snap["calls"] == 3
    assert snap["by_purpose"]["agent:a"]["total_tokens"] == 35
    assert snap["by_purpose"]["agent:a"]["calls"] == 2


def test_record_usage_noop_without_meter():
    # Outside a track_usage block this must not raise.
    from ai_employee.brains.schemas import Usage
    from ai_employee.core.usage import record_usage

    record_usage(Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))


async def test_research_reports_total_tokens(employee):
    out = await employee.research("classify chest x-rays for pneumonia")
    tok = out["tokens"]
    assert tok["total_tokens"] > 0 and tok["calls"] > 0
    # The total equals the sum of the per-agent buckets (no double counting / drops).
    assert tok["total_tokens"] == sum(b["total_tokens"] for b in tok["by_purpose"].values())
    # Every agent in the run is attributed (discovery always runs on research).
    assert any(k.startswith("agent:research") for k in tok["by_purpose"])
    # Surfaced on the result for the UI too.
    assert out["result"]["tokens"] == tok


async def test_main_run_reports_tokens(employee):
    out = await employee.run("say hello")
    assert out["tokens"]["total_tokens"] > 0


# ── API key management ───────────────────────────────────────────────────────
def _fresh_settings(tmp_path):
    from ai_employee.config import reload_settings

    s = reload_settings()
    s.secrets_file = tmp_path / "secrets.json"
    return s


def test_set_list_delete_key(tmp_path):
    from ai_employee.config.secrets_store import delete_key, list_keys, set_key

    s = _fresh_settings(tmp_path)
    set_key(s, "roboflow_api_key", "rf_ABCD1234SECRET")
    assert s.roboflow_api_key == "rf_ABCD1234SECRET"  # applied to live settings
    row = {k["name"]: k for k in list_keys(s)}["roboflow_api_key"]
    assert row["configured"] and row["masked"] == "••••CRET" and "SECRET" not in row["masked"]
    # Persisted to disk.
    assert (tmp_path / "secrets.json").exists()
    delete_key(s, "roboflow_api_key")
    assert s.roboflow_api_key == ""


def test_username_not_masked(tmp_path):
    from ai_employee.config.secrets_store import list_keys, set_key

    s = _fresh_settings(tmp_path)
    set_key(s, "kaggle_username", "andy")
    row = {k["name"]: k for k in list_keys(s)}["kaggle_username"]
    assert row["masked"] == "andy" and row["secret"] is False


def test_unknown_key_rejected(tmp_path):
    from ai_employee.config.secrets_store import set_key

    s = _fresh_settings(tmp_path)
    with pytest.raises(KeyError):
        set_key(s, "not_a_key", "x")


async def test_search_tool_picks_up_key_without_rebuild(registry, settings):
    # roboflow_search reads the key at call-time, so setting it takes effect live.
    from ai_employee.config.secrets_store import set_key

    res = await registry.execute("roboflow_search", {"query": "x"})
    assert res.output == []  # no key yet
    set_key(settings, "roboflow_api_key", "rf_live_key")
    res2 = await registry.execute("roboflow_search", {"query": "x"})
    # Now it attempts the call (and fails offline) rather than the no-key short-circuit.
    assert res2.metadata.get("note") is None


async def test_keys_api_roundtrip(client):
    listed = (await client.get("/ui/api/keys")).json()["keys"]
    assert any(k["name"] == "roboflow_api_key" for k in listed)
    r = await client.put("/ui/api/keys/roboflow_api_key", json={"value": "rf_TESTKEY99"})
    assert r.status_code == 200
    rows = {k["name"]: k for k in (await client.get("/ui/api/keys")).json()["keys"]}
    assert rows["roboflow_api_key"]["configured"]
    assert rows["roboflow_api_key"]["masked"].endswith("EY99")
    d = await client.delete("/ui/api/keys/roboflow_api_key")
    assert d.status_code == 200
    rows = {k["name"]: k for k in (await client.get("/ui/api/keys")).json()["keys"]}
    assert not rows["roboflow_api_key"]["configured"]


async def test_keys_api_unknown_name_404(client):
    r = await client.put("/ui/api/keys/bogus", json={"value": "x"})
    assert r.status_code == 404


# ── research surfaced through the UI task runner ─────────────────────────────
async def _wait(client, task_id, timeout=15.0):
    import asyncio

    for _ in range(int(timeout / 0.1)):
        snap = (await client.get(f"/ui/api/tasks/{task_id}")).json()
        if snap["status"] in {"completed", "failed", "awaiting_approval"}:
            return snap
        await asyncio.sleep(0.1)
    raise AssertionError("task did not finish")


async def test_research_mode_task_surfaces_findings_and_tokens(client):
    r = await client.post(
        "/ui/api/tasks", json={"goal": "detect pneumonia in chest x-rays", "mode": "research"}
    )
    snap = await _wait(client, r.json()["task_id"])
    assert snap["mode"] == "research"
    result = snap["result"]
    # Found resources are present for the UI's Result panel.
    assert "found_model" in result["result"] and "found_dataset" in result["result"]
    assert result["result"]["branch"] in {"both", "model_only", "data_only", "neither"}
    # Token totals + per-agent breakdown are surfaced for display.
    assert result["tokens"]["total_tokens"] > 0
    assert result["tokens"]["by_purpose"]


async def test_default_mode_is_task(client):
    r = await client.post("/ui/api/tasks", json={"goal": "say hi"})
    snap = await _wait(client, r.json()["task_id"])
    assert snap["mode"] == "task"
    assert snap["result"]["tokens"]["total_tokens"] > 0  # tokens shown for normal tasks too


# ── key testing ──────────────────────────────────────────────────────────────
async def test_key_tester_no_value_is_clear(settings):
    from ai_employee.config.key_tester import test_key

    res = await test_key(settings, "roboflow_api_key")  # nothing stored, nothing passed
    assert res["ok"] is False and "no key" in res["message"].lower()


async def test_key_tester_kaggle_needs_both(settings):
    from ai_employee.config.key_tester import test_key

    # Only the key, no username → must say both are needed (no network attempted).
    res = await test_key(settings, "kaggle_key", value="some_token")
    assert res["ok"] is False and "username" in res["message"].lower()


async def test_key_tester_unknown_raises(settings):
    from ai_employee.config.key_tester import test_key

    with pytest.raises(KeyError):
        await test_key(settings, "bogus")


async def test_key_test_endpoint(client):
    r = await client.post("/ui/api/keys/roboflow_api_key/test", json={})
    assert r.status_code == 200 and r.json()["ok"] is False  # no key configured
    assert (await client.post("/ui/api/keys/bogus/test", json={})).status_code == 404


# ── per-brain usage ledger, viewable by date ─────────────────────────────────
def test_usage_ledger_aggregates_by_provider_and_date(tmp_path):
    from ai_employee.brains.schemas import Usage
    from ai_employee.observability.usage_ledger import make_sink, query

    path = tmp_path / "usage.jsonl"
    sink = make_sink(path)
    sink(
        "deepseek",
        "deepseek-chat",
        Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        "agent:scout",
    )
    sink(
        "deepseek",
        "deepseek-chat",
        Usage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        "agent:codesmith",
    )
    sink(
        "openai", "gpt-4o", Usage(prompt_tokens=30, completion_tokens=5, total_tokens=35), "planner"
    )

    agg = query(path)
    assert agg["total"]["total_tokens"] == 215 and agg["total"]["calls"] == 3
    assert agg["by_provider"]["deepseek"]["total_tokens"] == 180
    assert agg["by_provider"]["openai"]["total_tokens"] == 35
    assert agg["days"] == 1


def test_usage_ledger_date_filter_excludes_out_of_range(tmp_path):
    from ai_employee.observability.usage_ledger import query

    path = tmp_path / "usage.jsonl"
    path.write_text(
        '{"date":"2026-06-01","provider":"a","prompt_tokens":1,"completion_tokens":1,"total_tokens":10}\n'
        '{"date":"2026-06-15","provider":"a","prompt_tokens":1,"completion_tokens":1,"total_tokens":20}\n'
        '{"date":"2026-07-01","provider":"a","prompt_tokens":1,"completion_tokens":1,"total_tokens":40}\n'
    )
    mid = query(path, start="2026-06-10", end="2026-06-30")
    assert mid["total"]["total_tokens"] == 20  # only the 06-15 row


async def test_run_writes_to_usage_ledger(employee, settings):
    from ai_employee.observability.usage_ledger import query

    await employee.run("say hi")
    agg = query(settings.usage_ledger_file)
    # The mock brain registers as provider "mock"; tokens land in the ledger by brain.
    assert agg["total"]["total_tokens"] > 0
    assert "mock" in agg["by_provider"]


async def test_usage_endpoint(client):
    r = await client.post("/ui/api/tasks", json={"goal": "say hi"})
    await _wait(client, r.json()["task_id"])
    u = (await client.get("/ui/api/usage")).json()
    assert u["total"]["total_tokens"] > 0
    assert u["by_provider"] and u["by_date"]
