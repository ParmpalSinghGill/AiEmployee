"""Autonomy features: budgets, audit trail, approval policies, schedules, notify.

All offline: the mock brain provides usage numbers, schedules fire against a
fake task manager, and no network is touched (ntfy is off without a topic).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ai_employee.config.approval_policies import auto_allows, save_policies
from ai_employee.core.exceptions import ApprovalRequired, BudgetExceeded
from ai_employee.core.usage import track_usage
from ai_employee.observability.audit import count_today, read_actions, record_action
from ai_employee.observability.schedules import (
    SchedulerRunner,
    create_schedule,
    delete_schedule,
    load_schedules,
    update_schedule,
)


# ── Token budgets ────────────────────────────────────────────────────────────
async def test_budget_stops_llm_calls(brain):
    from ai_employee.brains.metering import MeteredBrain
    from ai_employee.brains.schemas import Message

    metered = MeteredBrain(brain)
    with track_usage(budget=1) as meter:
        await metered.chat([Message.user("hello")])  # spends > 0 tokens
        assert meter.total_tokens >= 1
        with pytest.raises(BudgetExceeded):
            await metered.chat([Message.user("again")])
    # Outside the budgeted block calls work again.
    await metered.chat([Message.user("fine now")])


async def test_nested_meters_both_record(brain):
    from ai_employee.brains.metering import MeteredBrain
    from ai_employee.brains.schemas import Message

    metered = MeteredBrain(brain)
    with track_usage() as outer:
        with track_usage() as inner:
            await metered.chat([Message.user("hi")])
        assert inner.total_tokens > 0
    assert outer.total_tokens == inner.total_tokens  # outer saw the nested call


# ── Audit trail ──────────────────────────────────────────────────────────────
def test_audit_roundtrip(settings):
    assert read_actions(settings) == []
    record_action(
        settings,
        tool="gmail_send",
        approval_key="email_send",
        arguments={"to": "a@x.com", "body": "x" * 1000},
        ok=True,
        approved_by="human",
        task_id="t1",
    )
    record_action(
        settings,
        tool="shell",
        approval_key="shell_write",
        arguments={"cmd": "rm x"},
        ok=False,
        approved_by="policy",
    )
    actions = read_actions(settings)
    assert len(actions) == 2 and actions[0]["tool"] == "shell"  # newest first
    assert actions[1]["args"]["body"].endswith("…")  # long args clipped
    assert count_today(settings, "email_send") == 1
    assert count_today(settings, "shell_write") == 0  # failed calls don't count


# ── Approval policies ────────────────────────────────────────────────────────
def test_policy_modes_and_recipients(settings):
    assert not auto_allows(settings, "email_send", {"to": "a@x.com"})  # no policy
    save_policies(settings, {"email_send": {"mode": "gate"}})
    assert not auto_allows(settings, "email_send", {"to": "a@x.com"})
    save_policies(
        settings,
        {"email_send": {"mode": "auto", "allow_recipients": ["*@x.com", "me@gmail.com"]}},
    )
    assert auto_allows(settings, "email_send", {"to": "a@x.com, me@gmail.com"})
    assert not auto_allows(settings, "email_send", {"to": "a@x.com, evil@other.com"})
    # Attendee-style lists are matched too.
    assert not auto_allows(settings, "calendar_write", {"attendees": ["a@x.com"]})
    save_policies(settings, {"calendar_write": {"mode": "auto"}})
    assert auto_allows(settings, "calendar_write", {"attendees": ["anyone@anywhere"]})


def test_policy_daily_quota(settings):
    save_policies(settings, {"email_send": {"mode": "auto", "daily_limit": 2}})
    for _ in range(2):
        assert auto_allows(settings, "email_send", {"to": "a@x.com"})
        record_action(
            settings,
            tool="gmail_send",
            approval_key="email_send",
            arguments={"to": "a@x.com"},
            ok=True,
            approved_by="policy",
        )
    assert not auto_allows(settings, "email_send", {"to": "a@x.com"})  # quota spent


async def test_registry_policy_auto_approves_and_audits(settings, registry, monkeypatch):
    from tests.test_google_workspace import _write_client_file

    _write_client_file(settings)
    from ai_employee.tools.builtins.google_workspace import build_google_tools

    for t in build_google_tools(settings):
        registry.register(t, replace=True)

    async def fake_api(*a, **k):
        return {"id": "sent1"}

    monkeypatch.setattr("ai_employee.tools.builtins.google_workspace._api", fake_api)

    args = {"to": "boss@x.com", "subject": "s", "body": "b"}
    with pytest.raises(ApprovalRequired):
        await registry.execute("gmail_send", args)

    save_policies(settings, {"email_send": {"mode": "auto", "allow_recipients": ["*@x.com"]}})
    result = await registry.execute("gmail_send", args)
    assert result.ok
    actions = read_actions(settings, key="email_send")
    assert actions and actions[0]["approved_by"] == "policy"

    with pytest.raises(ApprovalRequired):  # outside the allowlist still gates
        await registry.execute("gmail_send", {**args, "to": "evil@other.com"})


# ── Schedules ────────────────────────────────────────────────────────────────
class _FakeTM:
    def __init__(self):
        self.started = []

    def start(self, emp, goal, **kwargs):
        class _T:
            id = f"task{len(self.started)}"

            class status:  # noqa: N801 - mimic TaskStatus enum member
                value = "completed"

        self.started.append({"goal": goal, **kwargs})
        return _T()

    def get(self, tid):
        return None


def test_schedule_create_validation_and_next_run(settings):
    with pytest.raises(ValueError):
        create_schedule(settings, {"type": "recurring", "goal": "x"})  # no time
    with pytest.raises(ValueError):
        create_schedule(settings, {"type": "nope", "goal": "x"})
    entry = create_schedule(
        settings, {"type": "recurring", "goal": "daily brief", "at": "08:00"}
    )
    assert entry["next_run"].endswith("08:00:00")
    every = create_schedule(
        settings, {"type": "recurring", "goal": "poll", "every_minutes": 15}
    )
    assert every["every_minutes"] == 15 and every["next_run"]
    assert len(load_schedules(settings)) == 2
    update_schedule(settings, entry["id"], {"enabled": False})
    assert not load_schedules(settings)[entry["id"]]["enabled"]
    assert delete_schedule(settings, every["id"])


async def test_scheduler_fires_due_and_disables_once(settings):
    tm = _FakeTM()
    runner = SchedulerRunner(settings, lambda: object(), tm)
    past = (datetime.now() - timedelta(minutes=1)).isoformat(timespec="seconds")
    create_schedule(
        settings,
        {"type": "once", "goal": "one shot", "run_at": past, "auto_approve": True},
    )
    fired = await runner.tick()
    assert len(fired) == 1 and tm.started[0]["goal"] == "one shot"
    assert tm.started[0]["auto_approve"] is True
    entry = next(iter(load_schedules(settings).values()))
    assert not entry["enabled"] and entry["runs"] == 1  # once → self-disables
    assert await runner.tick() == []  # doesn't fire again


async def test_gmail_watch_baselines_then_fires(settings):
    class _Tools:
        def __init__(self):
            self.outputs = []

        async def execute(self, name, args, **kw):
            from ai_employee.tools.base import ToolResult

            assert name == "gmail_search"
            return ToolResult(ok=True, output=self.outputs.pop(0))

    class _Emp:
        tools = _Tools()

    emp = _Emp()
    tm = _FakeTM()
    runner = SchedulerRunner(settings, lambda: emp, tm)
    entry = create_schedule(
        settings,
        {"type": "gmail_watch", "goal": "triage new mail", "query": "is:unread",
         "interval_minutes": 1},
    )
    # Make it due immediately.
    update_schedule(settings, entry["id"], {"interval_minutes": 1})
    sched = load_schedules(settings)
    sched[entry["id"]]["next_run"] = datetime.now().isoformat(timespec="seconds")

    emp.tools.outputs = ["[aaa1] old mail\n    snippet"]
    assert await runner._check_gmail(sched[entry["id"]], datetime.now()) is False
    assert sched[entry["id"]]["seen_ids"] == ["aaa1"]  # baselined, no task fired
    assert tm.started == []

    emp.tools.outputs = ["[bbb2] new mail\n    snippet\n[aaa1] old\n    snippet"]
    assert await runner._check_gmail(sched[entry["id"]], datetime.now()) is True
    assert len(tm.started) == 1 and "bbb2" in tm.started[0]["goal"]


# ── API endpoints ────────────────────────────────────────────────────────────
async def test_schedule_and_autonomy_endpoints(settings, client):
    resp = await client.post(
        "/ui/api/schedules",
        json={"type": "recurring", "goal": "daily digest", "at": "09:30", "name": "Digest"},
    )
    assert resp.status_code == 200
    sid = resp.json()["id"]
    assert (await client.get("/ui/api/schedules")).json()["schedules"][0]["name"] == "Digest"
    assert (
        await client.put(f"/ui/api/schedules/{sid}", json={"enabled": False})
    ).json()["enabled"] is False
    assert (await client.delete(f"/ui/api/schedules/{sid}")).status_code == 200
    assert (await client.delete(f"/ui/api/schedules/{sid}")).status_code == 404
    assert (
        await client.post("/ui/api/schedules", json={"type": "recurring", "goal": "x"})
    ).status_code == 400

    body = (await client.get("/ui/api/autonomy")).json()
    assert "email_send" in body["approval_keys"] and body["ntfy"]["enabled"] is False
    resp = await client.put(
        "/ui/api/autonomy/policies",
        json={"policies": {"email_send": {"mode": "auto", "daily_limit": 5}}},
    )
    assert resp.status_code == 200
    assert (await client.get("/ui/api/autonomy")).json()["policies"]["email_send"][
        "daily_limit"
    ] == 5

    assert (await client.get("/ui/api/audit")).json()["actions"] == []
    assert (await client.post("/ui/api/autonomy/notify_test")).status_code == 400
