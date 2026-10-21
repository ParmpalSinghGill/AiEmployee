"""Learned-skill management: memory API + UI endpoints (list/create/edit/delete)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from ai_employee.memory import MemoryNamespace
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.memory.manager import MemoryManager


@pytest.fixture
def memory(brain):
    return MemoryManager(InMemoryVectorStore(), brain)


class _FakeEmployee:
    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.persisted = 0

    def persist_memory(self) -> None:
        self.persisted += 1


@pytest.fixture
async def skills_client(memory):
    from ai_employee.api.app import create_app
    from ai_employee.api.deps import get_employee

    emp = _FakeEmployee(memory)
    app = create_app()
    app.dependency_overrides[get_employee] = lambda: emp
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, emp, memory


# ── memory manager ──────────────────────────────────────────────────────────


async def test_save_list_update_delete_skill(memory):
    record = await memory.save_skill(
        name="deploy django",
        description="Ship a Django app safely.",
        steps=["Run tests", "Collect static", "Restart workers"],
        tags=["django", "deploy"],
    )
    skills = await memory.list_skills()
    assert [s.id for s in skills] == [record.id]
    assert skills[0].metadata["name"] == "deploy django"
    assert skills[0].metadata["steps"] == ["Run tests", "Collect static", "Restart workers"]

    updated = await memory.save_skill(
        skill_id=record.id,
        name="deploy django (fixed)",
        description="Corrected workflow.",
        steps=["Run targeted tests"],
        tags=["django"],
    )
    assert updated.id == record.id
    assert updated.created_at == record.created_at
    assert "SKILL: deploy django (fixed)" in updated.content

    # Editing replaces the old record rather than creating a second one.
    skills = await memory.list_skills()
    assert len(skills) == 1
    assert skills[0].id == record.id
    assert skills[0].content == updated.content

    assert await memory.delete_skill(record.id) is True
    assert await memory.list_skills() == []
    assert await memory.delete_skill(record.id) is False


async def test_update_unknown_skill_raises(memory):
    with pytest.raises(KeyError):
        await memory.save_skill(
            skill_id="does-not-exist",
            name="x",
            description="",
            steps=[],
            tags=[],
        )


async def test_remember_skill_rejects_task_specific(memory):
    from ai_employee.memory.skill_policy import SkillRejectedError

    with pytest.raises(SkillRejectedError):
        await memory.remember_skill(
            "# Task: django__django-11555",
            "Reusable approach for: fix the bug",
            ["checkout 8dd5877f58f84f2b11126afbd0813e24545919ed"],
        )
    assert await memory.list_skills() == []


async def test_remember_skill_merges_duplicate_general_skills(memory):
    first = await memory.remember_skill(
        "Run Django tests", "how to run tests", ["python manage.py test"]
    )
    second = await memory.remember_skill("Run Django tests", "updated how-to", ["pytest"])
    assert second.id == first.id
    skills = await memory.list_skills()
    assert len(skills) == 1
    assert skills[0].metadata["steps"] == ["pytest"]


async def test_find_skill_skips_task_specific_legacy_records(memory):
    # Simulate a legacy on-disk skill that predates the guard.
    await memory.record(
        "SKILL: # Task: django__django-11555\nReusable approach for: fix\nSTEPS:\n- checkout abc",
        namespace=MemoryNamespace.skill,
        metadata={},
        embed=False,
    )
    await memory.remember_skill("Run Django tests", "how to run tests", ["python manage.py test"])

    hits = await memory.find_skill("django tests", k=5)
    assert hits
    assert all("django-11555" not in h.content for h in hits)


async def test_purge_task_specific_skills(memory):
    await memory.record(
        "SKILL: # Task: scikit-learn__scikit-learn-14053\nReusable approach for: fix\nSTEPS:\n- a",
        namespace=MemoryNamespace.skill,
        metadata={},
        embed=False,
    )
    await memory.remember_skill("Run Django tests", "how to run tests", ["python manage.py test"])

    removed = await memory.purge_task_specific_skills()
    assert removed == 1
    skills = await memory.list_skills()
    assert len(skills) == 1
    assert skills[0].metadata["name"] == "Run Django tests"


async def test_import_records_drops_legacy_task_specific_skills(memory):
    memory.import_records(
        [
            {
                "id": "badskill",
                "namespace": "skill",
                "content": "SKILL: # Task: django__django-11555\nReusable approach for: fix",
                "metadata": {"name": "# Task: django__django-11555"},
            },
            {
                "id": "goodskill",
                "namespace": "skill",
                "content": "SKILL: Run Django tests\nhow to run tests\nSTEPS:\n- python manage.py test",
                "metadata": {"name": "Run Django tests", "steps": ["python manage.py test"]},
            },
            {
                "id": "note",
                "namespace": "long_term",
                "content": "a useful note",
                "metadata": {},
            },
        ]
    )

    exported = memory.export_records()
    assert {r["id"] for r in exported} == {"goodskill", "note"}
    assert [r.id for r in await memory.list_skills()] == ["goodskill"]


# ── UI API ──────────────────────────────────────────────────────────────────


async def test_skills_api_create_edit_delete(skills_client):
    client, emp, _memory = skills_client

    empty = (await client.get("/ui/api/skills")).json()
    assert empty == {"skills": [], "count": 0}

    created = await client.post(
        "/ui/api/skills",
        json={
            "name": "Fix a flaky test",
            "description": "Diagnose and stabilise a flaky test.",
            "steps": ["Reproduce it", "Find the race", "Add a regression test"],
            "tags": ["testing"],
        },
    )
    assert created.status_code == 200
    body = created.json()
    skill = body["skill"]
    assert body["created"] is True
    assert skill["name"] == "Fix a flaky test"
    assert skill["steps"] == ["Reproduce it", "Find the race", "Add a regression test"]
    assert skill["tags"] == ["testing"]
    assert emp.persisted == 1

    listed = (await client.get("/ui/api/skills")).json()
    assert listed["count"] == 1
    assert listed["skills"][0]["id"] == skill["id"]

    edited = await client.put(
        f"/ui/api/skills/{skill['id']}",
        json={
            "name": "Fix a flaky test (safe)",
            "description": "Corrected diagnosis workflow.",
            "steps": ["Reproduce with --count=50", "Fix the race"],
            "tags": ["testing", "pytest"],
        },
    )
    assert edited.status_code == 200
    revised = edited.json()["skill"]
    assert revised["id"] == skill["id"]
    assert revised["name"] == "Fix a flaky test (safe)"
    assert revised["tags"] == ["testing", "pytest"]

    deleted = await client.delete(f"/ui/api/skills/{skill['id']}")
    assert deleted.status_code == 200
    assert (await client.get("/ui/api/skills")).json()["count"] == 0
    assert (await client.delete(f"/ui/api/skills/{skill['id']}")).status_code == 404
    assert emp.persisted == 3


async def test_skills_api_rejects_blank_name(skills_client):
    client, _emp, _memory = skills_client
    res = await client.post("/ui/api/skills", json={"name": "   ", "steps": []})
    assert res.status_code == 422


async def test_skills_api_rejects_task_specific_skill(skills_client):
    client, emp, _memory = skills_client
    res = await client.post(
        "/ui/api/skills",
        json={"name": "# Task: django__django-11555", "description": "fix", "steps": []},
    )
    assert res.status_code == 422
    assert emp.persisted == 0


async def test_skills_api_404_for_unknown_edit(skills_client):
    client, _emp, _memory = skills_client
    res = await client.put(
        "/ui/api/skills/nope",
        json={"name": "x", "description": "", "steps": [], "tags": []},
    )
    assert res.status_code == 404


async def test_legacy_skill_content_is_parsed_for_editing(skills_client):
    client, _emp, memory = skills_client
    await memory.record(
        "SKILL: legacy skill\nOld description.\nSTEPS:\n- first step\n- second step",
        namespace=MemoryNamespace.skill,
        metadata={},
        embed=False,
    )

    body = (await client.get("/ui/api/skills")).json()
    assert body["count"] == 1
    skill = body["skills"][0]
    assert skill["name"] == "legacy skill"
    assert skill["description"] == "Old description."
    assert skill["steps"] == ["first step", "second step"]


def test_ui_exposes_skills_badge_and_modal() -> None:
    static = Path(__file__).resolve().parents[1] / "src/ai_employee/api/static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    assert 'id="skills-badge"' in html
    assert "openSkillsModal" in js
    assert "/ui/api/skills" in js
    assert "mp-skills" in js
