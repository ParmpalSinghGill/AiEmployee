"""Memory, chunking and RAG tests."""

from __future__ import annotations

import pytest

from ai_employee.knowledge.chunking import chunk_text
from ai_employee.memory import MemoryNamespace
from ai_employee.memory.in_memory import InMemoryVectorStore
from ai_employee.memory.manager import MemoryManager


@pytest.fixture
def memory(brain):
    return MemoryManager(InMemoryVectorStore(), brain)


async def test_record_and_recall(memory):
    await memory.record("The capital of France is Paris.", namespace=MemoryNamespace.long_term)
    await memory.record("Python is a programming language.", namespace=MemoryNamespace.long_term)
    hits = await memory.recall("France capital", k=2)
    assert hits
    assert any("Paris" in h.content for h in hits)


async def test_conversation_window_order(memory):
    for i in range(5):
        await memory.record_message("user", f"msg {i}", session_id="s")
    window = await memory.conversation_window("s", limit=3)
    assert len(window) == 3
    assert window[-1].content == "msg 4"  # most recent last (chronological)


async def test_skill_memory(memory):
    await memory.remember_skill("deploy", "deploy the app", ["build", "push", "release"])
    hits = await memory.find_skill("how do I deploy")
    assert hits and "SKILL: deploy" in hits[0].content


def test_chunking_overlap():
    text = "para one.\n\n" + ("sentence. " * 300)
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 240 for c in chunks)


async def test_rag_answer(brain):
    from ai_employee.knowledge import KnowledgeBase

    mem = MemoryManager(InMemoryVectorStore(), brain)
    kb = KnowledgeBase(mem, brain)
    report = await kb.ingest("Redis is an in-memory data store used for caching and queues.")
    assert report.chunks >= 1
    ans = await kb.answer("What is Redis used for?")
    assert ans.used_chunks >= 1
    assert ans.citations


async def test_rag_empty_knowledge(brain):
    from ai_employee.knowledge import KnowledgeBase

    mem = MemoryManager(InMemoryVectorStore(), brain)
    kb = KnowledgeBase(mem, brain)
    ans = await kb.answer("anything?")
    assert ans.used_chunks == 0
