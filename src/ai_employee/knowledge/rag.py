"""Retrieval-Augmented Generation orchestration.

The :class:`KnowledgeBase` ingests sources into the ``knowledge`` memory namespace
(chunked + embedded) and answers questions by retrieving the most relevant chunks
and grounding the Brain's response in them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_employee.brains.base import Brain
from ai_employee.brains.schemas import Message
from ai_employee.core.logging import get_logger
from ai_employee.knowledge.chunking import chunk_text
from ai_employee.knowledge.loaders import load_source
from ai_employee.memory import MemoryManager, MemoryNamespace

log = get_logger("rag")


@dataclass
class IngestReport:
    documents: int
    chunks: int
    sources: list[str]


@dataclass
class RAGAnswer:
    answer: str
    citations: list[dict]
    used_chunks: int


class KnowledgeBase:
    def __init__(self, memory: MemoryManager, brain: Brain) -> None:
        self._memory = memory
        self._brain = brain

    async def ingest(
        self, source: str, *, chunk_size: int = 1200, overlap: int = 150
    ) -> IngestReport:
        """Load a source, chunk + embed it, and store in knowledge memory."""
        docs = load_source(source)
        total_chunks = 0
        sources: list[str] = []
        for doc in docs:
            if not doc.text.strip():
                continue
            sources.append(doc.source)
            for i, chunk in enumerate(chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)):
                await self._memory.record(
                    chunk,
                    namespace=MemoryNamespace.knowledge,
                    metadata={
                        "source": doc.source,
                        "title": doc.title,
                        "content_type": doc.content_type,
                        "chunk_index": i,
                    },
                )
                total_chunks += 1
        log.info("rag.ingested", documents=len(sources), chunks=total_chunks)
        return IngestReport(documents=len(sources), chunks=total_chunks, sources=sources)

    async def retrieve(self, query: str, *, k: int = 5):
        return await self._memory.recall(query, namespace=MemoryNamespace.knowledge, k=k)

    async def answer(self, question: str, *, k: int = 5) -> RAGAnswer:
        """Answer a question grounded strictly in retrieved knowledge."""
        hits = await self.retrieve(question, k=k)
        if not hits:
            return RAGAnswer(
                answer="I don't have any ingested knowledge relevant to that question.",
                citations=[],
                used_chunks=0,
            )
        context = "\n\n".join(
            f"[{i + 1}] (source: {h.metadata.get('source')})\n{h.content}"
            for i, h in enumerate(hits)
        )
        system = (
            "Answer the question using ONLY the numbered context below. Cite sources "
            "as [n]. If the answer is not in the context, say you don't know.\n\n"
            f"CONTEXT:\n{context}"
        )
        reply = await self._brain.chat(
            [Message.system(system), Message.user(question)], temperature=0.0
        )
        citations = [
            {"index": i + 1, "source": h.metadata.get("source"), "score": h.score}
            for i, h in enumerate(hits)
        ]
        return RAGAnswer(answer=reply.content, citations=citations, used_chunks=len(hits))
