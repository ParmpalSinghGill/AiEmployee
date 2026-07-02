"""Quickstart example — runs fully offline with the mock provider.

    LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock MEMORY_BACKEND=memory \\
        python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from ai_employee import AIEmployee


async def main() -> None:
    employee = AIEmployee()

    # 1) Plan + execute a high-level goal through the multi-agent pipeline.
    result = await employee.run(
        "Research the benefits of structured logging and write a short brief",
        session_id="demo",
    )
    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print(f"\nconfidence={result['confidence']}  steps={len(result['steps'])}")

    # 2) Ground an answer in ingested knowledge (RAG).
    await employee.knowledge.ingest(
        "Structured logging emits machine-parseable key/value events, "
        "making logs queryable and metrics derivable."
    )
    rag = await employee.knowledge.answer("Why is structured logging useful?")
    print("\n=== RAG ANSWER ===")
    print(rag.answer)

    # 3) Run a single specialized agent directly.
    doc = await employee.ask_agent("documentation", "Document what an AI Employee is")
    print("\n=== DOCUMENTATION AGENT ===")
    print(doc["output"][:300])

    await employee.aclose()


if __name__ == "__main__":
    asyncio.run(main())
