"""CEOAgent must not backfill a skipped step's deliverable from other steps' data.

Regression coverage for a real failure observed live: a 4-step plan (research,
research, coding, documentation) had its "coding" step skipped because a
prerequisite research step failed verification twice (unmet_dependency). The CEO
synthesis step was still handed all step outputs undifferentiated, and — asked to
produce "a single, coherent, high-quality deliverable" — it hand-assembled the CSV
the skipped coding step was supposed to build, straight out of the raw research
text. Nothing ever wrote that CSV to disk, but the final answer read exactly like
a completed deliverable. The fix labels each step's status (COMPLETED / SKIPPED /
UNVERIFIED) in the digest handed to the CEO and explicitly forbids reconstructing
a skipped step's output.
"""

from __future__ import annotations

from ai_employee.agents import AgentContext
from ai_employee.agents.custom import CEOAgent
from ai_employee.tools.registry import ToolRegistry


async def test_digest_labels_skipped_step_and_forbids_reconstruction(brain) -> None:
    captured: dict = {}
    original_chat = brain.chat

    async def spy_chat(messages, **kwargs):
        captured["messages"] = messages
        return await original_chat(messages, **kwargs)

    brain.chat = spy_chat

    step_results = [
        {
            "id": 1,
            "agent": "research",
            "description": "Research OpenAI/Anthropic/Google pricing",
            "output": "OpenAI GPT-6: $10/$50 per 1M tokens...",
            "confidence": 0.8,
            "verified": True,
        },
        {
            "id": 2,
            "agent": "research",
            "description": "Research Meta/Mistral pricing",
            "output": "Mistral prices could not be confirmed.",
            "confidence": 0.3,
            "verified": False,
        },
        {
            "id": 3,
            "agent": "coding",
            "description": "Assemble findings into comparison.csv",
            "output": (
                "(skipped — prerequisite step(s) [2] produced no usable result, "
                "so there is nothing for this step to act on.)"
            ),
            "confidence": 0.0,
            "verified": False,
            "skipped": True,
        },
    ]

    ceo = CEOAgent()
    ctx = AgentContext(brain=brain, tools=ToolRegistry())
    ctx.scratch["step_results"] = step_results
    await ceo.run("Build a pricing comparison CSV", ctx)

    system_msg = captured["messages"][0].content
    user_msg = captured["messages"][1].content

    # The digest must mark each step's real status...
    assert "[COMPLETED]" in user_msg
    assert "[SKIPPED" in user_msg
    assert "[UNVERIFIED" in user_msg
    # ...and the CEO must be explicitly told not to reconstruct the skipped step's
    # deliverable itself.
    assert "SKIPPED" in system_msg
    assert "do not" in system_msg.lower() or "not synthesize" in system_msg.lower()
    assert "[3]" in system_msg or "3" in system_msg


async def test_digest_has_no_skip_hint_when_nothing_skipped(brain) -> None:
    captured: dict = {}
    original_chat = brain.chat

    async def spy_chat(messages, **kwargs):
        captured["messages"] = messages
        return await original_chat(messages, **kwargs)

    brain.chat = spy_chat

    step_results = [
        {
            "id": 1,
            "agent": "research",
            "description": "Research pricing",
            "output": "All prices confirmed.",
            "confidence": 0.9,
            "verified": True,
        },
    ]
    ceo = CEOAgent()
    ctx = AgentContext(brain=brain, tools=ToolRegistry())
    ctx.scratch["step_results"] = step_results
    await ceo.run("Research pricing", ctx)

    system_msg = captured["messages"][0].content
    assert "SKIPPED" not in system_msg


async def test_no_step_results_skips_digest_entirely(brain) -> None:
    ceo = CEOAgent()
    ctx = AgentContext(brain=brain, tools=ToolRegistry())
    result = await ceo.run("Just restate the goal", ctx)
    assert result.success is True
    assert result.output
