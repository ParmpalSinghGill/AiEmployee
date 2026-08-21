# YouTube description — AI Employee Framework demo

Copy the block below into the YouTube description box. Fill the `[…]` placeholders
and delete any timestamp lines for scenes you didn't film.

**Verified against the code on 2026-08-10:** 29 agents, 24 tools, Apache-2.0,
repo `github.com/ParmpalSinghGill/AiEmployee`. Re-check with
`python -m ai_employee agents` / `tools` before publishing if the code has moved on.

---

## Title options

1. I built an AI Employee: give it one goal, it plans and delegates to 29 agents
2. This AI doesn't chat — it plans, delegates, and does the work (open source)
3. AI Employee Framework — multi-agent, any LLM, runs offline (Apache-2.0)

Shorter, punchier alternative: **Not a chatbot. An AI employee.**

---

## Description

Most AI tools give you a chat box. You ask, it answers, and you still do the work.

This is different. The AI Employee Framework takes a single high-level goal, plans the
work, delegates each step to a specialised agent, uses real tools, verifies its own
output, and remembers what it did — across restarts.

In this video I give it a real job: find a pretrained model and a dataset for detecting
defects on a factory production line, work out the GPU needed to fine-tune it, and write
runnable training code with a README. Then I watch it work.

WHAT YOU'LL SEE
• A one-sentence goal decomposed into a plan, with each step assigned to a different specialist
• Real tool calls — Hugging Face, arXiv, Kaggle, Roboflow, GitHub — not invented results
• Multiple agents running in parallel, with dependent steps waiting and starting the moment they're unblocked
• Self-verification with a confidence score on every step
• An "Explain" tab showing why each decision was made — what the model was asked and what it chose
• Python written and executed in a sandbox, producing a real chart rendered in the dashboard
• The server killed mid-session and restarted — the task resumes with full context

WHAT IT IS
An open-source, provider-independent multi-agent framework built on LangChain, LangGraph,
FastAPI and Postgres/pgvector.

• 29 specialised agents (planner, research, scout, coding, security, QA, DevOps, …)
• 24 sandboxed tools, with a human-approval gate on anything sensitive
• Swap the LLM with one environment variable — DeepSeek, OpenAI, Gemini, GLM, or a local
  model via Ollama. No business logic ever imports an LLM SDK.
• Runs fully offline with a mock brain and in-memory store — no API keys, no database
• RAG over your own documents (PDF, Markdown, Word, PowerPoint, CSV, Excel, web pages, codebases)
• Web console, REST API and CLI

LINKS
⭐ Source (Apache-2.0): https://github.com/ParmpalSinghGill/AiEmployee
📄 Setup guide: https://github.com/ParmpalSinghGill/AiEmployee/blob/main/docs/setup.md
🏗 Architecture: https://github.com/ParmpalSinghGill/AiEmployee/blob/main/docs/architecture.md

TRY IT IN TWO MINUTES (no API key needed)
    git clone https://github.com/ParmpalSinghGill/AiEmployee
    cd AiEmployee && conda env create -f environment.yml && conda activate AIEmployee
    pip install -e .
    LLM_PROVIDER=mock python -m ai_employee run "Plan a product launch"

Then for the web console:
    python -m ai_employee serve      # http://localhost:8010/ui

Questions or ideas? Leave a comment — I read them all.

#AI #OpenSource #LangChain #LangGraph #MultiAgent #Python #LLM #DeepSeek #Ollama #AIAgents

---

## Chapters (adjust timestamps to your final edit)

    0:00  The goal
    0:20  It plans before it answers
    1:00  Real tools, real searches
    2:00  The result you can actually use
    2:50  Why did it do that? (Explain)
    3:25  It writes and runs code
    4:15  Memory that survives a restart
    5:05  Multiple agents at once
    5:50  Swap the model with one variable
    6:25  Runs offline, no API keys
    6:55  CLI and API

---

## Pinned comment

Built this because I wanted an agent that finishes a job, not one that gives me a
paragraph and leaves the work to me.

It's Apache-2.0 and runs offline out of the box — `LLM_PROVIDER=mock` needs no API key,
no database, nothing. If you try it, I'd genuinely like to hear where it breaks:
https://github.com/ParmpalSinghGill/AiEmployee

---

## Short version (for Shorts / X / LinkedIn)

Not a chatbot — an AI employee.

One sentence in: "find a defect-detection model and dataset, tell me the GPU I need, and
write the training code."

It plans the work, splits it across specialised agents running in parallel, searches
Hugging Face and Kaggle for real, sizes the GPU, writes runnable code — then verifies
itself and remembers it all after a restart.

Open source, Apache-2.0. Works with DeepSeek, OpenAI, Gemini, or a local model — one
environment variable.

github.com/ParmpalSinghGill/AiEmployee
