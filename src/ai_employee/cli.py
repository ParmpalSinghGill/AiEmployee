"""Command-line interface for the AI Employee.

Usage:
    ai-employee serve                 # run the FastAPI server
    ai-employee run "<goal>"          # plan + execute a goal
    ai-employee chat "<message>"      # single-turn chat
    ai-employee ingest <source>       # add a document/dir/URL to the knowledge base
    ai-employee ask <agent> "<task>"  # run one agent directly
    ai-employee agents                # list available agents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-employee", description="AI Employee Framework CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Run the API server")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--reload", action="store_true")

    rp = sub.add_parser("run", help="Plan and execute a high-level goal")
    rp.add_argument("goal")
    rp.add_argument("--session", default="cli")

    cp = sub.add_parser("chat", help="Single-turn chat")
    cp.add_argument("message")
    cp.add_argument("--session", default="cli")

    ip = sub.add_parser("ingest", help="Ingest a source into the knowledge base")
    ip.add_argument("source")

    qp = sub.add_parser("query", help="Ask a question over ingested knowledge")
    qp.add_argument("question")

    ap = sub.add_parser("ask", help="Run a single agent")
    ap.add_argument("agent")
    ap.add_argument("task")

    sub.add_parser("agents", help="List available agents")
    sub.add_parser("tools", help="List available tools")
    sub.add_parser("config", help="Show the active brain (provider + model) and settings")
    return p


async def _run_async(args) -> int:
    from ai_employee import AIEmployee

    emp = AIEmployee()
    try:
        if args.command == "run":
            result = await emp.run(args.goal, session_id=args.session)
            console.print(
                Panel(result["answer"], title=f"Result (confidence={result['confidence']})")
            )
        elif args.command == "chat":
            reply = await emp.chat(args.message, session_id=args.session)
            console.print(Panel(reply, title="Reply"))
        elif args.command == "ingest":
            report = await emp.knowledge.ingest(args.source)
            console.print(f"[green]Ingested[/] {report.documents} docs, {report.chunks} chunks")
        elif args.command == "query":
            res = await emp.knowledge.answer(args.question)
            console.print(Panel(res.answer, title=f"Answer ({res.used_chunks} chunks)"))
            console.print(json.dumps(res.citations, indent=2))
        elif args.command == "ask":
            out = await emp.ask_agent(args.agent, args.task)
            console.print(
                Panel(out["output"], title=f"{args.agent} (confidence={out['confidence']})")
            )
        elif args.command == "agents":
            for a in emp.agents.catalog():
                console.print(f"[cyan]{a['name']:18}[/] {a['role']} — {a['description']}")
        elif args.command == "tools":
            for t in emp.tools.all():
                console.print(f"[magenta]{t.name:16}[/] [{t.category}] {t.description}")
        elif args.command == "config":
            s = emp.settings
            warn = emp.brain.name == "mock"
            color = "yellow" if warn else "green"
            console.print(
                Panel(
                    f"[bold]Brain (LLM):[/]    [{color}]{emp.brain.name}[/]  ·  model: {emp.brain.model_name}\n"
                    f"[bold]Reasoning:[/]      {emp.brain.supports_reasoning}\n"
                    f"[bold]Embeddings:[/]     {s.embedding_provider.value}  ·  dim {s.embedding_dim}\n"
                    f"[bold]Memory backend:[/] {s.memory_backend}\n"
                    f"[bold]Sessions dir:[/]   {s.sessions_dir}\n"
                    + (
                        "\n[yellow]⚠ Running the OFFLINE MOCK — answers are placeholders.\n"
                        "  Set LLM_PROVIDER in .env (e.g. local, deepseek) for real output.[/]"
                        if warn
                        else ""
                    ),
                    title="Active configuration",
                )
            )
            console.print("[dim]Change the brain: edit LLM_PROVIDER in .env, then restart.[/]")
        return 0
    finally:
        await emp.aclose()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from ai_employee.config import get_settings

        s = get_settings()
        uvicorn.run(
            "ai_employee.api.app:app",
            host=args.host or s.api_host,
            port=args.port or s.api_port,
            reload=args.reload,
        )
        return 0

    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
