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
from rich.table import Table

console = Console()


def _render_usage(result: dict) -> None:
    """Print the 'agents used' + 'tools used' sections for a run/agent result."""
    agents = result.get("agents_used") or []
    tools = result.get("tools_used") or []

    if agents:
        table = Table(
            title=f"Agents used ({len(agents)})",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Agent")
        table.add_column("Steps", justify="right")
        table.add_column("Tool calls", justify="right")
        table.add_column("OK", justify="right")
        table.add_column("Failed", justify="right")
        for a in agents:
            table.add_row(
                a.get("agent", "?"),
                str(a.get("steps", 0)),
                str(len(a.get("tools", []))),
                str(a.get("ok", 0)),
                str(a.get("failed", 0)),
            )
        console.print(table)

    if tools:
        table = Table(
            title=f"Tools used ({len(tools)})",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Tool")
        table.add_column("Calls", justify="right")
        table.add_column("OK", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Agents")
        for t in tools:
            table.add_row(
                t.get("tool", "?"),
                str(t.get("calls", 0)),
                str(t.get("ok", 0)),
                str(t.get("failed", 0)),
                ", ".join(t.get("agents", [])),
            )
        console.print(table)


def _task_identity(value: str | None) -> str | None:
    if not value:
        return None
    from ai_employee.resources.names import safe_task_identity

    return safe_task_identity(value)


async def _run_resources(emp, args) -> int:
    from ai_employee.resources import manager_for, migrate_workspace

    mgr = manager_for(emp.settings)
    cmd = args.resources_command

    if cmd == "list":
        records = mgr.list(
            task=_task_identity(args.task),
            kind=args.kind.strip() or None,
            query=args.query,
            shared_only=args.shared,
        )
        if not records:
            console.print("[dim]No resources found.[/]")
            return 0
        table = Table(title=f"Resources ({len(records)})", header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Id")
        table.add_column("Kind")
        table.add_column("Visibility")
        table.add_column("Owner")
        table.add_column("Size", justify="right")
        table.add_column("Ver", justify="right")
        table.add_column("Path")
        for r in records:
            table.add_row(
                r.name,
                r.id,
                r.kind,
                r.visibility,
                r.owner_task or "-",
                str(r.size),
                str(r.version),
                r.path,
            )
        console.print(table)
        return 0

    if cmd == "show":
        record = mgr.resolve(args.ref, task=_task_identity(args.task), mode="read")
        console.print_json(json.dumps(record.to_dict(), default=str))
        return 0

    if cmd == "verify":
        report = mgr.verify(task=_task_identity(args.task))
        color = "green" if report["ok"] else "yellow"
        console.print(
            Panel(
                f"[{color}]ok:[/] {report['ok']}\n"
                f"checked: {report['checked']}  present: {report['present']}  "
                f"missing: {len(report['missing'])}  mismatched: {len(report['mismatched'])}",
                title="Resource verification",
            )
        )
        for item in report["missing"]:
            console.print(f"[red]missing[/] {item['name']} ({item['path']})")
        for item in report["mismatched"]:
            console.print(f"[red]changed[/] {item['name']} expected {item['expected']}")
        return 0 if report["ok"] else 1

    if cmd == "migrate":
        report = migrate_workspace(mgr, apply=args.apply)
        mode = "applied" if report["applied"] else "dry-run"
        console.print(
            Panel(
                f"mode: {mode}\n"
                f"candidates: {report['candidates']}\n"
                f"registered: {report['registered']}\n"
                f"skipped (already cataloged): {report['skipped']}",
                title="Legacy workspace migration",
                style="cyan" if report["applied"] else "yellow",
            )
        )
        for entry in report["entries"][:40]:
            console.print(
                f"[dim]{entry['action']:8}[/] {entry['path']} "
                f"({entry['kind']}, {entry['visibility']})"
            )
        if not report["applied"]:
            console.print("[dim]Re-run with --apply to register these resources.[/]")
        return 0

    if cmd == "promote":
        record = mgr.promote(args.resource_id, task=_task_identity(args.task))
        console.print(
            Panel(
                f"[green]Promoted[/] {record.name} to [bold]shared[/].\nPath: {record.path}",
                title="Resource promoted",
            )
        )
        return 0

    if cmd == "links":
        rows = []
        for rec in sorted(mgr.records().values(), key=lambda r: r.name):
            tasks = mgr.tasks_for(rec.id)
            rows.append((rec, tasks))
        if not rows:
            console.print("[dim]No resources found.[/]")
            return 0
        table = Table(title="Task <-> source links", header_style="bold cyan")
        table.add_column("Source")
        table.add_column("Visibility")
        table.add_column("#tasks", justify="right")
        table.add_column("Linked tasks")
        for rec, tasks in rows:
            names = ", ".join(t["task_key"] or t["task_id"] for t in tasks) or "-"
            table.add_row(rec.name, rec.visibility, str(len(tasks)), names)
        console.print(table)
        return 0

    if cmd == "isolate":
        from ai_employee.resources.isolation import isolate_all, isolate_task

        allow = not args.no_redownload
        if args.all:
            report = await isolate_all(
                mgr, settings=emp.settings, allow_redownload=allow, purge_orphans=True
            )
        else:
            report = await isolate_task(
                mgr,
                args.task or "",
                settings=emp.settings,
                allow_redownload=allow,
                purge_orphans=True,
            )
        console.print(
            Panel(
                f"tasks:        {len(report.get('tasks', [])) if args.all else 1}\n"
                f"isolated:     {len(report.get('isolated', []))}\n"
                f"re-downloaded:{len(report.get('needs_redownload', []))} pending\n"
                f"orphans purged: {len(report.get('purged', []))}\n"
                f"unrecoverable: {len(report.get('unrecoverable', []))}",
                title="Isolation report",
                style="cyan",
            )
        )
        return 0

    if cmd == "purge":
        report = mgr.purge_all(purge_bytes=not args.keep_bytes)
        lines = [
            f"deleted:      {report['deleted']} resources",
            f"bytes freed:  {report['bytes_freed']}",
            f"files removed:{'yes' if report['purge_bytes'] else 'no (--keep-bytes)'}",
        ]
        if getattr(args, "workspace", False):
            ws_report = mgr.wipe_workspace()
            lines.append(
                f"workspace:    wiped {ws_report['removed_count']} entries "
                f"({ws_report['bytes_freed']} bytes)"
            )
        console.print(Panel("\n".join(lines), title="Resources purged — fresh start", style="red"))
        return 0

    return 1


def _run_recover_log(emp, args) -> int:
    """Import runs still present in the server log into the saved-session store."""
    from pathlib import Path

    from ai_employee.observability.recover import existing_task_ids, import_all_new, recover_runs
    from ai_employee.observability.store import SessionStore

    log_path = Path(args.log) if args.log else Path(emp.settings.sessions_dir).parent / "server.log"
    if not log_path.exists():
        console.print(f"[yellow]No server log at {log_path}[/]")
        return 1
    store = SessionStore(emp.settings.sessions_dir)
    runs = recover_runs(log_path)
    if not args.apply:
        have = existing_task_ids(store)
        new = [r for r in runs if r.task_id not in have]
        console.print(
            Panel(
                f"log:        {log_path}\n"
                f"runs found: {len(runs)}\n"
                f"not saved:  {len(new)}\n\n"
                "Re-run with --apply to import them into Saved tasks.",
                title="Recover runs from log (dry run)",
            )
        )
        for r in runs[:20]:
            console.print(f"  {r.created_at[:19]}  {r.task_id}  {r.status:<10} {r.goal[:50]}")
        return 0
    imported = import_all_new(store, log_path, owner_id=None, owner_name="cli")
    console.print(f"[green]Imported {len(imported)} run(s) from {log_path}[/]")
    for meta, _snap in imported:
        console.print(f"  {meta['run_ids'][0]}  {meta['title']}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-employee", description="AI Employee Framework CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Run the API server")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=8010)
    sp.add_argument("--reload", action="store_true")

    rp = sub.add_parser("run", help="Plan and execute a high-level goal")
    rp.add_argument("goal")
    rp.add_argument("--session", default="cli")
    rp.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured result (answer, plan, agents_used, tools_used) as JSON",
    )

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
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print the full agent result (output, agents_used, tools_used) as JSON",
    )

    sub.add_parser("agents", help="List available agents")
    sub.add_parser("tools", help="List available tools")
    sub.add_parser("config", help="Show the active brain (provider + model) and settings")

    rsp = sub.add_parser("resources", help="Manage task/shared resources (private, shared, dedup)")
    rsub = rsp.add_subparsers(dest="resources_command", required=True)
    rl = rsub.add_parser("list", help="List resources visible to a task (+ all shared)")
    rl.add_argument("--task", default=None, help="Task identity (session id). Omit for all.")
    rl.add_argument("--shared", action="store_true", help="Only shared, reusable resources.")
    rl.add_argument("--kind", default="", help="Filter: model | dataset | repo | media | file")
    rl.add_argument("--query", default="", help="Free-text filter over name/kind/url/hash.")
    rs = rsub.add_parser("show", help="Show one resource by id, name or hash")
    rs.add_argument("ref")
    rs.add_argument("--task", default=None)
    rv = rsub.add_parser("verify", help="Check every resource's bytes are present/unchanged")
    rv.add_argument("--task", default=None)
    rm = rsub.add_parser("migrate", help="Index legacy workspace artifacts into the catalog")
    rm.add_argument("--apply", action="store_true", help="Actually register (default: dry run).")
    rp = rsub.add_parser("promote", help="Promote a private resource to shared")
    rp.add_argument("resource_id")
    rp.add_argument("--task", required=True, help="Owning task identity.")
    rsub.add_parser("links", help="Show task <-> source links (which tasks share a source)")
    ri = rsub.add_parser(
        "isolate", help="Isolate tasks so no source is shared (copy, re-download if needed)"
    )
    ri.add_argument("--all", action="store_true", help="Isolate every task")
    ri.add_argument("--task", default=None, help="Isolate one task identity/session")
    ri.add_argument(
        "--no-redownload",
        action="store_true",
        help="Never re-fetch; just report which shared sources need it",
    )
    rp = rsub.add_parser("purge", help="Delete ALL resources (catalog + files) for a fresh start")
    rp.add_argument(
        "--keep-bytes",
        action="store_true",
        help="Remove catalog entries but leave the files on disk",
    )
    rp.add_argument(
        "--workspace",
        action="store_true",
        help="Also delete every other file in the workspace (models/, data/, sessions/, …)",
    )

    rec = sub.add_parser("recover-log", help="Recover runs from the server log into Saved tasks")
    rec.add_argument("--log", default=None, help="Path to server.log (default: <data>/server.log).")
    rec.add_argument("--apply", action="store_true", help="Actually import (default: show what would).")
    return p


async def _run_async(args) -> int:
    from ai_employee import AIEmployee

    emp = AIEmployee()
    try:
        if args.command == "run":
            result = await emp.run(args.goal, session_id=args.session)
            if getattr(args, "json", False):
                console.print_json(json.dumps(result, default=str))
                return 0
            pending = result.get("needs_approval")
            if pending:
                op = pending.get("operation", "an operation")
                batched = pending.get("batched_calls") or []
                detail = f" ({len(batched)} pending call(s))" if len(batched) > 1 else ""
                console.print(
                    Panel(
                        f"This run is paused waiting for human approval of '{op}'{detail}.\n"
                        "The CLI has no interactive approval step — approve it via the API "
                        "(POST the session's approved_ops) or the web console, then resume "
                        "this session, or re-run with a coding/tool backend that doesn't "
                        "require approval for this operation.",
                        title="Paused — needs approval",
                        style="yellow",
                    )
                )
            else:
                console.print(
                    Panel(result["answer"], title=f"Result (confidence={result['confidence']})")
                )
            _render_usage(result)
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
            if getattr(args, "json", False):
                console.print_json(json.dumps(out, default=str))
            else:
                console.print(
                    Panel(out["output"], title=f"{args.agent} (confidence={out['confidence']})")
                )
                _render_usage(out)
        elif args.command == "agents":
            for a in emp.agents.catalog():
                console.print(f"[cyan]{a['name']:18}[/] {a['role']} — {a['description']}")
        elif args.command == "tools":
            for t in emp.tools.all():
                console.print(f"[magenta]{t.name:16}[/] [{t.category}] {t.description}")
        elif args.command == "resources":
            return await _run_resources(emp, args)
        elif args.command == "recover-log":
            return _run_recover_log(emp, args)
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
