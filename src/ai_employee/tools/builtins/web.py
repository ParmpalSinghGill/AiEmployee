"""Web tools: raw HTTP, page fetch/summarize-ready extraction, and web search."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from ai_employee.config import Settings
from ai_employee.tools.base import Tool, ToolResult


class HttpRequestTool(Tool):
    name = "http_request"
    description = "Make an HTTP request and return status, headers and (truncated) body."
    category = "web"

    class Args(BaseModel):
        url: str
        method: str = Field("GET", pattern="^(GET|POST|PUT|PATCH|DELETE|HEAD)$")
        headers: dict[str, str] = {}
        json_body: dict[str, Any] | None = None
        timeout: int = Field(30, ge=1, le=120)
        max_bytes: int = 100_000

    async def _run(self, args: HttpRequestTool.Args) -> ToolResult:
        async with httpx.AsyncClient(follow_redirects=True, timeout=args.timeout) as client:
            resp = await client.request(
                args.method, args.url, headers=args.headers, json=args.json_body
            )
        body = resp.text[: args.max_bytes]
        return ToolResult(
            ok=resp.is_success,
            output={"status": resp.status_code, "body": body},
            metadata={"content_type": resp.headers.get("content-type", "")},
            error=None if resp.is_success else f"HTTP {resp.status_code}",
        )


class FetchPageTool(Tool):
    name = "fetch_page"
    description = (
        "Fetch a web page and return clean, readable text (HTML stripped to markdown) "
        "ready for summarization or extraction by the browser/research agents."
    )
    category = "web"

    class Args(BaseModel):
        url: str
        max_chars: int = Field(20_000, ge=500, le=200_000)

    async def _run(self, args: FetchPageTool.Args) -> ToolResult:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(args.url, headers={"User-Agent": "AIEmployee/0.1"})
        resp.raise_for_status()
        text = _html_to_text(resp.text)[: args.max_chars]
        return ToolResult(output=text, metadata={"url": str(resp.url), "chars": len(text)})


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web and return a list of {title, url, snippet} results."
    category = "web"

    class Args(BaseModel):
        query: str
        max_results: int = Field(5, ge=1, le=20)

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _run(self, args: WebSearchTool.Args) -> ToolResult:
        if self._settings.tavily_api_key:
            return await self._tavily(args)
        # Offline / no-key fallback: DuckDuckGo Instant Answer (best-effort).
        return await self._duckduckgo(args)

    async def _tavily(self, args: WebSearchTool.Args) -> ToolResult:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._settings.tavily_api_key,
                    "query": args.query,
                    "max_results": args.max_results,
                },
            )
        data = resp.json()
        results = [
            {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")}
            for r in data.get("results", [])
        ]
        return ToolResult(output=results, metadata={"engine": "tavily"})

    async def _duckduckgo(self, args: WebSearchTool.Args) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": args.query, "format": "json", "no_html": 1},
                )
            data = resp.json()
            results = []
            for topic in data.get("RelatedTopics", [])[: args.max_results]:
                if "Text" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "")[:80],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", ""),
                        }
                    )
            return ToolResult(output=results, metadata={"engine": "duckduckgo"})
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                ok=False,
                error=f"web search unavailable offline ({exc}); set TAVILY_API_KEY",
                output=[],
            )


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text("\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - bs4 missing or parse error
        import re

        return re.sub(r"<[^>]+>", " ", html)
