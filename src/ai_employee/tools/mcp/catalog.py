"""Catalog of popular MCP (Model Context Protocol) connectors.

Each :class:`ConnectorSpec` describes one well-known MCP server — how to launch
it (stdio command) or reach it (HTTP URL), which parameters/secrets it needs,
and which runtime (``npx`` / ``uvx``) it depends on. The UI renders this catalog
so a connector can be enabled with a couple of fields instead of hand-editing
JSON. Two ``custom_*`` entries cover any server not listed here.

Templates use ``{param}`` placeholders resolved from the per-server params saved
in the MCP store (see :mod:`ai_employee.tools.mcp.store`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamSpec:
    """One user-supplied parameter of a connector (path, token, URL…)."""

    name: str
    label: str
    description: str = ""
    secret: bool = False
    required: bool = True
    default: str = ""


@dataclass(frozen=True)
class ConnectorSpec:
    """A ready-to-enable MCP server definition."""

    id: str
    label: str
    description: str
    transport: str  # "stdio" | "http"
    command: str = ""  # stdio: full command line template ("npx -y pkg {param}")
    url: str = ""  # http: endpoint template
    env: dict[str, str] = field(default_factory=dict)  # ENV_VAR -> value template
    headers: dict[str, str] = field(default_factory=dict)  # http: header templates
    params: tuple[ParamSpec, ...] = ()
    runtime: str = ""  # binary the command needs on PATH ("npx" | "uvx" | "")
    docs: str = ""
    # "oauth": no keys/params — Connect opens the provider's consent page in the
    # browser (MCP OAuth 2.1: discovery + dynamic client registration + PKCE).
    # "google": no server at all — the card is backed by the native Google tools
    # and the account connected with "Connect Google account" (see google_scopes).
    auth: str = ""
    # Worker agents that get this connector's tools. Empty = every worker.
    agents: tuple[str, ...] = ()
    # auth="google" only: OAuth scopes the account must have granted, and the
    # native tools (tools/builtins/google_workspace.py) the card stands for.
    google_scopes: tuple[str, ...] = ()
    native_tools: tuple[str, ...] = ()


CONNECTOR_CATALOG: tuple[ConnectorSpec, ...] = (
    ConnectorSpec(
        "filesystem",
        "Filesystem",
        "Read/write files under a root directory (official reference server).",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-filesystem {root}",
        params=(
            ParamSpec(
                "root",
                "Root directory",
                "Absolute or repo-relative directory the server may access.",
                default="./data/workspace",
            ),
        ),
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
        agents=("coding", "testing", "documentation", "data_analysis"),
    ),
    ConnectorSpec(
        "github",
        "GitHub",
        "Repos, issues, PRs, and code search via the GitHub API.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-github",
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "{token}"},
        params=(ParamSpec("token", "Personal access token", secret=True),),
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/github",
        agents=("coding", "git", "reviewer", "research"),
    ),
    ConnectorSpec(
        "slack",
        "Slack",
        "Read channels and post messages in a Slack workspace.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-slack",
        env={"SLACK_BOT_TOKEN": "{bot_token}", "SLACK_TEAM_ID": "{team_id}"},
        params=(
            ParamSpec("bot_token", "Bot token (xoxb-…)", secret=True),
            ParamSpec("team_id", "Team ID (T…)"),
        ),
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/slack",
        agents=("comms",),
    ),
    ConnectorSpec(
        "postgres",
        "PostgreSQL",
        "Read-only SQL queries and schema inspection against a Postgres database.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-postgres {database_url}",
        params=(
            ParamSpec(
                "database_url",
                "Connection URL",
                "postgresql://user:pass@host:5432/db",
                secret=True,
            ),
        ),
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/postgres",
        agents=("data_analysis", "coding"),
    ),
    ConnectorSpec(
        "git",
        "Git",
        "Read, search, and manipulate local git repositories (status, diff, log, commit).",
        transport="stdio",
        command="uvx mcp-server-git --repository {repo_path}",
        params=(
            ParamSpec(
                "repo_path",
                "Repository path",
                "Path to the local git repository to operate on.",
                default=".",
            ),
        ),
        runtime="uvx",
        docs="https://github.com/modelcontextprotocol/servers/tree/main/src/git",
        agents=("git", "coding", "reviewer"),
    ),
    ConnectorSpec(
        "gmail",
        "Gmail",
        "Search, read, and send mail from your connected Google account. No key "
        "files or extra setup — Connect uses Google sign-in. Sending stays "
        "approval-gated.",
        transport="google",
        auth="google",
        agents=("comms",),
        google_scopes=(
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
        ),
        native_tools=("gmail_search", "gmail_read", "gmail_send"),
        docs="https://developers.google.com/gmail/api",
    ),
    ConnectorSpec(
        "google_calendar",
        "Google Calendar",
        "List and create events on your connected Google account's calendar. "
        "Connect uses Google sign-in; creating events stays approval-gated.",
        transport="google",
        auth="google",
        agents=("comms",),
        google_scopes=("https://www.googleapis.com/auth/calendar",),
        native_tools=("calendar_list_events", "calendar_create_event"),
        docs="https://developers.google.com/calendar/api",
    ),
    ConnectorSpec(
        "gdrive",
        "Google Drive",
        "Search and read files in your connected Google account's Drive (Docs, "
        "Sheets and Slides exported as text/CSV). Read-only; Connect uses Google "
        "sign-in.",
        transport="google",
        auth="google",
        agents=("comms", "research"),
        google_scopes=("https://www.googleapis.com/auth/drive.readonly",),
        native_tools=("drive_search", "drive_read"),
        docs="https://developers.google.com/drive/api",
    ),
    ConnectorSpec(
        "google_maps",
        "Google Maps",
        "Geocoding, place search & details, distance matrix, directions.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-google-maps",
        env={"GOOGLE_MAPS_API_KEY": "{api_key}"},
        params=(ParamSpec("api_key", "Maps API key", secret=True),),
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps",
        agents=("research",),
    ),
    ConnectorSpec(
        "sqlite",
        "SQLite",
        "Query and update a local SQLite database file.",
        transport="stdio",
        command="uvx mcp-server-sqlite --db-path {db_path}",
        params=(ParamSpec("db_path", "Database file", "Path to the .db/.sqlite file."),),
        runtime="uvx",
        docs="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/sqlite",
        agents=("data_analysis", "coding"),
    ),
    ConnectorSpec(
        "brave_search",
        "Brave Search",
        "Web and local search through the Brave Search API.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-brave-search",
        env={"BRAVE_API_KEY": "{api_key}"},
        params=(ParamSpec("api_key", "Brave API key", secret=True),),
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/brave-search",
        agents=("research",),
    ),
    ConnectorSpec(
        "fetch",
        "Fetch",
        "Fetch a URL and convert the page to LLM-friendly markdown.",
        transport="stdio",
        command="uvx mcp-server-fetch",
        runtime="uvx",
        docs="https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
        agents=("research",),
    ),
    ConnectorSpec(
        "memory",
        "Memory (knowledge graph)",
        "Persistent knowledge-graph memory: entities, relations, observations.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-memory",
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers/tree/main/src/memory",
        agents=("research", "documentation"),
    ),
    ConnectorSpec(
        "playwright",
        "Playwright browser",
        "Drive a real browser: navigate, click, fill forms, screenshot.",
        transport="stdio",
        command="npx -y @playwright/mcp@latest",
        runtime="npx",
        docs="https://github.com/microsoft/playwright-mcp",
        agents=("research", "testing"),
    ),
    ConnectorSpec(
        "notion",
        "Notion",
        "Search, read, and write Notion pages and databases. Connect opens "
        "Notion's consent page — approve once and you're in.",
        transport="http",
        url="https://mcp.notion.com/mcp",
        auth="oauth",
        docs="https://developers.notion.com/docs/mcp",
        agents=("documentation", "research", "comms"),
    ),
    ConnectorSpec(
        "context7",
        "Context7 docs",
        "Up-to-date library documentation and code examples for prompts.",
        transport="stdio",
        command="npx -y @upstash/context7-mcp",
        runtime="npx",
        docs="https://github.com/upstash/context7",
        agents=("coding", "research"),
    ),
    ConnectorSpec(
        "sentry",
        "Sentry",
        "Look up and analyze Sentry issues and stack traces. Connect opens "
        "Sentry's consent page.",
        transport="http",
        url="https://mcp.sentry.dev/mcp",
        auth="oauth",
        docs="https://docs.sentry.io/product/sentry-mcp/",
        agents=("reviewer", "coding"),
    ),
    ConnectorSpec(
        "linear",
        "Linear",
        "Search, create, and update Linear issues and projects. Connect opens "
        "Linear's consent page.",
        transport="http",
        url="https://mcp.linear.app/sse",
        auth="oauth",
        docs="https://linear.app/docs/mcp",
        agents=("comms", "research", "reviewer"),
    ),
    ConnectorSpec(
        "atlassian",
        "Atlassian (Jira & Confluence)",
        "Search and update Jira issues and Confluence pages. Connect opens "
        "Atlassian's consent page.",
        transport="http",
        url="https://mcp.atlassian.com/v1/sse",
        auth="oauth",
        docs="https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/",
        agents=("comms", "research", "documentation"),
    ),
    ConnectorSpec(
        "asana",
        "Asana",
        "Read and manage Asana tasks and projects. Connect opens Asana's " "consent page.",
        transport="http",
        url="https://mcp.asana.com/sse",
        auth="oauth",
        docs="https://developers.asana.com/docs/using-asanas-mcp-server",
        agents=("comms", "research"),
    ),
    ConnectorSpec(
        "zapier",
        "Zapier (Gmail, Calendar, 7000+ apps)",
        "One OAuth connection to your Zapier account unlocks Gmail, Google "
        "Calendar, Drive, Sheets, and thousands of other apps — each approved "
        "with its own consent screen inside Zapier.",
        transport="http",
        url="https://mcp.zapier.com/api/mcp/mcp",
        auth="oauth",
        docs="https://zapier.com/mcp",
        agents=("comms",),
    ),
    ConnectorSpec(
        "time",
        "Time",
        "Current time and timezone conversions.",
        transport="stdio",
        command="uvx mcp-server-time",
        runtime="uvx",
        docs="https://github.com/modelcontextprotocol/servers/tree/main/src/time",
    ),
    ConnectorSpec(
        "sequential_thinking",
        "Sequential thinking",
        "Structured step-by-step reasoning scratchpad for complex problems.",
        transport="stdio",
        command="npx -y @modelcontextprotocol/server-sequentialthinking",
        runtime="npx",
        docs="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking",
        agents=("research", "coding"),
    ),
    ConnectorSpec(
        "custom_stdio",
        "Custom (stdio)",
        "Any MCP server launched as a local command.",
        transport="stdio",
        command="{command}",
        params=(
            ParamSpec(
                "command",
                "Command line",
                'Full launch command, e.g. "npx -y my-mcp-server --flag".',
            ),
        ),
        docs="https://modelcontextprotocol.io/examples",
    ),
    ConnectorSpec(
        "custom_http",
        "Custom (HTTP)",
        "Any remote MCP server over streamable HTTP or SSE.",
        transport="http",
        url="{url}",
        headers={"Authorization": "Bearer {bearer_token}"},
        params=(
            ParamSpec("url", "Server URL", "e.g. https://host/mcp"),
            ParamSpec("bearer_token", "Bearer token", secret=True, required=False),
        ),
        docs="https://modelcontextprotocol.io/examples",
    ),
)

_BY_ID = {c.id: c for c in CONNECTOR_CATALOG}


def connector_for(connector_id: str) -> ConnectorSpec:
    if connector_id not in _BY_ID:
        raise KeyError(connector_id)
    return _BY_ID[connector_id]


__all__ = ["CONNECTOR_CATALOG", "ConnectorSpec", "ParamSpec", "connector_for"]
