"""Low-level resource download helpers (YouTube, HTTP, Hugging Face, Kaggle, GitHub)."""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ai_employee.core.logging import get_logger

log = get_logger("resource_download")

_YT_RE = re.compile(r"(youtube\.com|youtu\.be)", re.I)
_HF_PREFIX_RE = re.compile(
    r"huggingface\.co/(datasets|models)/([^/\s?#]+/[^/\s?#]+)",
    re.I,
)
# Standard model page: huggingface.co/Org/Repo (no /models/ prefix)
_HF_SLUG_RE = re.compile(
    r"huggingface\.co/(?!datasets/|models/|spaces/|docs/|blog/|api/)([^/\s?#]+/[^/\s?#]+)",
    re.I,
)
_KAGGLE_RE = re.compile(r"kaggle\.com/datasets/([^/\s?#]+/[^/\s?#]+)", re.I)
_GITHUB_RE = re.compile(r"github\.com/([^/\s?#]+/[^/\s?#]+)", re.I)
_AGE_RESTRICT_RE = re.compile(
    r"sign in to confirm your age|age.restr|inappropriate for some users",
    re.I,
)
_TRAILING_URL_JUNK = re.compile(r"[`.,;:!?)\]>]+$")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MODEL_FILE_EXTS = {
    ".bin",
    ".ckpt",
    ".h5",
    ".joblib",
    ".onnx",
    ".pb",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}


def sanitize_url(url: str) -> str:
    """Strip markdown/backtick junk agents often append to URLs."""
    u = (url or "").strip().strip("`")
    u = _TRAILING_URL_JUNK.sub("", u)
    return u.strip()


def parse_huggingface_url(url: str) -> tuple[str, str] | None:
    """Return (repo_type, repo_id) for a Hugging Face URL, or None."""
    u = (url or "").strip()
    m = _HF_PREFIX_RE.search(u)
    if m:
        kind = m.group(1).lower()
        repo_type = "dataset" if kind == "datasets" else "model"
        return repo_type, m.group(2)
    m = _HF_SLUG_RE.search(u)
    if m:
        return "model", m.group(1)
    return None


def detect_source(url: str) -> str:
    u = sanitize_url(url)
    if _YT_RE.search(u):
        return "youtube"
    if parse_huggingface_url(u):
        return "huggingface"
    if _KAGGLE_RE.search(u):
        return "kaggle"
    if _GITHUB_RE.search(u):
        return "github"
    if "roboflow.com" in u.lower():
        return "roboflow"
    return "http"


def prioritize_download_urls(urls: list[str]) -> list[str]:
    """Put YouTube/HF first — those hosts have automated downloaders."""
    order = {"youtube": 0, "huggingface": 1, "github": 2, "kaggle": 3}

    def _key(item: str) -> tuple[int, str]:
        return order.get(detect_source(item), 99), item

    seen: set[str] = set()
    ranked = sorted((sanitize_url(u) for u in urls if sanitize_url(u)), key=_key)
    out: list[str] = []
    for u in ranked:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _strip_workspace_prefix(path_str: str, ws: Path) -> str:
    """Agents sometimes pass ``data/workspace/foo`` when ``foo`` was intended."""
    parts = list(Path(path_str.replace("\\", "/").lstrip("./")).parts)
    ws_parts = list(ws.resolve().parts)
    for n in range(min(len(parts), len(ws_parts)), 0, -1):
        if parts[:n] == ws_parts[-n:]:
            return str(Path(*parts[n:])) if n < len(parts) else ""
    return str(Path(*parts)) if parts else ""


def _safe_dest(workspace: Path, dest: str) -> Path:
    ws = workspace.resolve()
    target = Path(dest.strip())
    if not target.is_absolute():
        rel = _strip_workspace_prefix(str(target), ws)
        target = ws / (rel or ".")
    target = target.resolve()
    if ws not in target.parents and target != ws:
        raise ValueError(f"destination must be inside workspace ({ws})")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def canonical_artifact_dest(
    workspace: Path,
    session_id: str | None,
    *,
    url: str,
    source_hint: str,
    requested_dest: str,
) -> Path:
    """Route downloads into one of three workspace buckets.

    - ``models/``   shared model weights/checkpoints
    - ``data/``     shared reusable datasets/repos/assets
    - ``sessions/`` session-specific files for one task run
    """
    ws = workspace.resolve()
    requested = _requested_relative_path(requested_dest, ws)
    bucket = _bucket_for_artifact(url, source_hint, requested)
    if bucket == "models":
        name = _artifact_name(url, requested, prefer_dir=_dir_like_resource(url, source_hint))
        target = ws / "models" / name
    elif bucket == "data":
        rel = _shared_data_relative(url, requested, source_hint)
        target = ws / "data" / rel
    else:
        rel = _session_relative(url, requested, source_hint)
        target = ws / "sessions" / _safe_slug(session_id or "default") / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def _requested_relative_path(dest: str, workspace: Path) -> Path:
    rel = _strip_workspace_prefix(dest or "", workspace.resolve()).strip()
    if not rel:
        return Path()
    p = Path(rel)
    if p.is_absolute():
        return Path(p.name)
    return p


def _normalized_requested_relative(requested: Path) -> Path | None:
    if requested and str(requested) not in {"", "."}:
        parts = list(requested.parts)
        while parts and parts[0].lower() in {
            "data",
            "datasets",
            "downloads",
            "model",
            "models",
            "repos",
            "resources",
            "session",
            "sessions",
            "videos",
            "workspace",
        }:
            parts.pop(0)
        if parts:
            return Path(*parts)
    return None


def _session_relative(url: str, requested: Path, source_hint: str) -> Path:
    rel = _normalized_requested_relative(requested)
    if rel is not None:
        return rel
    name = _artifact_name(url, requested, prefer_dir=_dir_like_resource(url, source_hint))
    return Path(name)


def _shared_data_relative(url: str, requested: Path, source_hint: str) -> Path:
    rel = _normalized_requested_relative(requested)
    if rel is not None:
        return rel
    name = _artifact_name(url, requested, prefer_dir=_dir_like_resource(url, source_hint))
    return Path(name)


def _artifact_name(url: str, requested: Path, *, prefer_dir: bool) -> str:
    requested_name = requested.name if requested and requested.name not in {"", "."} else ""
    if requested_name:
        if prefer_dir and requested.suffix:
            return _safe_slug(requested.stem)
        return requested_name
    url_name = Path(urlparse(sanitize_url(url)).path).name
    if url_name:
        if prefer_dir and "." in url_name:
            return _safe_slug(Path(url_name).stem)
        return url_name
    return "artifact"


def _dir_like_resource(url: str, source_hint: str) -> bool:
    kind = source_hint if source_hint != "auto" else detect_source(url)
    if kind in {"github", "kaggle"}:
        return True
    parsed = parse_huggingface_url(url)
    return bool(parsed and parsed[0] in {"dataset", "model"})


def _bucket_for_artifact(url: str, source_hint: str, requested: Path) -> str:
    parts = [p.lower() for p in requested.parts]
    if "models" in parts or "model" in parts:
        return "models"
    if "session" in parts or "sessions" in parts:
        return "sessions"
    if any(p in {"data", "dataset", "datasets", "repo", "repos"} for p in parts):
        return "data"
    if _is_model_artifact(url, source_hint, requested):
        return "models"
    if _is_shared_data_artifact(url, source_hint):
        return "data"
    return "sessions"


def _is_model_artifact(url: str, source_hint: str, requested: Path) -> bool:
    parsed = parse_huggingface_url(url)
    if parsed and parsed[0] == "model":
        return True
    parts = [p.lower() for p in requested.parts]
    if "models" in parts or "model" in parts:
        return True
    name = requested.name if requested and requested.name not in {"", "."} else ""
    if not name:
        name = Path(urlparse(sanitize_url(url)).path).name
    suffix = Path(name).suffix.lower()
    if suffix in _MODEL_FILE_EXTS:
        return True
    lowered = name.lower()
    return any(token in lowered for token in ("checkpoint", "weights", "model"))


def _is_shared_data_artifact(url: str, source_hint: str) -> bool:
    kind = source_hint if source_hint != "auto" else detect_source(url)
    if kind in {"github", "kaggle"}:
        return True
    parsed = parse_huggingface_url(url)
    return bool(parsed and parsed[0] == "dataset")


def _safe_slug(text: str) -> str:
    value = _NON_ALNUM_RE.sub("-", (text or "").strip()).strip("-._")
    return value or "default"


def build_ytdlp_command(
    url: str,
    out_tpl: str,
    *,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
    format_selector: str = "best[ext=mp4]/b/bv*+ba/b",
    extra_args: list[str] | None = None,
    js_runtimes: str | None = None,
) -> list[str]:
    """Assemble a yt-dlp argv list. Cookies unlock age-restricted YouTube videos."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--print",
        "after_move:%(filepath)s",
        "-f",
        format_selector,
        "-o",
        out_tpl,
        *(extra_args or []),
    ]
    if cookies_file or cookies_from_browser:
        runtime = (js_runtimes or "").strip() or _default_js_runtime()
        if runtime:
            cmd.extend(["--js-runtimes", runtime])
    if cookies_file:
        cf = Path(cookies_file).expanduser()
        if not cf.is_absolute():
            cf = cf.resolve()
        cmd.extend(["--cookies", str(cf)])
    elif cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser.strip()])
    cmd.append(url)
    return cmd


def _default_js_runtime() -> str | None:
    """Age-restricted / web-client downloads need a JS runtime for YouTube challenges."""
    if shutil.which("node"):
        return "node"
    if shutil.which("deno"):
        return "deno"
    return None


# Android client does NOT use cookies (yt-dlp skips it). When cookies are set,
# try web/tv clients first so age-restricted videos can authenticate.
_YTDLP_COOKIE_STRATEGIES: list[dict[str, Any]] = [
    {
        "format_selector": "best[ext=mp4]/best/best",
        "extra_args": ["--extractor-args", "youtube:player_client=web"],
    },
    {
        "format_selector": "best",
        "extra_args": ["--extractor-args", "youtube:player_client=tv"],
    },
    {
        "format_selector": "best[ext=mp4]/best/best",
        "extra_args": ["--extractor-args", "youtube:player_client=tv_embedded"],
    },
]

# Fallback strategies when the default web client hits YouTube bot challenges (no cookies).
_YTDLP_STRATEGIES: list[dict[str, Any]] = [
    {
        "format_selector": "best[ext=mp4]/b/bv*+ba/b",
        "extra_args": ["--extractor-args", "youtube:player_client=android"],
    },
    {
        "format_selector": "best",
        "extra_args": ["--extractor-args", "youtube:player_client=android,web"],
    },
    {
        "format_selector": "best[ext=mp4]/best/best",
        "extra_args": [],
    },
]


def youtube_needs_cookies(error: str) -> bool:
    return bool(_AGE_RESTRICT_RE.search(error or ""))


def inspect_youtube_cookies_file(path: str | Path) -> dict[str, Any]:
    """Check whether a Netscape cookies.txt looks like a logged-in YouTube session."""
    p = Path(path).expanduser()
    if not p.is_file():
        return {"path": str(p), "exists": False, "logged_in": False, "cookie_names": []}
    names: set[str] = set()
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 6:
            names.add(parts[5].strip())
    logged_in = "LOGIN_INFO" in names or ("SID" in names and "SAPISID" in names)
    return {
        "path": str(p.resolve()),
        "exists": True,
        "logged_in": logged_in,
        "cookie_names": sorted(names),
        "cookie_count": len(names),
    }


def _age_restricted_hint(
    error: str,
    *,
    cookies_file: str | Path | None,
    cookies_from_browser: str | None,
) -> str:
    if not youtube_needs_cookies(error):
        return ""
    if cookies_file:
        status = inspect_youtube_cookies_file(cookies_file)
        if status.get("exists") and not status.get("logged_in"):
            return (
                "\n\n[hint] YTDLP_COOKIES_FILE is set and passed to yt-dlp, but the file "
                "does not contain a YouTube login session (missing LOGIN_INFO / SID). "
                "Re-export cookies while signed into YouTube (and after confirming your age "
                "for this video in the browser), or set YTDLP_COOKIES_FROM_BROWSER=chrome "
                "to read live cookies from your browser."
            )
        return (
            "\n\n[hint] Cookies were used but YouTube still blocked the download. "
            "Open the video in your browser, confirm your age, then refresh the cookies file "
            "or use YTDLP_COOKIES_FROM_BROWSER."
        )
    if not cookies_from_browser:
        return (
            "\n\n[hint] This video is age-restricted. Set YTDLP_COOKIES_FILE or "
            "YTDLP_COOKIES_FROM_BROWSER in .env so yt-dlp can authenticate."
        )
    return (
        "\n\n[hint] Browser cookies were used but age verification failed. "
        "Sign into YouTube in that browser and confirm your age for this video."
    )


_AUTO_BROWSERS = ("chrome", "chromium", "brave", "firefox", "edge")


def _cookie_sources(
    cookies_from_browser: str | None,
    cookies_file: str | Path | None,
) -> list[tuple[str | None, str | Path | None]]:
    """Build cookie sources: logged-in file, explicit browser, or auto browser fallback."""
    sources: list[tuple[str | None, str | Path | None]] = []
    file_logged_in = False
    if cookies_file:
        status = inspect_youtube_cookies_file(cookies_file)
        if status.get("logged_in"):
            sources.append(("file", cookies_file))
            file_logged_in = True
        elif status.get("exists"):
            log.warning(
                "youtube.cookies_not_logged_in",
                path=status.get("path"),
                cookie_count=status.get("cookie_count"),
            )
    browser = (cookies_from_browser or "").strip()
    if browser:
        sources.append(("browser", browser))
    elif not file_logged_in:
        # cookies.txt missing LOGIN_INFO — read live session from the local browser.
        for name in _AUTO_BROWSERS:
            sources.append(("browser", name))
    if not sources:
        sources.append((None, None))
    return sources


def _strategies_for_source(source: str | None) -> list[dict[str, Any]]:
    if source:
        return _YTDLP_COOKIE_STRATEGIES + _YTDLP_STRATEGIES
    return list(_YTDLP_STRATEGIES)


async def download_http(url: str, dest: Path, *, timeout: int = 120) -> dict[str, Any]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        resp = await client.get(url, headers={"User-Agent": "AIEmployee/0.1"})
    resp.raise_for_status()
    content = resp.content
    await asyncio.to_thread(dest.write_bytes, content)
    return {"path": str(dest), "bytes": len(content), "source": "http"}


def _resolve_ytdlp_path(candidate: Path, workspace: Path) -> Path:
    """yt-dlp runs with cwd=workspace; relative paths are relative to that."""
    p = candidate.expanduser()
    if not p.is_absolute():
        p = (workspace / p).resolve()
    else:
        p = p.resolve()
    return p


async def download_youtube(
    url: str,
    dest: Path,
    *,
    workspace: Path,
    run_shell,
    cookies_from_browser: str | None = None,
    cookies_file: str | Path | None = None,
) -> dict[str, Any]:
    """``run_shell`` is an async callable(cmd: list[str]) -> (code, stdout, stderr)."""
    ws = workspace.resolve()
    dest_abs = dest.resolve() if dest.is_absolute() else (ws / dest).resolve()
    try:
        out_tpl = str(dest_abs.relative_to(ws).with_suffix("")) + ".%(ext)s"
    except ValueError:
        out_tpl = str(dest_abs.with_suffix("")) + ".%(ext)s"
    errors: list[str] = []
    sources = _cookie_sources(cookies_from_browser, cookies_file)

    for source, cred in sources:
        browser = str(cred) if source == "browser" else None
        cfile = cred if source == "file" else None
        for strat in _strategies_for_source(source):
            cmd = build_ytdlp_command(
                url,
                out_tpl,
                cookies_from_browser=browser,
                cookies_file=cfile,
                format_selector=str(strat["format_selector"]),
                extra_args=list(strat.get("extra_args") or []),
            )
            code, out, err = await run_shell(cmd)
            combined = f"{err or ''}\n{out or ''}".strip()
            if code == 0:
                path: Path | None = None
                for line in (out or "").splitlines():
                    candidate = _resolve_ytdlp_path(Path(line.strip()), ws)
                    if candidate.is_file():
                        path = candidate
                        break
                if path is None:
                    parent = dest_abs.parent
                    stem = dest_abs.stem
                    matches = sorted(
                        parent.glob(f"{stem}.*"), key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    if matches:
                        path = matches[0]
                if path is None:
                    errors.append("yt-dlp reported success but no output file found")
                    continue
                meta: dict[str, Any] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "source": "youtube",
                    "strategy": strat.get("extra_args") or "default",
                    "cookie_source": source or "none",
                }
                if browser:
                    meta["cookies_from_browser"] = browser
                elif cfile:
                    meta["cookies_file"] = str(cfile)
                return meta
            errors.append(combined or f"yt-dlp exited {code}")

    last = errors[-1] if errors else "yt-dlp failed"
    tried_browser = cookies_from_browser or any(s[0] == "browser" for s in sources)
    hint = _age_restricted_hint(
        last,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser if tried_browser else None,
    )
    if hint:
        raise RuntimeError(last + hint)
    raise RuntimeError(last)


async def download_huggingface(
    url: str,
    dest: Path,
    *,
    workspace: Path,
    token: str | None = None,
    run_shell=None,  # kept for tests; HF uses huggingface_hub in-process
) -> dict[str, Any]:
    parsed = parse_huggingface_url(url)
    if not parsed:
        raise ValueError("not a huggingface datasets/models url")
    repo_type, repo_id = parsed
    dest_dir = dest if dest.suffix == "" else dest.parent / dest.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    hf_home = workspace.resolve() / ".hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok = (token or "").strip() or None

    def _download() -> str:
        import os

        from huggingface_hub import snapshot_download

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
        if tok:
            os.environ["HF_TOKEN"] = tok
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(dest_dir),
            cache_dir=str(cache_dir),
            token=tok,
        )

    try:
        await asyncio.to_thread(_download)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if tok is None and any(
            x in msg.lower() for x in ("gated", "authorized", "401", "private", "authenticate")
        ):
            msg += (
                "\n\n[hint] This Hugging Face repo may be gated. "
                "Set HF_TOKEN in .env or add a Hugging Face token in 🔑 Keys."
            )
        raise RuntimeError(msg) from exc

    return {
        "path": str(dest_dir),
        "bytes": sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()),
        "source": "huggingface",
        "repo_id": repo_id,
        "repo_type": repo_type,
    }


async def download_kaggle(url: str, dest: Path, *, run_shell, has_creds: bool) -> dict[str, Any]:
    if not has_creds:
        raise RuntimeError("KAGGLE_USERNAME and KAGGLE_KEY required — add them in 🔑 Keys")
    m = _KAGGLE_RE.search(url)
    if not m:
        raise ValueError("not a kaggle dataset url")
    ref = m.group(1)
    dest_dir = dest if dest.suffix == "" else dest.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    code, out, err = await run_shell(
        ["kaggle", "datasets", "download", "-d", ref, "-p", str(dest_dir), "--unzip"]
    )
    if code != 0:
        raise RuntimeError(err or out or f"kaggle exited {code}")
    return {"path": str(dest_dir), "source": "kaggle", "stdout": out[:500]}


async def download_github(url: str, dest: Path, *, run_shell) -> dict[str, Any]:
    m = _GITHUB_RE.search(url)
    if not m:
        raise ValueError("not a github repo url")
    repo = m.group(1).removesuffix(".git")
    dest_dir = dest if dest.suffix == "" else dest.parent / dest.stem
    if dest_dir.exists():
        import shutil

        shutil.rmtree(dest_dir, ignore_errors=True)
    code, out, err = await run_shell(
        ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest_dir)]
    )
    if code != 0:
        # Fallback: zipball over HTTP.
        zip_url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
        parts = repo.split("/")
        if len(parts) == 2:
            zip_url = f"https://github.com/{repo}/archive/refs/heads/master.zip"
        try:
            await download_http(zip_url, dest_dir.with_suffix(".zip"))
            return {"path": str(dest_dir.with_suffix(".zip")), "source": "github_zip"}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(err or out or str(exc)) from exc
    return {"path": str(dest_dir), "source": "github"}


__all__ = [
    "detect_source",
    "sanitize_url",
    "prioritize_download_urls",
    "parse_huggingface_url",
    "build_ytdlp_command",
    "youtube_needs_cookies",
    "inspect_youtube_cookies_file",
    "_YTDLP_STRATEGIES",
    "_YTDLP_COOKIE_STRATEGIES",
    "download_http",
    "download_youtube",
    "download_huggingface",
    "download_kaggle",
    "download_github",
    "_safe_dest",
]
