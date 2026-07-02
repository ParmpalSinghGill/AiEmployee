"""Low-level resource download helpers (YouTube, HTTP, Hugging Face, Kaggle, GitHub)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx

_YT_RE = re.compile(r"(youtube\.com|youtu\.be)", re.I)
_HF_RE = re.compile(r"huggingface\.co/(datasets|models)/([^/\s?#]+)", re.I)
_KAGGLE_RE = re.compile(r"kaggle\.com/datasets/([^/\s?#]+/[^/\s?#]+)", re.I)
_GITHUB_RE = re.compile(r"github\.com/([^/\s?#]+/[^/\s?#]+)", re.I)


def detect_source(url: str) -> str:
    u = (url or "").strip()
    if _YT_RE.search(u):
        return "youtube"
    if _HF_RE.search(u):
        return "huggingface"
    if _KAGGLE_RE.search(u):
        return "kaggle"
    if _GITHUB_RE.search(u):
        return "github"
    if "roboflow.com" in u.lower():
        return "roboflow"
    return "http"


def _safe_dest(workspace: Path, dest: str) -> Path:
    ws = workspace.resolve()
    target = Path(dest)
    if not target.is_absolute():
        target = ws / target
    target = target.resolve()
    if ws not in target.parents and target != ws:
        raise ValueError(f"destination must be inside workspace ({ws})")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


async def download_http(url: str, dest: Path, *, timeout: int = 120) -> dict[str, Any]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        resp = await client.get(url, headers={"User-Agent": "AIEmployee/0.1"})
    resp.raise_for_status()
    content = resp.content
    await asyncio.to_thread(dest.write_bytes, content)
    return {"path": str(dest), "bytes": len(content), "source": "http"}


async def download_youtube(
    url: str, dest: Path, *, workspace: Path, run_shell
) -> dict[str, Any]:
    """``run_shell`` is an async callable(cmd: list[str]) -> (code, stdout, stderr)."""
    out_tpl = str(dest.with_suffix("")) + ".%(ext)s"
    code, out, err = await run_shell(
        ["yt-dlp", "-f", "best[ext=mp4]/best/best", "-o", out_tpl, url]
    )
    if code != 0:
        raise RuntimeError(err or out or f"yt-dlp exited {code}")
    # Resolve actual file (extension chosen by yt-dlp).
    parent = dest.parent
    stem = dest.stem
    matches = sorted(parent.glob(f"{stem}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        matches = sorted(parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError("yt-dlp reported success but no output file found")
    path = matches[0]
    return {"path": str(path), "bytes": path.stat().st_size, "source": "youtube"}


async def download_huggingface(url: str, dest: Path, *, run_shell) -> dict[str, Any]:
    m = _HF_RE.search(url)
    if not m:
        raise ValueError("not a huggingface datasets/models url")
    kind, repo_id = m.group(1), m.group(2)
    dest_dir = dest if dest.suffix == "" else dest.parent / dest.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    code, out, err = await run_shell(
        [
            "python",
            "-c",
            (
                "from huggingface_hub import snapshot_download; "
                f"p=snapshot_download(repo_id={repo_id!r}, repo_type={kind!r}, "
                f"local_dir={str(dest_dir)!r}); print(p)"
            ),
        ]
    )
    if code != 0:
        raise RuntimeError(err or out or f"hf download exited {code}")
    return {"path": str(dest_dir), "bytes": sum(f.stat().st_size for f in dest_dir.rglob('*') if f.is_file()), "source": "huggingface"}


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
    code, out, err = await run_shell(["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest_dir)])
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


__all__ = ["detect_source", "download_http", "download_youtube", "download_huggingface", "download_kaggle", "download_github", "_safe_dest"]
