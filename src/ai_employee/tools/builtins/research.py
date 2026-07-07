"""Research-assistant tools: discover models, datasets and papers across the
major ML hubs, and estimate the hardware needed to run/train a model.

Every network call degrades gracefully — on any error (offline, rate-limited,
missing key) the tool returns ``ok=False`` with an empty ``output`` list instead
of raising, so the offline test suite (golden rule #3) stays green and the
workflow can branch on "found nothing" rather than crashing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field

from ai_employee.config import Settings
from ai_employee.tools.base import Tool, ToolResult

_UA = {"User-Agent": "AIEmployee-Research/0.1"}


async def _get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 20) -> Any:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers={**_UA, **(headers or {})})
    resp.raise_for_status()
    return resp.json()


def _empty(error: str) -> ToolResult:
    return ToolResult(ok=False, error=error, output=[])


# ── Hugging Face Hub ─────────────────────────────────────────────────────────
class HfSearchTool(Tool):
    name = "hf_search"
    description = (
        "Search the Hugging Face Hub for pretrained models and/or datasets matching "
        "a query. Returns id, downloads, likes, task/pipeline_tag and url. Use this "
        "first — it is the backbone for finding both a model and its data."
    )
    category = "research"

    class Args(BaseModel):
        query: str
        kind: str = Field("both", pattern="^(models|datasets|both)$")
        limit: int = Field(8, ge=1, le=30)

    async def _run(self, args: HfSearchTool.Args) -> ToolResult:
        q = quote_plus(args.query)
        out: dict[str, list[dict[str, Any]]] = {}
        errors: list[str] = []
        kinds = ["models", "datasets"] if args.kind == "both" else [args.kind]
        for kind in kinds:
            url = (
                f"https://huggingface.co/api/{kind}?search={q}"
                f"&limit={args.limit}&sort=downloads&direction=-1"
            )
            try:
                data = await _get_json(url)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}: {exc}")
                out[kind] = []
                continue
            out[kind] = [
                {
                    "id": item.get("id") or item.get("modelId"),
                    "url": f"https://huggingface.co/{'datasets/' if kind == 'datasets' else ''}"
                    + str(item.get("id") or item.get("modelId")),
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "task": item.get("pipeline_tag"),
                    "tags": item.get("tags", [])[:8],
                }
                for item in (data or [])
            ]
        ok = any(out.get(k) for k in kinds)
        return ToolResult(
            ok=ok or not errors,
            output=out,
            error="; ".join(errors) or None,
            metadata={"source": "huggingface"},
        )


# ── Papers With Code ─────────────────────────────────────────────────────────
class PapersWithCodeSearchTool(Tool):
    name = "paperswithcode_search"
    description = (
        "Search Papers With Code: maps a task/problem to papers, their official code "
        "repositories and the datasets/benchmarks they use. Best source for the "
        "'find the paper' branch when no ready-made model exists."
    )
    category = "research"

    class Args(BaseModel):
        query: str
        limit: int = Field(6, ge=1, le=20)

    async def _run(self, args: PapersWithCodeSearchTool.Args) -> ToolResult:
        url = f"https://paperswithcode.com/api/v1/search/?q={quote_plus(args.query)}"
        try:
            data = await _get_json(url)
        except Exception as exc:  # noqa: BLE001
            return _empty(f"paperswithcode unavailable: {exc}")
        results = []
        for row in (data.get("results") or [])[: args.limit]:
            paper = row.get("paper") or {}
            repo = row.get("repository") or {}
            results.append(
                {
                    "paper_title": paper.get("title"),
                    "paper_url": paper.get("url_abs") or paper.get("url_pdf"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "repo_url": repo.get("url"),
                    "repo_stars": repo.get("stars"),
                    "is_official": row.get("is_official"),
                }
            )
        return ToolResult(output=results, metadata={"source": "paperswithcode"})


# ── arXiv ────────────────────────────────────────────────────────────────────
_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = (
        "Search arXiv for papers matching a query. Returns title, authors, abstract, "
        "pdf url and arxiv id. Use to locate the method paper to reproduce."
    )
    category = "research"

    class Args(BaseModel):
        query: str
        limit: int = Field(5, ge=1, le=15)

    async def _run(self, args: ArxivSearchTool.Args) -> ToolResult:
        url = (
            "http://export.arxiv.org/api/query?search_query=all:"
            f"{quote_plus(args.query)}&start=0&max_results={args.limit}"
            "&sortBy=relevance"
        )
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers=_UA)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as exc:  # noqa: BLE001
            return _empty(f"arxiv unavailable: {exc}")
        results = []
        for entry in root.findall(f"{_ATOM}entry"):
            pdf = ""
            for link in entry.findall(f"{_ATOM}link"):
                if link.get("title") == "pdf":
                    pdf = link.get("href", "")
            results.append(
                {
                    "title": (entry.findtext(f"{_ATOM}title") or "").strip(),
                    "abstract": (entry.findtext(f"{_ATOM}summary") or "").strip()[:600],
                    "authors": [
                        a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")
                    ][:6],
                    "url": (entry.findtext(f"{_ATOM}id") or "").strip(),
                    "pdf_url": pdf,
                }
            )
        return ToolResult(output=results, metadata={"source": "arxiv"})


# ── GitHub ───────────────────────────────────────────────────────────────────
class GithubSearchTool(Tool):
    name = "github_search"
    description = (
        "Search GitHub repositories for reference implementations. Returns full name, "
        "stars, description and url. Uses GITHUB_TOKEN if set to raise the rate limit."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        query: str
        limit: int = Field(6, ge=1, le=20)

    async def _run(self, args: GithubSearchTool.Args) -> ToolResult:
        url = (
            "https://api.github.com/search/repositories?q="
            f"{quote_plus(args.query)}&sort=stars&order=desc&per_page={args.limit}"
        )
        headers = {"Accept": "application/vnd.github+json"}
        if self._settings.github_token:
            headers["Authorization"] = f"Bearer {self._settings.github_token}"
        try:
            data = await _get_json(url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            return _empty(f"github unavailable: {exc}")
        results = [
            {
                "name": r.get("full_name"),
                "url": r.get("html_url"),
                "stars": r.get("stargazers_count"),
                "description": r.get("description"),
                "language": r.get("language"),
            }
            for r in (data.get("items") or [])
        ]
        return ToolResult(output=results, metadata={"source": "github"})


# ── Kaggle ───────────────────────────────────────────────────────────────────
class KaggleSearchTool(Tool):
    name = "kaggle_search"
    description = (
        "Search Kaggle datasets. Requires KAGGLE_USERNAME and KAGGLE_KEY; without "
        "them it reports that credentials are needed instead of failing the run."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        query: str
        limit: int = Field(8, ge=1, le=20)

    async def _run(self, args: KaggleSearchTool.Args) -> ToolResult:
        user, key = self._settings.kaggle_username, self._settings.kaggle_key
        if not (user and key):
            return ToolResult(
                ok=True,
                output=[],
                metadata={"source": "kaggle", "note": "set KAGGLE_USERNAME/KAGGLE_KEY to enable"},
            )
        url = f"https://www.kaggle.com/api/v1/datasets/list?search={quote_plus(args.query)}"
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers=_UA, auth=(user, key))
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return _empty(f"kaggle unavailable: {exc}")
        results = [
            {
                "ref": r.get("ref"),
                "title": r.get("title"),
                "url": f"https://www.kaggle.com/datasets/{r.get('ref')}",
                "size": r.get("totalBytes"),
                "downloads": r.get("downloadCount"),
            }
            for r in (data or [])[: args.limit]
        ]
        return ToolResult(output=results, metadata={"source": "kaggle"})


# ── Roboflow Universe ────────────────────────────────────────────────────────
class RoboflowSearchTool(Tool):
    name = "roboflow_search"
    description = (
        "Search Roboflow Universe for computer-vision datasets and models (object "
        "detection, classification, segmentation). Great for CV problems. Requires "
        "ROBOFLOW_API_KEY; without it, reports that a key is needed instead of failing. "
        "Supports inline filters in the query, e.g. 'helmet model:yolov8' or 'cars images>1000'."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    class Args(BaseModel):
        query: str
        limit: int = Field(8, ge=1, le=30)

    async def _run(self, args: RoboflowSearchTool.Args) -> ToolResult:
        key = self._settings.roboflow_api_key
        if not key:
            return ToolResult(
                ok=True,
                output=[],
                metadata={"source": "roboflow", "note": "set ROBOFLOW_API_KEY to enable"},
            )
        url = (
            "https://api.roboflow.com/universe/search"
            f"?q={quote_plus(args.query)}&api_key={quote_plus(key)}"
        )
        try:
            data = await _get_json(url)
        except Exception as exc:  # noqa: BLE001
            return _empty(f"roboflow unavailable: {exc}")
        # Response wrapper varies; accept a bare list or a common results key.
        if isinstance(data, dict):
            rows = (
                data.get("results")
                or data.get("projects")
                or data.get("datasets")
                or data.get("hits")
                or []
            )
        else:
            rows = data or []
        results = [
            {
                "name": r.get("name"),
                "url": r.get("url"),
                "type": r.get("type"),
                "stars": r.get("stars"),
                "images": r.get("images"),
                "thumbnail": r.get("annotationThumbnail") or r.get("thumbnail"),
            }
            for r in rows[: args.limit]
        ]
        return ToolResult(output=results, metadata={"source": "roboflow"})


# ── Hardware estimate ────────────────────────────────────────────────────────
_BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}


class EstimateHardwareTool(Tool):
    name = "estimate_hardware"
    description = (
        "Estimate the GPU VRAM needed to run inference or training for a model, from "
        "its parameter count (or a Hugging Face model id, whose param count is looked "
        "up). Compares against the local GPU (LOCAL_GPU_VRAM_GB) and says whether it "
        "fits locally or needs a bigger/cloud GPU."
    )
    category = "research"

    def __init__(self, settings: Settings) -> None:
        self._local_vram = settings.local_gpu_vram_gb

    class Args(BaseModel):
        params_billions: float | None = Field(
            None, description="Model size in billions of parameters, if known."
        )
        model_id: str | None = Field(
            None, description="Hugging Face model id to look up param count for."
        )
        dtype: str = Field("fp16", pattern="^(fp32|fp16|bf16|int8|int4)$")

    async def _run(self, args: EstimateHardwareTool.Args) -> ToolResult:
        params_b = args.params_billions
        source = "provided"
        if params_b is None and args.model_id:
            params_b, source = await self._lookup_params(args.model_id)
        if params_b is None:
            return ToolResult(
                ok=False,
                error="need params_billions or a resolvable model_id",
            )
        bpp = _BYTES_PER_PARAM[args.dtype]
        weights_gb = params_b * 1e9 * bpp / 1024**3
        # Inference: weights + ~20% for activations/KV/overhead.
        infer_gb = round(weights_gb * 1.2, 2)
        # Full fine-tune with Adam (mixed precision): weights + grads + 2 optimizer
        # states ≈ 16 bytes/param effective.
        train_gb = round(params_b * 1e9 * 16 / 1024**3, 2)
        local = self._local_vram
        verdict = {
            "inference_fits_locally": local > 0 and infer_gb <= local,
            "full_finetune_fits_locally": local > 0 and train_gb <= local,
            "local_gpu_vram_gb": local or "unknown (set LOCAL_GPU_VRAM_GB)",
        }
        recommendation = self._recommend(infer_gb, train_gb, local)
        return ToolResult(
            output={
                "params_billions": round(params_b, 3),
                "params_source": source,
                "dtype": args.dtype,
                "inference_vram_gb": infer_gb,
                "full_finetune_vram_gb": train_gb,
                **verdict,
                "recommendation": recommendation,
            },
            metadata={"source": "estimate"},
        )

    async def _lookup_params(self, model_id: str) -> tuple[float | None, str]:
        try:
            data = await _get_json(f"https://huggingface.co/api/models/{model_id}")
        except Exception:  # noqa: BLE001
            return None, "lookup_failed"
        total = (data.get("safetensors") or {}).get("total")
        if isinstance(total, (int, float)) and total > 0:
            return total / 1e9, "huggingface_safetensors"
        return None, "unknown_on_hub"

    @staticmethod
    def _recommend(infer_gb: float, train_gb: float, local: float) -> str:
        def tier(gb: float) -> str:
            if gb <= 8:
                return "consumer GPU (e.g. RTX 3060/4060, 8GB)"
            if gb <= 16:
                return "16GB GPU (e.g. RTX 4080 / A4000)"
            if gb <= 24:
                return "24GB GPU (e.g. RTX 3090/4090, A5000)"
            if gb <= 48:
                return "48GB GPU (e.g. A6000) or 2×24GB"
            if gb <= 80:
                return "80GB GPU (A100/H100) or multi-GPU"
            return "multi-GPU / cloud cluster"

        msg = f"Inference needs ~{infer_gb}GB → {tier(infer_gb)}. "
        msg += f"Full fine-tune needs ~{train_gb}GB → {tier(train_gb)}."
        if local > 0:
            if infer_gb > local:
                msg += f" Inference exceeds your {local}GB GPU — use cloud or quantize."
            elif train_gb > local:
                msg += f" Inference fits your {local}GB GPU, but training does not."
            else:
                msg += f" Both fit your {local}GB GPU."
        return msg


def build_research_tools(settings: Settings) -> list[Tool]:
    return [
        HfSearchTool(),
        PapersWithCodeSearchTool(),
        ArxivSearchTool(),
        GithubSearchTool(settings),
        KaggleSearchTool(settings),
        RoboflowSearchTool(settings),
        EstimateHardwareTool(settings),
    ]


__all__ = [
    "HfSearchTool",
    "PapersWithCodeSearchTool",
    "ArxivSearchTool",
    "GithubSearchTool",
    "KaggleSearchTool",
    "RoboflowSearchTool",
    "EstimateHardwareTool",
    "build_research_tools",
]
