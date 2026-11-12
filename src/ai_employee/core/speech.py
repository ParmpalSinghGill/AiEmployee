"""Local speech-to-text for the console's 🎤 mic button (faster-whisper).

Audio never leaves the machine: the browser records a clip, posts it to
``/ui/api/transcribe``, and this module turns it into text with a Whisper model
loaded once and kept in memory. The model is fetched from Hugging Face on first
use only; ``faster-whisper`` decodes webm/ogg/wav/mp3 itself (bundled PyAV).

The import is lazy so the rest of the framework (and the offline test suite)
works when ``faster-whisper`` isn't installed — the endpoint then says so.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

from ai_employee.config import get_settings
from ai_employee.core.logging import get_logger

log = get_logger(__name__)


class SpeechUnavailableError(RuntimeError):
    """faster-whisper isn't installed, or the model could not be loaded."""


@dataclass
class Transcript:
    text: str
    language: str
    duration: float
    model: str
    device: str


def speech_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _cuda_ok() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 — any CUDA probe failure means "use CPU"
        return False


_SAMPLE_RATE = 16000  # what Whisper expects


def _decode_with_ffmpeg(audio: bytes) -> Any | None:
    """Clip → 16 kHz mono float32 samples via the system ffmpeg.

    Preferred over faster-whisper's own PyAV decoder, which breaks across PyAV
    releases (PyAV 19 rejects an argument it passes). ``None`` when ffmpeg isn't
    installed — the caller then lets faster-whisper decode the bytes itself.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    import numpy as np

    proc = subprocess.run(  # noqa: S603 — fixed argv, audio on stdin
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "pipe:1",
        ],
        input=audio,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError(err[-1] if err else "ffmpeg could not decode the audio")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


class _Transcriber:
    """One Whisper model per (name, device), loaded on first use."""

    def __init__(self) -> None:
        self._model: Any = None
        self._key: tuple[str, str, str] | None = None
        self._device = "cpu"
        self._load_lock = threading.Lock()
        # One clip at a time: concurrent decodes on one GPU only slow each other.
        self._run_lock = threading.Lock()

    def _wanted(self) -> tuple[str, str, str]:
        s = get_settings()
        device = (s.whisper_device or "auto").lower()
        if device == "auto":
            device = "cuda" if _cuda_ok() else "cpu"
        compute = (s.whisper_compute_type or "auto").lower()
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return s.whisper_model or "small", device, compute

    def _load(self) -> Any:
        key = self._wanted()
        with self._load_lock:
            if self._model is not None and self._key == key:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise SpeechUnavailableError(
                    "Voice input needs faster-whisper: pip install faster-whisper"
                ) from e
            name, device, compute = key
            log.info("speech.model_loading", model=name, device=device, compute=compute)
            try:
                model = WhisperModel(name, device=device, compute_type=compute)
            except Exception as e:  # noqa: BLE001
                if device != "cuda":
                    raise SpeechUnavailableError(f"Could not load Whisper '{name}': {e}") from e
                # Missing cuBLAS/cuDNN or no VRAM left: CPU still works.
                log.warning("speech.cuda_failed", error=str(e)[:300])
                device, compute = "cpu", "int8"
                try:
                    model = WhisperModel(name, device=device, compute_type=compute)
                except Exception as e2:  # noqa: BLE001
                    raise SpeechUnavailableError(f"Could not load Whisper '{name}': {e2}") from e2
            self._model, self._key, self._device = model, key, device
            log.info("speech.model_ready", model=name, device=device)
            return model

    def transcribe(self, audio: bytes, language: str | None = None) -> Transcript:
        model = self._load()
        lang = (language or get_settings().whisper_language or "").strip() or None
        samples = _decode_with_ffmpeg(audio)
        with self._run_lock:
            segments, info = model.transcribe(
                samples if samples is not None else io.BytesIO(audio),
                language=lang,
                beam_size=5,
                vad_filter=True,  # drop silence so a pause isn't "transcribed"
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(
            text=text,
            language=getattr(info, "language", "") or "",
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            model=self._key[0] if self._key else "",
            device=self._device,
        )


_transcriber = _Transcriber()


async def transcribe_audio(audio: bytes, language: str | None = None) -> Transcript:
    """Transcribe an audio clip off the event loop."""
    if not audio:
        raise ValueError("empty audio")
    return await asyncio.to_thread(_transcriber.transcribe, audio, language)


async def warm_up() -> None:
    """Load the model ahead of the first click (best effort)."""
    if not speech_available():
        return
    try:
        await asyncio.to_thread(_transcriber._load)
    except SpeechUnavailableError as e:
        log.warning("speech.warmup_failed", error=str(e)[:300])


__all__ = [
    "SpeechUnavailableError",
    "Transcript",
    "speech_available",
    "transcribe_audio",
    "warm_up",
]
