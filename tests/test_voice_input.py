"""🎤 voice input: the transcribe endpoint and the local Whisper wrapper.

The real model is never loaded here — a fake stands in, so the suite stays
offline and fast.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from ai_employee.core import speech


class _FakeModel:
    def __init__(self, text: str = " Research the best   tracker. ", fail: bool = False):
        self.text, self.fail, self.calls = text, fail, []

    def transcribe(self, audio, **kw):
        self.calls.append((audio.read() if isinstance(audio, io.BytesIO) else audio, kw))
        if self.fail:
            raise ValueError("Invalid data found when processing input")
        segs = [SimpleNamespace(text=p) for p in self.text.split("|")]
        return iter(segs), SimpleNamespace(language="en", duration=2.5)


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    t = speech._transcriber
    monkeypatch.setattr(t, "_load", lambda: model)
    monkeypatch.setattr(t, "_key", ("small", "cpu", "int8"))
    monkeypatch.setattr(t, "_device", "cpu")
    monkeypatch.setattr(speech, "_decode_with_ffmpeg", lambda audio: None)
    return model


async def test_transcribe_joins_segments_and_passes_options(fake_model):
    fake_model.text = " Research the best | tracker. "
    out = await speech.transcribe_audio(b"RIFFdata", language="en")
    assert out.text == "Research the best tracker."
    assert out.language == "en" and out.model == "small" and out.device == "cpu"
    audio, kw = fake_model.calls[0]
    assert audio == b"RIFFdata"
    assert kw["language"] == "en" and kw["vad_filter"] is True


async def test_empty_audio_rejected():
    with pytest.raises(ValueError):
        await speech.transcribe_audio(b"")


async def test_endpoint_returns_text(client, fake_model):
    r = await client.post(
        "/ui/api/transcribe", content=b"\x1aE\xdf\xa3webm", headers={"Content-Type": "audio/webm"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Research the best   tracker."
    assert body["model"] == "small" and body["duration"] == 2.5


async def test_endpoint_language_query(client, fake_model):
    r = await client.post("/ui/api/transcribe?language=hi", content=b"x")
    assert r.status_code == 200
    assert fake_model.calls[0][1]["language"] == "hi"


async def test_endpoint_rejects_empty_and_huge(client, fake_model, monkeypatch):
    assert (await client.post("/ui/api/transcribe", content=b"")).status_code == 400
    from ai_employee.api import ui

    monkeypatch.setattr(ui, "_MAX_AUDIO_BYTES", 10)
    assert (await client.post("/ui/api/transcribe", content=b"x" * 11)).status_code == 413


async def test_endpoint_bad_audio_is_422(client, fake_model):
    fake_model.fail = True
    r = await client.post("/ui/api/transcribe", content=b"not audio")
    assert r.status_code == 422
    assert "Could not transcribe" in r.json()["detail"]


async def test_endpoint_unavailable_is_503(client, monkeypatch):
    def boom():
        raise speech.SpeechUnavailableError("Voice input needs faster-whisper")

    monkeypatch.setattr(speech._transcriber, "_load", boom)
    r = await client.post("/ui/api/transcribe", content=b"x")
    assert r.status_code == 503
    assert "faster-whisper" in r.json()["detail"]


async def test_status_endpoint(client):
    r = await client.get("/ui/api/transcribe/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"available", "model", "device", "language"}


def test_cuda_failure_falls_back_to_cpu(monkeypatch):
    pytest.importorskip("faster_whisper")
    import faster_whisper

    built = []

    class Model:
        def __init__(self, name, device, compute_type):
            built.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("libcublas.so.12 not found")

    monkeypatch.setattr(faster_whisper, "WhisperModel", Model)
    monkeypatch.setattr(speech, "_cuda_ok", lambda: True)
    t = speech._Transcriber()
    t._load()
    assert built == [("cuda", "float16"), ("cpu", "int8")]
    assert t._device == "cpu"


def test_ffmpeg_decode_to_16k_mono(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(wav),
        ],
        check=True,
    )
    samples = speech._decode_with_ffmpeg(wav.read_bytes())
    assert samples is not None and samples.dtype.name == "float32"
    assert abs(len(samples) - 16000) < 400  # 1 s resampled to 16 kHz mono
    with pytest.raises(ValueError):
        speech._decode_with_ffmpeg(b"definitely not audio")


def test_ffmpeg_missing_falls_back(monkeypatch):
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    assert speech._decode_with_ffmpeg(b"x") is None
