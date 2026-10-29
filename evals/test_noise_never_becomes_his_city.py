"""Noise never becomes his city, on the real Whisper.

The listener gives Whisper his names as a hint (listening.spec.md, "The
names he uses"). Over noise Whisper writes the hint back: measured with
large-v3-turbo, 162 noise clips in 219 came out "Bagneux" and cleared
both confidence filters. The listener reads a transcript that is mostly
the hint again without it. This runs that path on the real model over
noise and silence and holds it to the promise: the hint may change how
a word is spelled, never whether anything was said.

Needs a faster-whisper model already on this machine; nothing is
downloaded. ``JARVIS_EVAL_WHISPER_MODEL`` names the one to use, else
the first cached of large-v3-turbo, small, base, tiny.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from types import SimpleNamespace

import numpy as np
import pytest

SR = 16000
CITY = "Bagneux"


def _load_model():
    try:
        from jarvis.listening.listener import _setup_nvidia_dll_path
        from faster_whisper import WhisperModel
    except Exception:
        return None
    _setup_nvidia_dll_path()
    wanted = os.environ.get("JARVIS_EVAL_WHISPER_MODEL")
    for name in ([wanted] if wanted else ["large-v3-turbo", "small", "base", "tiny"]):
        try:
            return WhisperModel(name, device="auto", compute_type="int8", local_files_only=True)
        except Exception:
            continue
    return None


@pytest.fixture(scope="module")
def listener():
    model = _load_model()
    if model is None:
        pytest.skip("no faster-whisper model cached on this machine")
    from jarvis.listening.listener import VoiceListener

    heard = VoiceListener.__new__(VoiceListener)
    heard.cfg = SimpleNamespace(
        weather_city=CITY, whisper_vocabulary=[], wake_word="yuba",
        wake_aliases=["youba", "juba"], whisper_min_confidence=0.3,
        whisper_no_speech_threshold=0.5, voice_debug=False,
    )
    heard.model = model
    heard._whisper_backend = "faster-whisper"
    heard._whisper_device = str(model.model.device)
    heard.transcribe_lock = threading.Lock()
    heard._last_detected_language = None
    return heard


def _at(x, dbfs):
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 10 ** (dbfs / 20)).astype(np.float32)


def _pink(n, rng):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    k = np.arange(len(spectrum))
    k[0] = 1
    return np.fft.irfft(spectrum / np.sqrt(k), n)


def _noise(kind, rng, seconds=2.5):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if kind == "white":
        return rng.standard_normal(n)
    if kind == "pink":
        return _pink(n, rng)
    if kind == "brown":
        walk = np.cumsum(rng.standard_normal(n))
        return walk - np.convolve(walk, np.ones(801) / 801, mode="same")
    if kind == "hum":
        return sum(np.sin(2 * np.pi * 50 * h * t) / h for h in (1, 2, 3, 5)) + 0.05 * rng.standard_normal(n)
    if kind == "clicks":
        x = 0.01 * rng.standard_normal(n)
        for at in rng.integers(0, n - 400, size=15):
            x[at:at + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
        return x
    if kind == "music":
        x = np.zeros(n)
        step = int(0.4 * SR)
        for i in range(0, n, step):
            f = (220, 261.6, 329.6, 392, 440)[rng.integers(5)]
            s = np.arange(min(step, n - i)) / SR
            x[i:i + len(s)] += np.sin(2 * np.pi * f * s) * np.exp(-s * 3)
        return x
    raise ValueError(kind)


CLIPS = [
    (kind, level)
    for kind in ("white", "pink", "brown", "hum", "clicks", "music")
    for level in (-45, -35, -25)
]


def _fold(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _names_the_city(text):
    return re.search(r"\b" + _fold(CITY) + r"\b", _fold(text)) is not None


@pytest.mark.parametrize("kind,level", CLIPS, ids=[f"{k}{l}dB" for k, l in CLIPS])
def test_noise_does_not_come_out_as_his_city(listener, kind, level):
    rng = np.random.default_rng(CLIPS.index((kind, level)))
    audio = _at(_noise(kind, rng), level)

    assert not _names_the_city(listener._transcribe(audio))


def test_silence_does_not_come_out_as_his_city(listener):
    assert not _names_the_city(listener._transcribe(np.zeros(2 * SR, dtype=np.float32)))
