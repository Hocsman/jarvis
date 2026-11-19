"""Noise never becomes his city, on the real Whisper.

The listener gives Whisper his names as a hint (listening.spec.md, "The
names he uses"). Over noise Whisper writes the hint back: measured with
large-v3-turbo and a town for a hint, 162 noise clips in 219 came out as
the town and cleared both confidence filters. The listener reads a
transcript that is mostly the hint again without it. This runs that path
on the real model and holds it to the promise: the hint may change how a
word is spelled, never whether anything was said.

It runs with Whisper's voice detector and without it. Without it (the
setting off, or a detector that failed) the second reading is all that
stands between noise and his city. With it, noise that holds no voice
comes back empty whatever the hint, and what the detector lets through
is voices: people across the room who speak to nobody. Measured, such
clips came out as the town alone in 16 of 150 before the second reading.
So the clips are of both kinds: noise and silence, and several voices
talking at once away from the microphone.

The city is an invented name, and there are two of them, because how
readily Whisper writes a hint back depends on the name: on this file's
clips, before the second reading, noise came out as the name in 12 clips
of 18 for one and 18 for the other with the detector off, and voices
across the room in none of 30 for one and 7 for the other with it on.

Needs a faster-whisper model already on this machine, and for the voices
at least two Piper voices; nothing is downloaded.
``JARVIS_EVAL_WHISPER_MODEL`` names the model to use, else the first
cached of large-v3-turbo, small, base, tiny.
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
CITIES = ["Varnelle", "Ostrevane"]


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
def whisper():
    model = _load_model()
    if model is None:
        pytest.skip("no faster-whisper model cached on this machine")
    return model


@pytest.fixture(
    params=[(detector, city) for detector in (True, False) for city in CITIES],
    ids=lambda p: f"voice detector {'on' if p[0] else 'off'}, {p[1]}",
)
def listener(whisper, request):
    from jarvis.listening.listener import VoiceListener

    model = whisper
    detector, city = request.param
    heard = VoiceListener.__new__(VoiceListener)
    heard.cfg = SimpleNamespace(
        weather_city=city, whisper_vocabulary=[], wake_word="yuba",
        wake_aliases=["youba", "juba"], whisper_min_confidence=0.3,
        whisper_no_speech_threshold=0.5, whisper_language=None,
        whisper_vad=detector, voice_debug=False,
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


def _names_his_city(listener, text):
    return re.search(r"\b" + _fold(listener.cfg.weather_city) + r"\b", _fold(text)) is not None


@pytest.mark.parametrize("kind,level", CLIPS, ids=[f"{k}{l}dB" for k, l in CLIPS])
def test_noise_does_not_come_out_as_his_city(listener, kind, level):
    rng = np.random.default_rng(CLIPS.index((kind, level)))
    audio = _at(_noise(kind, rng), level)

    text, _language, _pieces = listener._transcribe(audio)

    assert not _names_his_city(listener, text)


# Voices in the room that speak to nobody: what the voice detector lets
# through. Sentences any household says; the names are no one's.
ROOM_TALK = [
    "Tu peux baisser un peu le son s'il te plaît ?",
    "Je crois que le four est encore allumé.",
    "On mange quoi ce soir, des pâtes ou du riz ?",
    "Elle a oublié ses clés sur la table du salon.",
    "Le match commence à vingt et une heures.",
    "J'ai encore trois courriels à envoyer avant midi.",
    "Tu as pensé à sortir les poubelles ?",
    "Il faut racheter du café et du lait.",
    "Attends, je finis ça et j'arrive.",
    "Je vais prendre une douche et je reviens.",
    "Ce film était vraiment trop long.",
    "Il pleut depuis ce matin sans arrêt.",
]
FAR_VOICE_CLIPS = 30
_MAX_VOICES = 4


def _load_voices():
    """The Piper voices already on this machine, at most four."""
    try:
        from jarvis.output.tts import _get_piper_models_dir
        from piper.voice import PiperVoice
    except Exception:
        return []
    voices = []
    for path in sorted(_get_piper_models_dir().glob("*.onnx"))[:_MAX_VOICES]:
        try:
            voices.append(PiperVoice.load(str(path), str(path) + ".json"))
        except Exception:
            continue
    return voices


def _spoken(voice, sentence):
    """*sentence* in *voice*, at the listener's sample rate."""
    audio = np.concatenate([chunk.audio_int16_array for chunk in voice.synthesize(sentence)])
    audio = audio.astype(np.float32) / 32768.0
    rate = voice.config.sample_rate
    at = np.arange(int(len(audio) * SR / rate)) * rate / SR
    return np.interp(at, np.arange(len(audio)), audio).astype(np.float32)


@pytest.fixture(scope="module")
def room_talk():
    voices = _load_voices()
    if len(voices) < 2:
        pytest.skip("fewer than two Piper voices cached on this machine")
    return [[_at(_spoken(voice, sentence), -26) for sentence in ROOM_TALK] for voice in voices]


def _through_the_room(x, rng, decay=0.7):
    """*x* as a microphone across the room hears it."""
    n = int(SR * decay)
    echo = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n) * 0.25
    echo[0] = 1.0
    size = len(x) + n
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(echo, size), size)[:len(x)]


def _far_voices(room_talk, rng):
    """Several people talking at once, away from the microphone."""
    n = int(SR * rng.uniform(2.0, 4.5))
    mix = np.zeros(n, dtype=np.float32)
    for voice in rng.permutation(len(room_talk))[:3]:
        stream = []
        while sum(len(part) for part in stream) < n + SR:
            stream.append(room_talk[voice][rng.integers(len(ROOM_TALK))])
            stream.append(np.zeros(int(SR * rng.uniform(0.1, 0.4)), dtype=np.float32))
        stream = np.concatenate(stream)
        start = rng.integers(0, len(stream) - n)
        mix += stream[start:start + n]
    return _at(_through_the_room(mix, rng), rng.uniform(-42, -34)) + _at(_pink(n, rng), -52)


@pytest.mark.parametrize("clip", range(FAR_VOICE_CLIPS))
def test_voices_across_the_room_do_not_come_out_as_his_city(listener, room_talk, clip):
    audio = _far_voices(room_talk, np.random.default_rng(1000 + clip))

    text, _language, _pieces = listener._transcribe(audio)

    assert not _names_his_city(listener, text)


def test_silence_does_not_come_out_as_his_city(listener):
    text, _language, _pieces = listener._transcribe(np.zeros(2 * SR, dtype=np.float32))

    assert not _names_his_city(listener, text)
