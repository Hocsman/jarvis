"""Recording him is opt-in, and it says so out loud when it is on.

The ASR bench needs his own audio: the failure it exists to measure only
shows up on his voice, at his distance, on his proper nouns. Reading a
script into the microphone would measure a different problem.

So the listener can keep a copy of each utterance. That is a microphone
writing his voice to disk, which is the most sensitive thing this
codebase can do, so the switch is an environment variable and nothing
else: no config key, no default, no UI toggle that could be left on by a
past session. A capture nobody asked for is a capture nobody knows about.

It also keeps the utterances that transcribed to nothing. Those are the
interesting ones — silence and garbage are where the two architectures
differ — and a corpus of only the successes would measure the easy half.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest


def _capture(monkeypatch, dossier=None):
    from src.jarvis.listening.audio_capture import UtteranceCapture

    if dossier is None:
        monkeypatch.delenv("JARVIS_SAVE_AUDIO", raising=False)
    else:
        monkeypatch.setenv("JARVIS_SAVE_AUDIO", str(dossier))
    return UtteranceCapture.from_env()


def _son(secondes=0.5, rate=16000):
    t = np.linspace(0, secondes, int(rate * secondes), endpoint=False)
    return (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_without_the_variable_it_records_nothing(tmp_path, monkeypatch):
    capture = _capture(monkeypatch, None)

    capture.save(_son(), 16000, "la météo à Bagneux", model="medium",
                 device="Micro MacBook Air")

    assert capture.enabled is False
    assert list(tmp_path.iterdir()) == []


def test_the_directory_is_not_even_created_when_it_is_off(tmp_path, monkeypatch):
    """An empty folder appearing in his home is already a claim that
    something was recorded."""
    cible = tmp_path / "jamais"
    monkeypatch.delenv("JARVIS_SAVE_AUDIO", raising=False)

    _capture(monkeypatch, None)

    assert not cible.exists()


def test_with_the_variable_the_audio_comes_back_as_it_went_in(tmp_path, monkeypatch):
    capture = _capture(monkeypatch, tmp_path)
    son = _son()

    capture.save(son, 16000, "la météo à Bagneux", model="medium",
                 device="Micro MacBook Air")

    wavs = sorted(tmp_path.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == 16000
        relu = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    # int16 round-trip, so compare on the shape and on the envelope rather
    # than bit-for-bit.
    assert relu.size == son.size
    assert abs(float(np.max(np.abs(relu))) / 32767 - float(np.max(np.abs(son)))) < 0.01


def test_the_sidecar_carries_what_whisper_made_of_it(tmp_path, monkeypatch):
    capture = _capture(monkeypatch, tmp_path)

    capture.save(_son(), 16000, "Théo sur Banyo", model="medium",
                 device="Micro MacBook Air")

    fiches = sorted(tmp_path.glob("*.json"))
    assert len(fiches) == 1
    fiche = json.loads(fiches[0].read_text(encoding="utf-8"))
    assert fiche["hypothesis"] == "Théo sur Banyo"
    assert fiche["model"] == "medium"
    assert fiche["device"] == "Micro MacBook Air"
    assert fiche["sample_rate"] == 16000
    assert fiche["reference"] == "", "la vérité terrain est à lui, laissée vide"


def test_an_utterance_that_transcribed_to_nothing_is_kept(tmp_path, monkeypatch):
    """Silence and garbage are where a transducer and a seq2seq differ, so
    dropping the empty ones would measure only the easy half."""
    capture = _capture(monkeypatch, tmp_path)

    capture.save(_son(), 16000, "", model="medium", device="x")

    assert len(list(tmp_path.glob("*.wav"))) == 1
    fiche = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert fiche["hypothesis"] == ""


def test_two_utterances_do_not_land_on_the_same_name(tmp_path, monkeypatch):
    capture = _capture(monkeypatch, tmp_path)

    for _ in range(3):
        capture.save(_son(), 16000, "x", model="m", device="d")

    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_a_disk_that_refuses_does_not_reach_the_voice_path(tmp_path, monkeypatch):
    """The listener's loop drains an audio queue on a deadline. A capture
    that raises would cost him the sentence it was supposed to record."""
    cible = tmp_path / "fichier"
    cible.write_text("je ne suis pas un dossier", encoding="utf-8")
    capture = _capture(monkeypatch, cible)

    capture.save(_son(), 16000, "x", model="m", device="d")  # must not raise
