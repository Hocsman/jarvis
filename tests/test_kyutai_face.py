"""Kyutai feeding a talking face (`src/jarvis/output/tts.spec.md`, "Kyutai",
"The face"; `src/jarvis/output/face/face.spec.md`).

The model and the speakers are the stand-ins of ``test_kyutai_tts.py``; the
face is a stand-in that records what it is handed. What is pinned: an item is
drawn on the face only while the face is ready and watched, its audio reaches
the face stamped with when it will be heard, the voice waits for that moment,
the sentences of a reply keep their spacing, and a stop cuts the face too.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from jarvis.output import tts as tts_module
from test_kyutai_tts import SAMPLE_RATE, SAMPLES_PER_FRAME, _speak, _until, make_engine, stage  # noqa: F401

DELAY = 0.3


class FakeFace:
    def __init__(self, ready=True, watched=True):
        self._ready = ready
        self._watched = watched
        self.delay = DELAY
        self.sent = []    # (heard_at, samples, rate)
        self.cuts = []

    def ready(self):
        return self._ready

    def watched(self):
        return self._ready and self._watched

    def send_audio(self, heard_at, samples, rate):
        self.sent.append((heard_at, len(samples), rate))

    def cut(self, at):
        self.cuts.append(at)


def _leading_silence(stage):
    audio = np.concatenate(stage.streams[-1].audio)
    return int(np.argmax(audio != 0)), audio


class TestNoFaceToFollow:
    @pytest.mark.parametrize("face", [None, FakeFace(watched=False), FakeFace(ready=False)])
    def test_she_speaks_at_once_and_nothing_is_handed_over(self, stage, make_engine, face):
        tts = make_engine(face=face)

        heard, _ = _speak(tts, "Bonjour.")

        assert heard
        assert _leading_silence(stage)[0] == 0
        assert tts.voice_delay() == 0.0
        assert face is None or face.sent == []


class TestDrawnOnTheFace:
    def test_her_audio_reaches_the_face_stamped_with_when_it_is_heard(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        before = time.perf_counter()

        heard, _ = _speak(tts, "Bonjour.")

        assert heard
        total = sum(n for _at, n, _rate in face.sent)
        assert total == stage.frames * SAMPLES_PER_FRAME
        assert {rate for _at, _n, rate in face.sent} == {SAMPLE_RATE}
        first = face.sent[0][0]
        assert first >= before + DELAY
        offsets = np.cumsum([0] + [n for _at, n, _rate in face.sent[:-1]]) / SAMPLE_RATE
        assert np.allclose([at for at, _n, _rate in face.sent], first + offsets)

    def test_the_voice_waits_for_the_moment_the_face_was_given(self, stage, make_engine):
        tts = make_engine(face=FakeFace())
        assert tts._ensure_initialized()

        heard, _ = _speak(tts, "Bonjour.")

        silence, audio = _leading_silence(stage)
        assert heard
        assert 0 < silence <= DELAY * SAMPLE_RATE
        assert np.count_nonzero(audio) == stage.frames * SAMPLES_PER_FRAME

    def test_the_first_sound_is_marked_when_she_is_heard_not_when_the_stream_opens(
            self, stage, make_engine, monkeypatch):
        face = FakeFace()
        marks = []
        monkeypatch.setattr(tts_module, "_sound_started", lambda: marks.append(time.perf_counter()))
        tts = make_engine(face=face)
        assert tts._ensure_initialized()

        heard, _ = _speak(tts, "Bonjour.")

        assert heard
        assert marks and marks[0] >= face.sent[0][0] - 0.01

    def test_the_sentences_of_a_reply_keep_their_spacing(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        done = threading.Event()

        tts.speak("Première.")
        tts.speak("Deuxième.", completion_callback=done.set)

        assert done.wait(10)
        per_item = stage.frames * SAMPLES_PER_FRAME
        first_start, second_start = face.sent[0][0], None
        counted = 0
        for at, n, _rate in face.sent:
            if counted == per_item:
                second_start = at
                break
            counted += n
        assert second_start == pytest.approx(first_start + per_item / SAMPLE_RATE + tts._FACE_GAP_SECONDS)

    def test_the_voice_delay_is_the_faces_while_it_is_watched(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)

        assert tts.voice_delay() == DELAY
        face._watched = False
        assert tts.voice_delay() == 0.0


class TestStopping:
    def test_a_stop_cuts_the_face_at_that_instant(self, stage, make_engine):
        face = FakeFace()
        stage.hold_playback = True
        tts = make_engine(face=face)
        tts.speak("Une longue phrase.")
        assert _until(lambda: stage.streams)

        before = time.perf_counter()
        tts.interrupt()

        assert len(face.cuts) == 1 and before <= face.cuts[0] <= time.perf_counter()


class TestWiring:
    def test_the_factory_hands_the_face_to_kyutai_only(self, stage):
        face = FakeFace()

        kyutai = tts_module.create_tts_engine(engine="kyutai", face=face)
        piper = tts_module.create_tts_engine(engine="piper", face=face)

        assert kyutai.voice_delay() == DELAY
        assert piper.voice_delay() == 0.0
