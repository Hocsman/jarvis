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
        assert second_start == pytest.approx(first_start + per_item / SAMPLE_RATE + tts.face_gap())

    def test_an_item_shorter_than_the_first_stretch_is_drawn_too(self, stage, make_engine):
        face = FakeFace()
        stage.frames = 2   # under the quarter second a stream waits for
        tts = make_engine(face=face)

        heard, _ = _speak(tts, "Oui.")

        assert heard
        assert sum(n for _at, n, _rate in face.sent) == 2 * SAMPLES_PER_FRAME

    def test_pieces_handed_over_after_the_voice_waited_are_stamped_later_by_as_much(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        stage.frames = 30
        stage.gate_after = 8
        stage.mid_gate = threading.Event()
        done = threading.Event()

        tts.speak("Une phrase qui attend.", completion_callback=done.set)
        assert _until(lambda: len(face.sent) >= 8)
        time.sleep(0.2)             # the stream runs out and waits for the synthesis
        stage.mid_gate.set()
        assert done.wait(10)

        first = face.sent[0][0]
        offsets = np.cumsum([0] + [n for _at, n, _rate in face.sent[:-1]]) / SAMPLE_RATE
        drift = np.array([at for at, _n, _rate in face.sent]) - (first + offsets)
        assert np.allclose(drift[:8], 0.0)
        assert (drift[8:] > 0.05).all()

    def test_the_voice_delay_is_the_faces_while_it_is_watched(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)

        assert tts.voice_delay() == DELAY
        face._watched = False
        assert tts.voice_delay() == 0.0


class TestFollowingTheVoice:
    def test_no_item_is_drawn_while_one_played_without_the_face_is_ahead(self, stage, make_engine):
        face = FakeFace(watched=False)
        stage.hold_playback = True
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        first, second, third = threading.Event(), threading.Event(), threading.Event()

        tts.speak("Avant le regard.", completion_callback=first.set)
        assert _until(lambda: stage.streams)
        face._watched = True
        tts.speak("Pendant.", completion_callback=second.set)
        assert _until(lambda: tts._undrawn == 2)
        stage.hold_playback = False
        assert first.wait(10) and second.wait(10)

        assert face.sent == []
        tts.speak("Apres.", completion_callback=third.set)
        assert third.wait(10)
        assert sum(n for _at, n, _rate in face.sent) == stage.frames * SAMPLES_PER_FRAME

    def test_a_face_nobody_watches_any_more_is_not_drawn_from_the_next_item(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        first, second = threading.Event(), threading.Event()

        tts.speak("Regardee.", completion_callback=first.set)
        assert _until(lambda: face.sent)
        face._watched = False
        tts.speak("Plus regardee.", completion_callback=second.set)
        assert first.wait(10) and second.wait(10)

        assert sum(n for _at, n, _rate in face.sent) == stage.frames * SAMPLES_PER_FRAME
        assert _leading_silence(stage)[0] == 0

    def test_after_a_synthesis_that_failed_midway_the_next_item_follows_what_was_said(self, stage, make_engine):
        face = FakeFace()
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        stage.fail_after = 6
        stage.fail_midway = RuntimeError("boom")

        tts.speak("Interrompue par une panne.")
        assert _until(lambda: face.sent)
        assert _until(lambda: not tts.is_speaking())
        stage.fail_midway = None
        done = threading.Event()
        tts.speak("Suivante.", completion_callback=done.set)
        assert done.wait(10)

        # The second item's first piece is where the stamps jump by more than the voice can wait.
        pieces = face.sent
        split = next(i for i in range(1, len(pieces))
                     if pieces[i][0] > pieces[i - 1][0] + pieces[i - 1][1] / SAMPLE_RATE + 0.15)
        first_start, second_start = pieces[0][0], pieces[split][0]
        said = sum(n for _at, n, _rate in pieces[:split]) / SAMPLE_RATE
        assert said < stage.seconds   # the synthesis did stop midway
        assert second_start >= first_start + said + tts.face_gap() - 1e-6
        assert second_start < first_start + stage.seconds


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

    def test_the_reply_after_a_stop_does_not_wait_for_the_one_it_cut(self, stage, make_engine):
        face = FakeFace()
        stage.hold_playback = True
        tts = make_engine(face=face)
        assert tts._ensure_initialized()
        tts.speak("Une longue phrase coupee.")
        assert _until(lambda: face.sent)
        cut_one = face.sent[0][0]
        tts.interrupt()
        stage.hold_playback = False
        face.sent.clear()
        done = threading.Event()

        asked = time.perf_counter()
        tts.speak("La suivante.", completion_callback=done.set)
        assert done.wait(10)

        assert face.sent[0][0] < cut_one + stage.seconds
        assert face.sent[0][0] >= asked + DELAY


class TestWiring:
    def test_the_factory_hands_the_face_to_kyutai_only(self, stage):
        face = FakeFace()

        kyutai = tts_module.create_tts_engine(engine="kyutai", face=face)
        piper = tts_module.create_tts_engine(engine="piper", face=face)

        assert kyutai.voice_delay() == DELAY
        assert piper.voice_delay() == 0.0


class TestTheGapBetweenSentences:
    def test_it_covers_the_sound_cards_latency_and_what_a_late_start_taught_it(self, stage, make_engine):
        tts = make_engine(face=FakeFace())
        assert tts.face_gap() == pytest.approx(tts._FACE_GAP_SECONDS)

        tts._output_latency = 0.25
        assert tts.face_gap() == pytest.approx(tts._FACE_RESTART_SECONDS + 0.25)

        tts._face_gap_learnt = 0.1
        assert tts.face_gap() == pytest.approx(tts._FACE_RESTART_SECONDS + 0.25 + 0.1)

