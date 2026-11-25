"""She answers when his turn is over, read from his voice.

A fixed silence (`voice_collect_seconds`) would make a question he has
finished wait all of it, and send half a question when he pauses in the
middle of a sentence for longer. The end-of-turn model reads his voice when
an utterance joins the question, and its verdict sets that wait: none when
he has finished, at least `turn_unfinished_wait_seconds` and never less
than the fixed silence when he has not, and the fixed silence when it gives
no verdict. Her name alone is never a finished question.

The utterances below go through the whole path the microphone takes: the
voice detector closes them, Whisper (stubbed) reads them, and the wake
word, or the hot window, sends them to the question being collected.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.listening import turn_end
from test_hot_window_input import (
    _create_listener, _install_intent_judge, _lines_for_him, _make_judgment,
    _simulate_tts_finish, _starts_with_an_emoji, _wait_for_hot_window_active,
)

FIXED = 2.0
UNFINISHED = 3.0
SR = turn_end.SAMPLE_RATE


class _Ears:
    """An end-of-turn model, loaded, that gives the verdict it is told and
    keeps what it heard."""

    status, why = "ready", ""

    def __init__(self, probability):
        self.verdict = probability
        self.heard = []

    def probability(self, sound):
        self.heard.append(np.array(sound, copy=True))
        return self.verdict


def _listener(ears, fixed=FIXED, unfinished=UNFINISHED, **kwargs):
    listener, _tts = _create_listener(**kwargs)
    cfg = listener.cfg
    cfg.vad_frame_ms, cfg.vad_pre_roll_ms, cfg.endpoint_silence_ms = 20, 240, 400
    cfg.max_utterance_ms, cfg.tts_max_utterance_ms = 12000, 3000
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    cfg.voice_collect_seconds = fixed
    cfg.turn_detector_enabled = True
    cfg.turn_unfinished_wait_seconds = unfinished
    listener.state_manager.voice_collect_seconds = fixed
    listener._configure_frames()
    listener._whisper_backend, listener._whisper_device = "faster-whisper", "cuda"
    listener.model = MagicMock()
    listener._turn_end = ears
    handed = []
    listener._hand_over = handed.append
    return listener, handed


class _Mouth:
    """Says things into the listener, frame by frame, on the clock the
    microphone stamps them with."""

    def __init__(self, listener, now=None):
        self.listener = listener
        self.step = listener._frame_ms / 1000.0
        self.now = time.time() - 60.0 if now is None else now
        self.closed_at = None
        self._voiced = False
        listener._is_speech_frame = lambda frame: self._voiced

    def _frame(self, voiced):
        n = self.listener._frame_samples
        if not voiced:
            return np.zeros(n, dtype=np.float32)
        rate = getattr(self.listener, "_stream_samplerate", None) or self.listener._samplerate
        t = np.arange(n) / rate
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def _take(self, voiced):
        self._voiced = voiced
        self.listener._take_frame(self._frame(voiced), self.now)
        self.now += self.step

    def says(self, text, seconds=1.0, pause_inside=0.0):
        """*text*, spoken for *seconds*, with a pause shorter than the
        voice detector's in the middle when *pause_inside* is given. The
        utterance is closed by the silence after it."""
        self.listener.model.transcribe.return_value = (
            [SimpleNamespace(text=" " + text, avg_logprob=-0.1, no_speech_prob=0.01)],
            SimpleNamespace(language="fr"))
        half = int(seconds / 2 / self.step)
        for _ in range(half):
            self._take(True)
        for _ in range(int(pause_inside / self.step)):
            self._take(False)
        for _ in range(half):
            self._take(True)
        while self.listener.is_speech_active:
            self._take(False)
        self.closed_at = self.now

    def keeps_quiet(self, seconds):
        for _ in range(int(round(seconds / self.step))):
            self._take(False)


@pytest.fixture(autouse=True)
def _quiet():
    with patch("builtins.print") as mock_print:
        yield mock_print


def test_a_finished_question_goes_without_waiting_the_fixed_silence():
    listener, handed = _listener(_Ears(0.9))
    mouth = _Mouth(listener)

    mouth.says("jarvis quelle heure est-il")
    mouth.keeps_quiet(0.1)

    assert handed == ["quelle heure est-il"]
    listener.state_manager.stop()


def test_an_unfinished_sentence_is_waited_for_longer_than_the_fixed_silence(_quiet):
    listener, handed = _listener(_Ears(0.1))
    mouth = _Mouth(listener)

    mouth.says("jarvis est-ce que tu peux me dire")
    mouth.keeps_quiet(FIXED + 0.5)
    assert handed == [], "half a sentence was answered after the fixed silence"
    lines = _lines_for_him(_quiet)
    assert any(_starts_with_an_emoji(line) and f"{UNFINISHED:g}" in line for line in lines), lines

    mouth.keeps_quiet(UNFINISHED - FIXED)
    assert handed == ["est-ce que tu peux me dire"]
    listener.state_manager.stop()


def test_the_rest_of_an_unfinished_sentence_completes_the_question():
    ears = _Ears(0.1)
    listener, handed = _listener(ears)
    mouth = _Mouth(listener)

    mouth.says("jarvis est-ce que tu peux me dire")
    mouth.keeps_quiet(1.2)
    ears.verdict = 0.9
    # Closed past the fixed silence, before the unfinished wait ran out.
    mouth.says("la météo à Valbrume demain", seconds=0.8)
    mouth.keeps_quiet(0.1)

    assert handed == ["est-ce que tu peux me dire la météo à valbrume demain"]
    listener.state_manager.stop()


def test_with_no_verdict_the_fixed_silence_decides():
    listener, handed = _listener(_Ears(None))
    mouth = _Mouth(listener)

    mouth.says("jarvis quelle heure est-il")
    mouth.keeps_quiet(FIXED - 0.2)
    assert handed == []
    mouth.keeps_quiet(0.3)
    assert handed == ["quelle heure est-il"]
    listener.state_manager.stop()


def test_turned_off_the_model_is_not_asked_and_the_fixed_silence_decides():
    ears = _Ears(0.9)
    listener, handed = _listener(ears)
    listener.cfg.turn_detector_enabled = False
    mouth = _Mouth(listener)

    mouth.says("jarvis quelle heure est-il")
    mouth.keeps_quiet(FIXED - 0.2)

    assert handed == []
    assert ears.heard == []
    mouth.keeps_quiet(0.3)
    assert handed == ["quelle heure est-il"]
    listener.state_manager.stop()


def test_her_name_alone_is_never_a_finished_question():
    """"Jarvis ?" with a finished intonation still waits for his question:
    a finished verdict on her name alone would end the wait before he asks."""
    ears = _Ears(0.9)
    listener, handed = _listener(ears)
    mouth = _Mouth(listener)

    mouth.says("jarvis", seconds=0.6)
    mouth.keeps_quiet(0.1)
    assert listener.state_manager.is_collecting(), "she stopped waiting for his question"

    mouth.says("quelle heure est-il", seconds=1.0)
    mouth.keeps_quiet(0.1)
    assert handed == ["quelle heure est-il"]
    listener.state_manager.stop()


def test_her_name_alone_said_as_unfinished_is_waited_for_longer():
    listener, handed = _listener(_Ears(0.1))
    mouth = _Mouth(listener)

    mouth.says("jarvis", seconds=0.6)
    mouth.keeps_quiet(FIXED + 0.5)

    assert listener.state_manager.is_collecting(), "she stopped waiting for his question"
    listener.state_manager.stop()


def test_the_model_hears_the_whole_question_pauses_included():
    ears = _Ears(0.1)
    listener, _handed = _listener(ears)
    mouth = _Mouth(listener)

    mouth.says("jarvis est-ce que tu peux", seconds=1.0, pause_inside=0.3)
    first = ears.heard[-1]
    mouth.keeps_quiet(0.5)
    mouth.says("me dire la météo", seconds=1.0)
    whole = ears.heard[-1]

    # The pause inside the utterance is heard, though Whisper is not given
    # it, and so is the start of the silence that closed it.
    assert first.size / turn_end.SAMPLE_RATE >= 1.0 + 0.3 + turn_end.CLOSING_SILENCE_HEARD - 0.05
    assert whole.size > first.size
    assert np.array_equal(whole[:first.size], first)
    listener.state_manager.stop()


def test_the_model_hears_only_the_start_of_the_silence_that_closed_his_words():
    """The voice detector waits `endpoint_silence_ms` before closing an
    utterance; the model reads a pause better when it hears only the first
    `CLOSING_SILENCE_HEARD` of it."""
    ears = _Ears(0.1)
    listener, _handed = _listener(ears)
    listener.cfg.endpoint_silence_ms = 800
    listener._configure_frames()
    mouth = _Mouth(listener)

    mouth.says("jarvis est-ce que tu peux", seconds=1.0)

    heard = ears.heard[-1]
    voiced = np.flatnonzero(heard)
    trailing = (heard.size - 1 - voiced[-1]) / turn_end.SAMPLE_RATE
    assert trailing == pytest.approx(turn_end.CLOSING_SILENCE_HEARD, abs=0.03)
    listener.state_manager.stop()


def test_the_model_hears_no_more_than_its_window():
    ears = _Ears(0.1)
    listener, _handed = _listener(ears)
    listener.cfg.turn_unfinished_wait_seconds = 10.0
    mouth = _Mouth(listener)

    mouth.says("jarvis raconte moi comment on prépare une tarte aux poires pour six", seconds=6.0)
    mouth.says("avec une pâte brisée maison et un peu de cannelle sur le dessus", seconds=6.0)

    assert ears.heard[-1].size <= turn_end.WINDOW_SAMPLES
    listener.state_manager.stop()


def test_whisper_reads_the_same_audio_with_the_model_on_or_off():
    def whisper_input(enabled):
        listener, _handed = _listener(_Ears(0.9))
        listener.cfg.turn_detector_enabled = enabled
        mouth = _Mouth(listener)
        mouth.says("jarvis quelle heure est-il", seconds=1.0, pause_inside=0.3)
        listener.state_manager.stop()
        (call,) = listener.model.transcribe.call_args_list
        return call.args[0]

    assert np.array_equal(whisper_input(True), whisper_input(False))


def test_an_unfinished_sentence_never_waits_less_than_the_fixed_silence():
    """With a fixed silence longer than the unfinished wait, a sentence
    heard as unfinished keeps the longer of the two: the model never cuts a
    pause the fixed silence would have let through."""
    fixed, unfinished = 5.0, 3.0
    listener, handed = _listener(_Ears(0.1), fixed=fixed, unfinished=unfinished)
    mouth = _Mouth(listener)

    mouth.says("jarvis est-ce que tu peux me dire")
    mouth.keeps_quiet(fixed - 0.3)
    assert handed == [], "an unfinished sentence was cut before the fixed silence"

    mouth.keeps_quiet(0.5)
    assert handed == ["est-ce que tu peux me dire"]
    listener.state_manager.stop()


def test_her_name_alone_said_as_unfinished_never_waits_less_than_the_fixed_silence():
    fixed, unfinished = 5.0, 3.0
    listener, _handed = _listener(_Ears(0.1), fixed=fixed, unfinished=unfinished)
    mouth = _Mouth(listener)

    mouth.says("jarvis", seconds=0.6)
    mouth.keeps_quiet(fixed - 0.3)

    assert listener.state_manager.is_collecting(), "she stopped waiting for his question"
    listener.state_manager.stop()


def _in_the_hot_window(ears, judgment):
    listener, handed = _listener(ears, echo_tolerance=0.02, hot_window_seconds=3.0)
    listener.echo_detector.track_tts_start("Il fait beau à Valbrume aujourd'hui.")
    _simulate_tts_finish(listener)
    assert _wait_for_hot_window_active(listener)
    _install_intent_judge(listener, judgment)
    return listener, handed, _Mouth(listener, now=time.time())


def test_a_follow_up_in_the_hot_window_is_read_by_the_model():
    ears = _Ears(0.9)
    listener, handed, mouth = _in_the_hot_window(
        ears, _make_judgment(directed=True, query="et demain"))

    mouth.says("et demain", seconds=0.8)
    mouth.keeps_quiet(0.1)

    assert len(ears.heard) == 1
    assert handed == ["et demain"]
    listener.state_manager.stop()


def test_her_name_alone_in_the_hot_window_is_never_a_finished_question():
    """In the hot window the judge may keep her name as his query. Her name
    is taken out before the question is weighed, so a finished verdict on it
    still leaves him time to ask."""
    listener, handed, mouth = _in_the_hot_window(
        _Ears(0.9), _make_judgment(directed=True, query="Jarvis ?"))

    mouth.says("jarvis", seconds=0.6)
    mouth.keeps_quiet(0.1)

    assert handed == [], "her name alone was handed over as a finished question"
    listener.state_manager.stop()


def test_a_new_question_does_not_hear_the_one_before():
    ears = _Ears(0.9)
    listener, handed = _listener(ears)
    mouth = _Mouth(listener)

    mouth.says("jarvis quelle heure est-il", seconds=1.5)
    mouth.keeps_quiet(0.5)
    mouth.says("jarvis et demain", seconds=1.0)
    mouth.keeps_quiet(0.1)

    assert len(handed) == 2
    pre_roll = listener.cfg.vad_pre_roll_ms / 1000
    longest_alone = 1.0 + pre_roll + turn_end.CLOSING_SILENCE_HEARD + 0.05
    assert ears.heard[-1].size / SR <= longest_alone, "the second question heard the first"
    listener.state_manager.stop()


@pytest.mark.parametrize("capture_rate", [44100, 48000])
def test_at_another_capture_rate_the_model_hears_the_same_sound(capture_rate):
    """A microphone that refuses 16 kHz is read at its own rate; the model
    hears the utterance at its own rate, as long and with as much closing
    silence as at 16 kHz."""
    def heard_at(rate):
        ears = _Ears(0.1)
        listener, _handed = _listener(ears)
        listener._stream_samplerate = rate
        listener._frame_samples = int(rate * listener._frame_ms / 1000)
        _Mouth(listener).says("jarvis est-ce que tu peux", seconds=1.0)
        listener.state_manager.stop()
        return ears.heard[-1]

    reference, heard = heard_at(16000), heard_at(capture_rate)

    frame_seconds = 0.02
    assert abs(heard.size - reference.size) / SR <= frame_seconds
    trailing = (heard.size - 1 - np.flatnonzero(np.abs(heard) > 1e-3)[-1]) / SR
    assert trailing == pytest.approx(turn_end.CLOSING_SILENCE_HEARD, abs=frame_seconds + 0.01)


# ── What the console says at start-up ─────────────────────────────────


class _Loaded:
    def __init__(self, status, why=""):
        self.status, self.why = status, why


@pytest.mark.parametrize("status", ["ready", "loading", "unavailable"])
def test_the_warm_up_says_how_his_turn_is_read(_quiet, status):
    listener, _handed = _listener(_Loaded(status, why="could not download it"))

    listener._say_how_his_turn_is_read()

    lines = _lines_for_him(_quiet)
    assert len(lines) == 1 and _starts_with_an_emoji(lines[0]), lines
    if status == "unavailable":
        assert "could not download it" in lines[0]
        assert f"{FIXED:g}s" in lines[0]
    listener.state_manager.stop()


@pytest.mark.parametrize("settles_as", ["ready", "unavailable"])
def test_a_model_that_settles_after_the_warm_up_says_so(_quiet, settles_as):
    """The warm-up said it was still loading: once loading ends, one more
    line says how it ended, so a first start without network is not left at
    "still loading"."""
    loaded = _Loaded("loading", why="could not download it")
    listener, _handed = _listener(loaded)
    listener._say_how_his_turn_is_read()

    loaded.status = settles_as
    listener._turn_end_settled(loaded)

    lines = _lines_for_him(_quiet)
    assert len(lines) == 2 and all(_starts_with_an_emoji(line) for line in lines), lines
    assert lines[1] != lines[0]
    listener.state_manager.stop()


def test_a_model_that_settles_before_the_warm_up_line_is_said_once(_quiet):
    loaded = _Loaded("ready")
    listener, _handed = _listener(loaded)

    listener._turn_end_settled(loaded)
    listener._say_how_his_turn_is_read()

    assert len(_lines_for_him(_quiet)) == 1
    listener.state_manager.stop()


def test_turned_off_the_warm_up_says_nothing_of_it(_quiet):
    listener, _handed = _listener(_Loaded("idle"))
    listener.cfg.turn_detector_enabled = False

    listener._say_how_his_turn_is_read()

    assert _lines_for_him(_quiet) == []
    listener.state_manager.stop()
