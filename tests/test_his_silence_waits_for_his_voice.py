"""His silence is not over while he speaks.

The question being collected goes to the reply thread once his silence has
lasted `voice_collect_seconds`. An utterance he began before then and is
still saying when it runs out is waited for: once it closes it completes
the question, or is left out as any addition is, and only then does the
question go. Only while the echo canceller cleans the microphone: without
it the thinking tune reaches the voice detector, and would hold nearly
every question.

The utterances below go through the whole path the microphone takes, frame
by frame on the clock the microphone stamps them with: the voice detector
opens and closes them, Whisper (stubbed) reads them, and the wake word
starts the question.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_hot_window_input import (
    _WebRtcStandIn, _create_listener, _install_intent_judge, _make_judgment,
)
from jarvis.utils.echo_cancellation import EchoCanceller

COLLECT = 1.0
ENDPOINT = 0.4
QUESTION = "quelle heure est-il"
REST = "à l'autre bout du monde"


@pytest.fixture
def canceller():
    """The echo canceller the listener reads, on a stand-in processor,
    cleaning a capture of one channel at 16 kHz."""
    canceller = EchoCanceller(processor_factory=_WebRtcStandIn)
    assert canceller.start(16000) is None
    canceller.capture_opened(16000, 1)
    with patch("jarvis.listening.listener.ECHO_CANCELLER", canceller):
        yield canceller
    canceller.close()


@pytest.fixture
def no_canceller():
    """The echo canceller off, as with `voice_echo_cancellation` off or
    after it failed."""
    canceller = EchoCanceller(processor_factory=_WebRtcStandIn)
    with patch("jarvis.listening.listener.ECHO_CANCELLER", canceller):
        yield canceller
    canceller.close()


@pytest.fixture(autouse=True)
def _quiet():
    with patch("builtins.print") as mock_print:
        yield mock_print


def _listener(max_collect=60.0, max_utterance_ms=12000):
    listener, _tts = _create_listener()
    cfg = listener.cfg
    cfg.vad_frame_ms, cfg.vad_pre_roll_ms = 20, 240
    cfg.endpoint_silence_ms = int(ENDPOINT * 1000)
    cfg.max_utterance_ms, cfg.tts_max_utterance_ms = max_utterance_ms, 3000
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    cfg.voice_collect_seconds = COLLECT
    listener.state_manager.voice_collect_seconds = COLLECT
    listener.state_manager.max_collect_seconds = max_collect
    listener._configure_frames()
    listener._whisper_backend, listener._whisper_device = "faster-whisper", "cuda"
    listener.model = MagicMock()
    return listener


def _handed(listener) -> list:
    """The questions handed to the reply thread, oldest first."""
    with listener._turns.mutex:
        return [turn.query for turn in listener._turns.queue]


class _Mouth:
    """Says things into the listener, frame by frame, on the clock the
    microphone stamps them with."""

    def __init__(self, listener):
        self.listener = listener
        self.step = listener._frame_ms / 1000.0
        self.now = time.time() - 120.0
        self.closed_at = None
        self._voiced = False
        listener._is_speech_frame = lambda frame: self._voiced

    def _frame(self, voiced):
        n = self.listener._frame_samples
        if not voiced:
            return np.zeros(n, dtype=np.float32)
        t = np.arange(n) / self.listener._samplerate
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def _take(self, voiced):
        self._voiced = voiced
        self.listener._take_frame(self._frame(voiced), self.now)
        self.now += self.step

    def will_be_heard_as(self, text):
        """What Whisper makes of the next utterance it reads; nothing for
        an empty *text*."""
        segments = ([SimpleNamespace(text=" " + text, avg_logprob=-0.1, no_speech_prob=0.01)]
                    if text else [])
        self.listener.model.transcribe.return_value = (segments, SimpleNamespace(language="fr"))

    def starts_saying(self, text, seconds):
        """The first *seconds* of an utterance heard as *text*, left open."""
        self.will_be_heard_as(text)
        for _ in range(int(round(seconds / self.step))):
            self._take(True)

    def goes_on(self, seconds):
        for _ in range(int(round(seconds / self.step))):
            self._take(True)

    def stops(self):
        """Silence until the voice detector closes the utterance."""
        while self.listener.is_speech_active:
            self._take(False)
        self.closed_at = self.now

    def says(self, text, seconds=1.0):
        self.starts_saying(text, seconds)
        self.stops()

    def keeps_quiet(self, seconds):
        for _ in range(int(round(seconds / self.step))):
            self._take(False)

    def until_his_silence_runs_out(self, then_left=0.0):
        """Quiet until *then_left* seconds before his silence after the
        last utterance runs out. It is counted from the frame that closed
        that utterance."""
        closing_frame = self.closed_at - self.step
        self.keeps_quiet(closing_frame + COLLECT - then_left - self.now)


def _stop(listener):
    listener.state_manager.stop()


def test_what_he_is_still_saying_when_his_silence_runs_out_completes_the_question(canceller):
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.3)
    mouth.starts_saying(REST, seconds=1.5)
    assert _handed(listener) == [], "the question went while he was saying the rest of it"

    mouth.stops()
    mouth.keeps_quiet(COLLECT + 0.1)

    assert _handed(listener) == [f"{QUESTION} {REST}"]
    _stop(listener)


def test_his_question_begun_just_before_the_silence_after_her_name_runs_out_is_heard(canceller):
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis", seconds=0.4)
    mouth.until_his_silence_runs_out(then_left=0.2)
    mouth.says(QUESTION, seconds=1.2)
    mouth.keeps_quiet(COLLECT + 0.1)

    assert _handed(listener) == [QUESTION]
    _stop(listener)


def test_a_sound_heard_as_nothing_lets_the_question_go_the_moment_it_closes(canceller):
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.2)
    mouth.starts_saying("", seconds=0.8)
    assert _handed(listener) == []

    mouth.stops()

    assert _handed(listener) == [QUESTION], "the question waited past the sound that held it"
    _stop(listener)


def test_a_sound_the_judge_calls_noise_is_left_out_of_the_question(canceller):
    listener = _listener()
    judge = _install_intent_judge(listener, None)
    judge.judge.side_effect = [
        _make_judgment(directed=True, query=QUESTION),
        _make_judgment(directed=False, noise=True),
    ]
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.2)
    mouth.says("sous-titres de la communauté", seconds=0.8)

    assert _handed(listener) == [QUESTION]
    assert judge.judge.call_count == 2, "the sound was not judged before the question went"
    _stop(listener)


def test_an_utterance_begun_once_his_silence_has_run_out_holds_nothing(canceller):
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out()
    mouth.keeps_quiet(0.04)
    mouth.starts_saying(REST, seconds=0.2)

    assert _handed(listener) == [QUESTION]
    mouth.stops()
    _stop(listener)


def test_without_the_echo_canceller_his_silence_ends_as_before(no_canceller):
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.3)
    mouth.starts_saying(REST, seconds=1.0)

    assert _handed(listener) == [QUESTION]
    mouth.stops()
    _stop(listener)


def test_a_capture_the_canceller_cannot_clean_ends_his_silence_as_before(canceller):
    canceller.capture_opened(44100, 2)
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.3)
    mouth.starts_saying(REST, seconds=1.0)

    assert _handed(listener) == [QUESTION]
    mouth.stops()
    _stop(listener)


def test_the_longest_collection_is_not_held_by_his_voice(canceller):
    listener = _listener(max_collect=COLLECT + 0.3)
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.2)
    mouth.starts_saying(REST, seconds=0.2)
    mouth.goes_on(ENDPOINT + 0.4)

    assert _handed(listener) == [QUESTION]
    mouth.stops()
    _stop(listener)


def test_a_voice_that_does_not_stop_holds_the_question_no_longer_than_an_utterance_lasts(canceller):
    listener = _listener(max_utterance_ms=3000)
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.2)
    # No pause in it, so the voice detector never closes it; with the
    # pre-roll, 2.5 s of voice is still shorter than an utterance may be.
    mouth.starts_saying("", seconds=2.5)
    assert _handed(listener) == []

    mouth.goes_on(0.4)

    assert _handed(listener) == [QUESTION]
    _stop(listener)


def test_with_no_audio_arriving_an_open_utterance_holds_nothing(canceller):
    """While no audio comes (dictation has the microphone, the capture
    stalls), the utterance left open is not his voice going on."""
    listener = _listener()
    mouth = _Mouth(listener)

    mouth.says("jarvis " + QUESTION)
    mouth.until_his_silence_runs_out(then_left=0.3)
    mouth.starts_saying(REST, seconds=0.2)
    assert listener.is_speech_active

    # The loop's check when the audio queue stays empty, on the wall clock,
    # which is past his silence: the microphone's clock started two
    # minutes back.
    listener._check_query_timeout()

    assert _handed(listener) == [QUESTION]
    _stop(listener)
