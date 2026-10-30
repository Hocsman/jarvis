"""She hears while she thinks.

Field trace (2026-10-29): the listener ran each reply on the thread that
reads the microphone. Its queue held a little over a second of audio, so
everything said while she prepared an answer, and while she began to speak
it, was thrown away: "Audio capture: queue full; 470 blocks dropped", a
"stop" that could not land, and "Est-ce que tu as toujours accès à mon..."
cut in half.

So a reply runs on a thread of its own, one at a time, and that thread is
the only one that speaks. The microphone loop hands a query over and goes
back to listening; a stop it hears while a reply is being prepared or said
silences that reply and drops the questions waiting behind it.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_hot_window_input import _create_listener

WAIT = 5.0


class _Engine:
    """A reply engine that holds each turn until it is released, and can
    stream sentences before and after that point."""

    def __init__(self, before=(), after=(), reply="Il est midi."):
        self.before, self.after, self.reply = list(before), list(after), reply
        self.release = threading.Event()
        self.started = threading.Event()
        self.done = threading.Event()
        self.queries = []
        self._running = 0
        self.overlapped = False

    def __call__(self, db, cfg, tts, query, dm, **kwargs):
        self._running += 1
        self.overlapped |= self._running > 1
        self.queries.append(query)
        on_token = kwargs.get("on_token")
        for sentence in self.before:
            on_token and on_token(sentence + " ")
        self.started.set()
        self.release.wait(WAIT)
        for sentence in self.after:
            on_token and on_token(sentence + " ")
        self._running -= 1
        self.done.set()
        return self.reply


def _listener_with(engine, **kwargs):
    listener, tts = _create_listener(**kwargs)
    patcher = patch("jarvis.reply.engine.run_reply_engine", side_effect=engine)
    patcher.start()
    listener._start_replies()
    return listener, tts, patcher


def _stop(listener, patcher):
    listener.stop()
    patcher.stop()


def _wait_for(predicate, timeout=WAIT):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _spoken(tts):
    return [c.args[0] for c in tts.speak.call_args_list]


@patch("builtins.print")
def test_the_microphone_loop_is_not_held_by_a_reply(_print):
    """What the loop calls once his question is collected returns at once,
    however long the reply takes."""
    engine = _Engine()
    listener, _tts, patcher = _listener_with(engine)
    try:
        listener.state_manager.voice_collect_seconds = 0.01
        listener.state_manager.start_collection("quelle heure est-il")
        time.sleep(0.05)

        t0 = time.monotonic()
        listener._check_query_timeout()
        held = time.monotonic() - t0

        assert engine.started.wait(WAIT), "the reply never started"
        assert held < 0.5, f"the loop was held {held:.2f}s by the reply"
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_second_question_waits_its_turn(_print):
    engine = _Engine()
    listener, _tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        listener._hand_over("et la météo ?")
        time.sleep(0.1)
        assert engine.queries == ["quelle heure est-il"], "the second started before the first ended"

        engine.release.set()
        assert _wait_for(lambda: engine.queries == ["quelle heure est-il", "et la météo ?"])
        assert not engine.overlapped
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_stop_while_she_thinks_says_nothing_of_that_reply(_print):
    from jarvis.state import JarvisState, get_jarvis_state

    engine = _Engine()
    listener, tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)

        listener._process_transcript("stop", utterance_energy=0.01)
        engine.release.set()
        assert engine.done.wait(WAIT)
        time.sleep(0.1)

        assert _spoken(tts) == []
        assert get_jarvis_state().state == JarvisState.IDLE
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_stop_while_she_speaks_cuts_the_rest_of_the_reply(_print):
    engine = _Engine(before=["Il est midi pile, Monsieur."],
                     after=["Et il fera beau cet après-midi."],
                     reply="Il est midi pile, Monsieur. Et il fera beau cet après-midi.")
    listener, tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        assert _wait_for(lambda: _spoken(tts) == ["Il est midi pile, Monsieur."])

        tts.is_speaking.return_value = True
        listener._process_transcript("stop", utterance_energy=0.01)
        tts.is_speaking.return_value = False
        engine.release.set()
        assert engine.done.wait(WAIT)
        time.sleep(0.1)

        assert _spoken(tts) == ["Il est midi pile, Monsieur."]
        assert tts.interrupt.called
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_stop_drops_the_questions_waiting_their_turn(_print):
    engine = _Engine()
    listener, _tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        listener._hand_over("et la météo ?")

        listener._process_transcript("stop", utterance_energy=0.01)
        engine.release.set()
        assert engine.done.wait(WAIT)
        time.sleep(0.2)

        assert engine.queries == ["quelle heure est-il"]
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_question_after_a_stop_is_answered(_print):
    engine = _Engine()
    listener, tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        listener._process_transcript("stop", utterance_energy=0.01)
        engine.release.set()
        assert engine.done.wait(WAIT)

        engine.done.clear()
        listener._hand_over("et la météo ?")
        assert engine.done.wait(WAIT)

        assert _wait_for(lambda: _spoken(tts) == ["Il est midi."])
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_one_thread_speaks_and_it_is_not_the_microphone_loop(_print):
    """Her own replies and the replies handed over from elsewhere (a
    click-approved action's narration) leave from one thread, so the echo
    detector's record of what she last said is written by one writer."""
    engine = _Engine()
    engine.release.set()
    listener, tts, patcher = _listener_with(engine)
    speakers = []
    tts.speak.side_effect = lambda text, **kw: speakers.append(threading.current_thread())
    try:
        listener._hand_over("quelle heure est-il")
        assert _wait_for(lambda: len(speakers) == 1)
        listener.enqueue_reply("c'est fait")
        assert _wait_for(lambda: len(speakers) == 2)

        assert speakers[0] is speakers[1]
        assert speakers[0] is not threading.current_thread()
    finally:
        _stop(listener, patcher)


# ── What the microphone hands over ───────────────────────────────────


def _frame(listener):
    return np.zeros(listener._frame_samples, dtype=np.float32)


@patch("builtins.print")
def test_an_utterance_keeps_the_time_its_audio_was_captured(_print):
    """Audio read late, after the loop was busy, still says when it was
    spoken: the echo detector and the hot window compare these times with
    when she spoke."""
    from types import SimpleNamespace

    listener, _tts = _create_listener()
    cfg = listener.cfg
    cfg.vad_frame_ms, cfg.vad_pre_roll_ms, cfg.endpoint_silence_ms = 20, 240, 200
    cfg.max_utterance_ms, cfg.tts_max_utterance_ms = 12000, 3000
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._configure_frames()
    listener._whisper_backend, listener._whisper_device = "faster-whisper", "cuda"
    listener.model = MagicMock()
    listener.model.transcribe.return_value = (
        [SimpleNamespace(text=" Bonjour.", avg_logprob=-0.1, no_speech_prob=0.01)],
        SimpleNamespace(language="fr"))
    heard = []
    listener._process_transcript = lambda text, energy, start, end, *_a, **_k: heard.append((start, end))
    listener._is_speech_frame = lambda frame: True

    captured = time.time() - 5.0
    step = listener._frame_ms / 1000.0
    voiced = 25
    for i in range(voiced):
        listener._take_frame(_frame(listener), captured + i * step)
    listener._is_speech_frame = lambda frame: False
    i = voiced
    while not heard and i < voiced + 200:
        listener._take_frame(_frame(listener), captured + i * step)
        i += 1

    assert heard, "the utterance was never closed"
    start, end = heard[0]
    assert abs(start - captured) < 0.5
    assert abs(end - (captured + (i - 1) * step)) < 0.05
    listener.state_manager.stop()


def test_a_stall_as_long_as_the_backlog_loses_nothing():
    """The loop still waits on the intent judge and on Whisper. The queue
    holds longer than that, so a stall delays his words instead of
    dropping them."""
    from jarvis.listening.listener import AUDIO_BACKLOG_SECONDS

    listener, _tts = _create_listener()
    listener._configure_frames()
    block = np.zeros((listener._frame_samples, 1), dtype=np.float32)
    blocks = int(AUDIO_BACKLOG_SECONDS * 1000 / listener._frame_ms)

    for _ in range(blocks):
        listener._on_audio(block, len(block), None, None)
    assert listener._audio_dropped == 0

    listener._on_audio(block, len(block), None, None)
    assert listener._audio_dropped == 1
    listener.state_manager.stop()


def test_the_backlog_outlasts_the_intent_judge():
    """The longest the loop waits by default is the judge's deadline."""
    from jarvis.config import get_default_config
    from jarvis.listening.listener import AUDIO_BACKLOG_SECONDS

    assert AUDIO_BACKLOG_SECONDS > get_default_config()["intent_judge_timeout_sec"]
