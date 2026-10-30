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
    tts.voice_delay.return_value = 0.0
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


def _settled(listener):
    """Every turn handed over has been answered, stopped or set aside."""
    return _wait_for(lambda: not listener._a_reply_is_on_its_way())


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
        assert listener._turns.qsize() == 1, "the second was not kept waiting"
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
        assert _settled(listener)

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
        assert _settled(listener)

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
        assert _settled(listener)

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


@patch("builtins.print")
def test_a_stop_the_judge_hears_cuts_the_rest_of_the_reply_too(_print):
    """Not every stop is one of the configured words: the judge reads one
    out of a sentence ("arrête de parler, s'il te plaît")."""
    from test_hot_window_input import _install_intent_judge, _make_judgment

    engine = _Engine(before=["Il est midi pile, Monsieur."],
                     after=["Et il fera beau cet après-midi."])
    listener, tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert _wait_for(lambda: _spoken(tts) == ["Il est midi pile, Monsieur."])

        tts.is_speaking.return_value = True
        _install_intent_judge(listener, _make_judgment(directed=True, stop=True))
        listener._process_transcript("arrête de parler s'il te plaît", utterance_energy=0.01)
        tts.is_speaking.return_value = False
        engine.release.set()
        assert engine.done.wait(WAIT)
        assert _settled(listener)

        assert _spoken(tts) == ["Il est midi pile, Monsieur."]
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_question_holding_a_stop_word_is_a_question_while_she_thinks(_print):
    engine = _Engine()
    listener, tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)

        listener._process_transcript("jarvis how do i stop the timer", utterance_energy=0.01)
        engine.release.set()

        assert _wait_for(lambda: _spoken(tts) == ["Il est midi."]), "the reply in hand was stopped"
        assert listener.state_manager.is_collecting(), "the question was not kept"
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_the_next_turn_waits_until_she_has_finished_speaking(_print):
    """Started while the last reply is still audible, it would replace the
    echo detector's record of that reply and its window would open over the
    new one."""
    engine = _Engine()
    engine.release.set()
    listener, tts, patcher = _listener_with(engine)
    try:
        tts.is_speaking.return_value = True
        listener._hand_over("et la météo ?")
        time.sleep(0.3)
        assert engine.queries == [], "a turn started while she was speaking"

        tts.is_speaking.return_value = False
        assert _wait_for(lambda: engine.queries == ["et la météo ?"])
    finally:
        _stop(listener, patcher)


@patch("builtins.print")
def test_what_others_say_while_she_thinks_leaves_her_thinking(_print):
    from jarvis.state import JarvisState, get_jarvis_state

    engine = _Engine()
    listener, _tts, patcher = _listener_with(engine)
    tune = MagicMock()
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        listener._tune_player = tune

        listener._process_transcript("the weather is nice today", utterance_energy=0.01)

        assert get_jarvis_state().state == JarvisState.THINKING
        assert not tune.stop_tune.called
    finally:
        listener._tune_player = None
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_what_he_says_next_stays_in_the_queue_when_a_query_is_handed_over(_print):
    """The queue holds what he said after the utterance just handled: his
    next words, not stale audio."""
    listener, _tts = _create_listener()
    block = np.zeros((listener._frame_samples, 1), dtype=np.float32)
    for _ in range(10):
        listener._on_audio(block, len(block), None, None)

    listener._hand_over("quelle heure est-il")

    assert listener._audio_q.qsize() == 10
    listener.state_manager.stop()


@patch("builtins.print")
def test_a_stop_while_she_thinks_drops_the_question_being_collected(_print):
    engine = _Engine()
    listener, _tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        listener.state_manager.start_collection("et la météo")

        listener._process_transcript("stop", utterance_energy=0.01)

        assert not listener.state_manager.is_collecting()
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_turn_stopped_while_it_waits_for_the_query_lock_never_runs(_print):
    """A typed question holds the lock; a stop heard meanwhile means the
    spoken one waiting for it is not run at all, tools included."""
    from jarvis.daemon import query_lock

    engine = _Engine()
    engine.release.set()
    listener, _tts, patcher = _listener_with(engine)
    try:
        with query_lock():
            listener._hand_over("quelle heure est-il")
            assert _wait_for(lambda: listener._turn_in_hand is not None)
            listener._process_transcript("stop", utterance_energy=0.01)
        assert _settled(listener)

        assert engine.queries == []
    finally:
        _stop(listener, patcher)


@patch("builtins.print")
def test_nothing_handed_over_is_answered_once_the_listener_stops(_print):
    engine = _Engine()
    listener, tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("quelle heure est-il")
        assert engine.started.wait(WAIT)
        listener._hand_over("et la météo ?")

        listener.stop()
        engine.release.set()
        assert engine.done.wait(WAIT)
        time.sleep(0.3)

        assert engine.queries == ["quelle heure est-il"]
        assert _spoken(tts) == []
    finally:
        engine.release.set()
        patcher.stop()


@patch("builtins.print")
def test_the_turn_keeps_the_language_he_asked_in(_print):
    seen = []
    engine = _Engine()
    engine.release.set()
    listener, tts, patcher = _listener_with(engine)
    original = engine.__call__

    def recording(*args, **kwargs):
        seen.append(kwargs.get("language"))
        return original(*args, **kwargs)

    patcher.stop()
    patcher = patch("jarvis.reply.engine.run_reply_engine", side_effect=recording)
    patcher.start()
    try:
        tts.is_speaking.return_value = True
        listener._last_detected_language = "tr"
        listener._hand_over("bugün hava nasıl")
        listener._last_detected_language = "en"
        tts.is_speaking.return_value = False

        assert _wait_for(lambda: _spoken(tts) == ["Il est midi."])
        assert seen == ["tr"]
        assert [c.kwargs.get("language") for c in tts.speak.call_args_list] == ["tr"]
    finally:
        _stop(listener, patcher)


@patch("builtins.print")
def test_past_the_waiting_limit_the_oldest_question_is_dropped(_print):
    from jarvis.listening.listener import MAX_WAITING_TURNS

    engine = _Engine()
    listener, _tts, patcher = _listener_with(engine)
    try:
        listener._hand_over("q0")
        assert engine.started.wait(WAIT)
        for i in range(1, MAX_WAITING_TURNS + 2):
            listener._hand_over(f"q{i}")
        engine.release.set()
        assert _settled(listener)

        expected = ["q0"] + [f"q{i}" for i in range(2, MAX_WAITING_TURNS + 2)]
        assert engine.queries == expected
    finally:
        engine.release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_failed_turn_leaves_no_streamed_remainder_behind(_print):
    """A narration said after a reply that failed mid-stream is said whole."""
    listener, tts, patcher = _listener_with(_Engine())
    patcher.stop()

    def failing(*_args, on_token=None, **_kwargs):
        on_token("Une première phrase assez longue. ")
        on_token("Et la suite ")
        raise RuntimeError("provider down")

    patcher = patch("jarvis.reply.engine.run_reply_engine", side_effect=failing)
    patcher.start()
    try:
        listener._hand_over("quelle heure est-il")
        assert _settled(listener)
        tts.speak.reset_mock()

        listener.enqueue_reply("Le rappel de midi.")
        assert _wait_for(lambda: _spoken(tts) == ["Le rappel de midi."])
    finally:
        _stop(listener, patcher)


# ── Her spoken question ──────────────────────────────────────────────
#
# A spoken question is answered by the next voice turn. Only words heard
# after she asked it can answer it.


def _asks(dm):
    """An engine whose reply ends on a spoken question."""
    from jarvis.tools.confirmation import CHANNEL_PAROLE, PendingAction

    def engine(*_args, **_kwargs):
        dm.begin_turn("voix")
        dm.raise_pending(PendingAction.create(
            tool="getWeather", args={}, risk="lecture", channel=CHANNEL_PAROLE,
            origin="voix", query_redacted="", raised_at_turn=dm.current_turn("voix"),
            ttl_sec=180.0,
        ))
        return "Je regarde la météo ?"
    return engine


def _with_a_question(listener, release):
    from jarvis.memory.conversation import DialogueMemory

    dm = DialogueMemory()
    listener.dialogue_memory = dm
    asks = _asks(dm)
    answered = []

    def engine(db, cfg, tts, query, dm_, **kwargs):
        answered.append(query)
        if len(answered) == 1:
            release.wait(WAIT)
            return asks()
        return "D'accord."
    return dm, engine, answered


@patch("builtins.print")
def test_a_question_asked_before_hers_does_not_answer_it(_print):
    release = threading.Event()
    listener, tts, patcher = _listener_with(_Engine())
    patcher.stop()
    dm, engine, answered = _with_a_question(listener, release)
    patcher = patch("jarvis.reply.engine.run_reply_engine", side_effect=engine)
    patcher.start()
    try:
        listener._hand_over("mets de la musique")
        assert _wait_for(lambda: answered == ["mets de la musique"])
        listener._last_heard_at = time.time()
        listener._hand_over("oui vas-y et la météo")
        release.set()
        assert _settled(listener)

        assert answered == ["mets de la musique"]
        assert dm.peek_pending() is not None, "her question was consumed"
    finally:
        release.set()
        _stop(listener, patcher)


@patch("builtins.print")
def test_what_he_says_once_he_heard_her_question_answers_it(_print):
    release = threading.Event()
    release.set()
    listener, tts, patcher = _listener_with(_Engine())
    patcher.stop()
    dm, engine, answered = _with_a_question(listener, release)
    patcher = patch("jarvis.reply.engine.run_reply_engine", side_effect=engine)
    patcher.start()
    try:
        listener._hand_over("mets de la musique")
        assert _wait_for(lambda: _spoken(tts) == ["Je regarde la météo ?"])
        tts.speak.call_args.kwargs["completion_callback"]()   # he heard it
        time.sleep(0.05)
        listener._last_heard_at = time.time()
        listener._hand_over("oui")

        assert _wait_for(lambda: answered == ["mets de la musique", "oui"])
    finally:
        _stop(listener, patcher)


@patch("builtins.print")
def test_a_question_he_never_heard_is_withdrawn(_print):
    release = threading.Event()
    listener, tts, patcher = _listener_with(_Engine())
    patcher.stop()
    dm, engine, answered = _with_a_question(listener, release)
    patcher = patch("jarvis.reply.engine.run_reply_engine", side_effect=engine)
    patcher.start()
    try:
        listener._hand_over("mets de la musique")
        assert _wait_for(lambda: answered == ["mets de la musique"])
        listener._process_transcript("stop", utterance_energy=0.01)
        release.set()
        assert _settled(listener)
        assert dm.peek_pending() is not None

        listener._last_heard_at = time.time()
        listener._hand_over("oui")
        assert _settled(listener)

        assert dm.peek_pending() is None, "a question he never heard is still waiting"
        assert answered == ["mets de la musique", "oui"]
    finally:
        release.set()
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
    """With room to spare over the judge's default per-read deadline, the
    longest single wait the loop has on an utterance."""
    from jarvis.config import get_default_config
    from jarvis.listening.listener import AUDIO_BACKLOG_SECONDS

    assert AUDIO_BACKLOG_SECONDS > get_default_config()["intent_judge_timeout_sec"]
