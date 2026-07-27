"""One thread speaks, and it is always the listener's.

A click-approved action produces its reply on a resume worker. If that
worker spoke directly, it would race the listener: the listener speaks
*outside* the block that holds the query lock, so both could be inside
`track_tts_start` at once — two writes to the echo detector's record of
what was last said, and two hot-window activations. The echo detector
would then be comparing the next transcript against the wrong sentence,
which is how a user's answer gets deleted as an echo.

So a reply from elsewhere is queued, and the listener drains it on its
own thread through the same speaking path it uses for its own replies.

The hot window widens while a spoken question is waiting, because 3
seconds is tuned for a follow-up and a person weighing whether to let
something happen pauses first. The echo threshold is never relaxed:
letting her own voice approve her own question is the inverted-safety
case, and it is the one direction this must not move.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.jarvis.listening.echo_detection import EchoDetector


@pytest.fixture
def listener():
    """A VoiceListener with its heavy parts stubbed out."""
    from src.jarvis.listening.listener import VoiceListener

    obj = VoiceListener.__new__(VoiceListener)
    obj.tts = MagicMock()
    obj.tts.enabled = True
    obj.echo_detector = EchoDetector()
    obj.cfg = MagicMock()
    obj.cfg.hot_window_seconds = 3.0
    obj.cfg.confirmation_hot_window_sec = 12.0
    obj.dialogue_memory = None
    obj._reply_queue = None
    # Bits `_speak_reply` touches that a real listener sets up in __init__.
    obj._tune_player = None
    obj.state_manager = MagicMock()
    obj._recent_audio_energy = []
    return obj


# ── The queue ─────────────────────────────────────────────────────────


def test_a_queued_reply_is_not_spoken_by_the_caller(listener):
    """Handing it over must not speak it on the handing thread."""
    listener.enqueue_reply("c'est fait")

    assert listener.tts.speak.called is False


def test_the_listener_speaks_what_was_queued(listener):
    listener.enqueue_reply("c'est fait")

    listener.drain_reply_queue()

    assert listener.tts.speak.call_args.args[0] == "c'est fait"


def test_draining_an_empty_queue_does_nothing(listener):
    listener.drain_reply_queue()

    assert listener.tts.speak.called is False


def test_a_drained_reply_is_not_spoken_twice(listener):
    listener.enqueue_reply("c'est fait")

    listener.drain_reply_queue()
    listener.drain_reply_queue()

    assert listener.tts.speak.call_count == 1


def test_nothing_is_queued_while_tts_is_unavailable(listener):
    """Queueing into a dead engine would hold a reply forever and speak it
    at some unrelated later moment."""
    listener.tts.enabled = False

    listener.enqueue_reply("c'est fait")
    listener.drain_reply_queue()

    assert listener.tts.speak.called is False


def test_an_empty_reply_is_not_queued(listener):
    listener.enqueue_reply("")
    listener.enqueue_reply(None)

    listener.drain_reply_queue()

    assert listener.tts.speak.called is False


def test_a_queued_reply_goes_through_the_echo_bookkeeping(listener):
    """Same path as her own replies. A sentence spoken without being
    recorded is a sentence the echo detector will not recognise coming
    back, so she answers herself."""
    listener.enqueue_reply("c'est fait")

    listener.drain_reply_queue()

    assert listener.echo_detector._last_tts_text == "c'est fait"


# ── The hot window ────────────────────────────────────────────────────


def test_the_window_is_the_configured_default_with_nothing_pending(listener):
    assert listener.hot_window_duration() == 3.0


def test_a_waiting_spoken_question_widens_the_window(listener):
    """Three seconds is tuned for "tell me more". Deciding whether to let
    something happen takes longer than that."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.tools.confirmation import CHANNEL_PAROLE, PendingAction

    dm = DialogueMemory()
    dm.begin_turn()
    dm.raise_pending(PendingAction.create(
        tool="getWeather", args={}, risk="lecture", channel=CHANNEL_PAROLE,
        origin="voix", query_redacted="", raised_at_turn=dm.current_turn(),
        ttl_sec=180.0,
    ))
    listener.dialogue_memory = dm

    assert listener.hot_window_duration() == 12.0


def test_a_gesture_only_question_does_not_widen_it(listener):
    """Nothing said can settle it, so the extra listening would only pick
    up an answer that has to be thrown away."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.tools.confirmation import CHANNEL_GESTE, PendingAction

    dm = DialogueMemory()
    dm.begin_turn()
    dm.raise_pending(PendingAction.create(
        tool="localFiles", args={}, risk="destructif", channel=CHANNEL_GESTE,
        origin="voix", query_redacted="", raised_at_turn=dm.current_turn(),
        ttl_sec=180.0,
    ))
    listener.dialogue_memory = dm

    assert listener.hot_window_duration() == 3.0


def test_the_echo_threshold_is_never_relaxed_for_a_question():
    """The one direction this must not move. Her own voice approving her
    own question is the inverted-safety case, and the wording of the
    question is chosen against this exact threshold."""
    assert EchoDetector.PURE_ECHO_THRESHOLD == 70
