"""The whole reply is said, once, and the window opens at the end.

The pieces are tested apart: the streamer cuts, the echo reference
accumulates, the queue carries its callbacks. This is the assembly, and
the two things it must not get wrong are the two that would be heard.

Nothing may be said twice, and nothing may be dropped. `_speak_reply`
finishes what streaming started, so it must know exactly how much was
already spoken.

And the hot window opens once, after everything. Opening it per sentence
would let her own next sentence in as a follow-up — the failure three
guards were written for on 2026-08-16, which streaming would otherwise
walk straight back into.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.jarvis.listening.echo_detection import EchoDetector


def _listener():
    from src.jarvis.listening.listener import VoiceListener

    obj = VoiceListener.__new__(VoiceListener)
    obj.tts = MagicMock()
    obj.tts.enabled = True
    obj.echo_detector = EchoDetector()
    obj.cfg = MagicMock()
    obj._tune_player = None
    obj.state_manager = MagicMock()
    obj._recent_audio_energy = []
    obj._streamed_chars = 0
    obj._reply_queue = None
    obj.dialogue_memory = None
    obj.tts.is_speaking.return_value = False
    obj.activate_hot_window = MagicMock()
    return obj


REPONSE = ("Il fait beau à Bagneux aujourd'hui. "
           "La température atteint vingt-quatre degrés. "
           "Prends une veste quand même.")


def _tour(listener, reponse: str, taille: int = 9):
    """Generate `reponse` through the token callback, then finish."""
    sur_jeton = listener._speak_as_it_comes()
    for i in range(0, len(reponse), taille):
        sur_jeton(reponse[i:i + taille])
    listener._speak_reply(reponse)


def _dit(listener):
    return [c.args[0] for c in listener.tts.speak.call_args_list]


def test_the_first_sentence_leaves_before_the_last_is_written():
    """The point of the whole exercise."""
    listener = _listener()
    sur_jeton = listener._speak_as_it_comes()

    sur_jeton("Il fait beau à Bagneux aujourd'hui. La tempér")

    assert _dit(listener) == ["Il fait beau à Bagneux aujourd'hui."]


def test_everything_is_said_exactly_once():
    """Nothing spoken twice, nothing dropped: the assembled speech is the
    reply, word for word."""
    listener = _listener()

    _tour(listener, REPONSE)

    assemble = " ".join(m for m in _dit(listener) if m.strip())
    assert assemble.split() == REPONSE.split()


def test_the_window_opens_once_and_at_the_end():
    """Per sentence, she would take her own next sentence for a
    follow-up."""
    listener = _listener()

    _tour(listener, REPONSE)

    rappels = [c.kwargs.get("completion_callback")
               for c in listener.tts.speak.call_args_list]
    assert sum(1 for r in rappels if r is not None) == 1
    assert rappels[-1] is not None


def test_the_echo_reference_holds_the_whole_reply():
    """Streaming must not undo the guard that stopped her answering
    herself: an echo of the first sentence has to still match."""
    from src.jarvis.listening.listener import _carries_speech_she_did_not_say

    listener = _listener()

    _tour(listener, REPONSE)

    assert _carries_speech_she_did_not_say(
        "il fait beau à bagneux aujourd'hui",
        listener.echo_detector._last_tts_text) is False


def test_a_reply_that_never_streamed_is_spoken_whole():
    """The fallback, and the path every non-voice caller still takes. A
    backend that does not stream must lose nothing but latency."""
    listener = _listener()

    listener._speak_reply(REPONSE)

    assert _dit(listener) == [REPONSE]


def test_a_reply_of_one_unterminated_fragment_is_still_said():
    listener = _listener()

    _tour(listener, "bien sûr")

    assert " ".join(_dit(listener)).strip() == "bien sûr"


def test_without_tts_there_is_no_token_callback():
    """No mouth, no streaming — and the engine is handed nothing rather
    than a callback that would raise on every token."""
    listener = _listener()
    listener.tts.enabled = False

    assert listener._speak_as_it_comes() is None
