"""A reply is spoken in one language, from its first sentence to its last.

The engine picks a voice per queued item, so the listener decides what each
item says about its language. Two things have to hold.

The language handed over is the one Whisper heard the user speak, because
that is the language she answers in when `response_language` is empty. The
engine, which knows the configured language, decides which of the two wins.

And one reply carries one language. A reply is spoken sentence by sentence
while the microphone is still open, so a transcript of something else can
land mid-reply and move the detected language. Sentence two in another
voice than sentence one is the failure; it is also an echo reference that
no longer matches what was said.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.jarvis.listening.echo_detection import EchoDetector


def _listener(detected="fr"):
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
    obj._last_detected_language = detected
    return obj


REPLY = ("Il fait beau à Bagneux aujourd'hui. "
         "La température atteint vingt-quatre degrés. "
         "Prends une veste quand même.")


def _languages(listener):
    return [c.kwargs.get("language") for c in listener.tts.speak.call_args_list]


def _tour(listener, reply, size=9, between=None):
    """Generate `reply` through the token callback, then finish."""
    on_token = listener._speak_as_it_comes()
    for i in range(0, len(reply), size):
        on_token(reply[i:i + size])
        if between is not None:
            between(i)
    listener._speak_reply(reply)


@pytest.mark.unit
def test_a_reply_that_never_streamed_carries_the_language_it_was_heard_in():
    listener = _listener(detected="fr")

    listener._speak_reply("Il fait beau.")

    assert _languages(listener) == ["fr"]


@pytest.mark.unit
def test_every_part_of_a_streamed_reply_carries_the_same_language():
    listener = _listener(detected="fr")

    _tour(listener, REPLY)

    langues = _languages(listener)
    assert len(langues) >= 3, "the reply should have been spoken in several pieces"
    assert set(langues) == {"fr"}


@pytest.mark.unit
def test_a_language_detected_mid_reply_does_not_change_the_voice_of_the_rest():
    listener = _listener(detected="fr")

    def _someone_else_speaks(_position):
        listener._last_detected_language = "en"

    _tour(listener, REPLY, between=_someone_else_speaks)

    assert set(_languages(listener)) == {"fr"}


@pytest.mark.unit
def test_the_next_reply_follows_the_language_heard_for_it():
    listener = _listener(detected="fr")
    _tour(listener, REPLY)
    listener.tts.speak.reset_mock()

    listener._last_detected_language = "de"
    _tour(listener, "Es ist heute schön in Berlin. Nimm eine Jacke mit.")

    assert set(_languages(listener)) == {"de"}


@pytest.mark.unit
def test_nothing_heard_means_no_language_is_claimed():
    """Whisper reports nothing before the first transcription. Inventing a
    language would pick a voice for it; saying nothing lets the engine fall
    back on what was configured."""
    listener = _listener(detected=None)

    listener._speak_reply("Hello.")

    assert _languages(listener) == [None]


@pytest.mark.unit
def test_a_reply_queued_from_elsewhere_follows_the_last_language_heard():
    """A confirmation or a reminder has no utterance behind it, so the last
    language the user spoke is the best evidence of the one they read."""
    listener = _listener(detected="es")

    listener.enqueue_reply("Listo, ya está hecho.")
    listener.drain_reply_queue()

    assert _languages(listener) == ["es"]
