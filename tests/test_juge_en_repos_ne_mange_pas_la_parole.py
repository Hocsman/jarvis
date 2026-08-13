"""What a judge in back-off is allowed to cost him.

When the intent judge cannot reach its backend it returns no verdict, and
the listener has a branch for exactly that: it says so on screen and, in a
hot window, keeps the follow-up anyway. That branch lives inside a guard
that requires the judge to be reachable, so it never runs for the failure
it was written for.

The back-off is thirty seconds long and opens on a connection error. The
sequence that reaches it is ordinary: the wake word carries the first
question without the judge, the server comes back, she answers, the hot
window opens a few seconds later — still inside the cooldown. His
follow-up then disappears with nothing printed, which is exactly what
speech never addressed to her looks like.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from tests.test_hot_window_input import (
    _create_listener,
    _simulate_tts_finish,
    _wait_for_hot_window_active,
)


def _juge_en_repos(listener):
    """A real IntentJudge put into back-off by a real connection error."""
    from jarvis.listening.intent_judge import create_intent_judge

    from jarvis.listening.transcript_buffer import TranscriptSegment

    juge = create_intent_judge(listener.cfg)
    import requests

    with patch("jarvis.listening.intent_judge.get_llm_backend") as backend:
        backend.return_value.chat.side_effect = requests.exceptions.ConnectionError(
            "connection refused"
        )
        juge.judge([TranscriptSegment("jarvis hello", 1000.0, 1001.0)])

    assert juge.available is False, "the back-off has to be open for this test to mean anything"
    listener._intent_judge = juge
    return juge


def _fenetre_chaude(listener):
    _simulate_tts_finish(listener)
    assert _wait_for_hot_window_active(listener)


def test_hot_window_speech_survives_the_judge_back_off(capsys):
    """His follow-up is kept. The branch that keeps it already exists; it
    was only ever reached when the judge could answer."""
    listener, _ = _create_listener()
    listener.echo_detector._last_tts_text = "The meeting is at four."
    _juge_en_repos(listener)
    _fenetre_chaude(listener)

    listener._process_transcript("tell me more about that")

    assert listener.state_manager.get_pending_query() == "tell me more about that"


def test_the_back_off_is_announced_rather_than_silent(capsys):
    """A judge that cannot answer is not the same thing as speech that was
    never addressed to her, and only one of the two is his problem to
    solve. He cannot solve it if it is invisible."""
    listener, _ = _create_listener()
    listener.echo_detector._last_tts_text = "The meeting is at four."
    _juge_en_repos(listener)
    _fenetre_chaude(listener)
    capsys.readouterr()

    listener._process_transcript("tell me more about that")

    sortie = capsys.readouterr().out
    assert "unavailable" in sortie
    assert "ConnectionError" in sortie


def test_a_wake_word_question_during_the_back_off_still_says_so(capsys):
    """Outside a hot window the wake word carries the question on its own,
    so nothing is lost — but he is running degraded and is told."""
    listener, _ = _create_listener()
    _juge_en_repos(listener)
    capsys.readouterr()

    listener._process_transcript("jarvis what time is it")

    assert "what time is it" in listener.state_manager.get_pending_query()
    assert "unavailable" in capsys.readouterr().out


def test_the_back_off_still_spares_the_dead_server():
    """The whole point of the cooldown is not to hammer a server that is
    down, and not to block the audio loop on a timeout. Entering the block
    must not cost a single request."""
    listener, _ = _create_listener()
    listener.echo_detector._last_tts_text = "The meeting is at four."
    _juge_en_repos(listener)
    _fenetre_chaude(listener)

    with patch("jarvis.listening.intent_judge.get_llm_backend") as backend:
        listener._process_transcript("tell me more about that")

    assert not backend.called
