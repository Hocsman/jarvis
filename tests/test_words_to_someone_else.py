"""
His words to someone else, heard in the hot window.

He is not always alone: a guest in the room, someone on a call, the people
watching a live stream. Right after her reply the hot window takes what he
says without her name, and what he says then to one of them is not for her.
Field trace: the judge read his thanks to a named viewer as said to someone
else, with high confidence, the hot window answered it anyway, and she told
him that was not her name.

The judge says so with ``aside``, and no hot-window override answers it.
These tests use the real state manager and echo detector, and a judge that
returns the verdict each test names.
"""

import time
from unittest.mock import patch

import pytest

from test_hot_window_input import (
    _accepted_query, _console_lines, _create_listener, _install_intent_judge,
    _is_beeping, _make_judgment, _prompt_shown_to, _simulate_tts_finish,
    _wait_for_hot_window_active,
)

_HER_REPLY = "Pour demain, le ciel sera dégagé, Monsieur."
_TO_A_GUEST = "merci camille pour le café, tu vas bien ?"
_HIS_FOLLOW_UP = "et après-demain ?"


def _to_someone_else(**kwargs):
    return _make_judgment(directed=False, confidence="high", aside=True,
                          reasoning="said to a guest he names, not to the assistant", **kwargs)


def _in_the_window(reply=_HER_REPLY):
    listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
    listener.echo_detector.track_tts_start(reply)
    _simulate_tts_finish(listener)
    assert _wait_for_hot_window_active(listener)
    return listener


def _hear(listener, text):
    """What the audio loop does before handing the text over."""
    now = time.time()
    listener._transcript_buffer.add(text, now - 1.0, now, 0.01)
    listener._process_transcript(text, utterance_energy=0.01)


@pytest.mark.unit
class TestHisWordsToSomeoneElse:

    @patch("builtins.print")
    def test_they_are_not_answered(self, _print):
        listener = _in_the_window()
        listener.cfg.tune_enabled = True
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, _TO_A_GUEST)

        assert _accepted_query(listener) == ""
        assert not _is_beeping(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_console_says_what_became_of_them(self, mock_print):
        listener = _in_the_window()
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, _TO_A_GUEST)

        said = [line for line in _console_lines(mock_print) if "not to her" in line]
        assert said and _TO_A_GUEST[:30] in said[0]
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_they_do_not_cost_him_his_follow_up(self, _print):
        """The window stays open: what he then says to her is a follow-up."""
        listener = _in_the_window()
        _install_intent_judge(listener, _to_someone_else())
        _hear(listener, _TO_A_GUEST)

        _install_intent_judge(listener, _make_judgment(
            directed=True, query=_HIS_FOLLOW_UP, confidence="high"))
        _hear(listener, _HIS_FOLLOW_UP)

        assert _accepted_query(listener) == _HIS_FOLLOW_UP
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_they_stay_as_context_for_what_he_asks_her_next(self, _print):
        """Unlike a phrase Whisper invents, they are words someone said: he
        can ask her about them."""
        listener = _in_the_window()
        _install_intent_judge(listener, _to_someone_else())
        _hear(listener, _TO_A_GUEST)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query=_HIS_FOLLOW_UP, confidence="high"))
        _hear(listener, _HIS_FOLLOW_UP)

        assert "camille" in _prompt_shown_to(judge).lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_judge_that_also_says_directed_is_believed(self, _print):
        """A verdict that contradicts itself falls to his side."""
        listener = _in_the_window()
        _install_intent_judge(listener, _make_judgment(
            directed=True, query=_HIS_FOLLOW_UP, confidence="high", aside=True))

        _hear(listener, _HIS_FOLLOW_UP)

        assert _accepted_query(listener) == _HIS_FOLLOW_UP
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_rejection_that_does_not_say_so_is_still_overridden(self, _print):
        """Only the judge's word for it sets his speech aside: any other
        rejection of speech in the hot window is answered as before."""
        listener = _in_the_window()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", reasoning="not a question"))

        _hear(listener, _HIS_FOLLOW_UP)

        assert _accepted_query(listener) == _HIS_FOLLOW_UP
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_name_in_them_is_still_his_call(self, _print):
        """The wake word decides, in the hot window as outside it."""
        listener = _in_the_window()
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, "jarvis, quelle heure est-il ?")

        assert "quelle heure" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_outside_the_window_her_name_is_still_his_call(self, _print):
        """Outside the hot window the judge only runs on a wake word or while
        she speaks, and a doubtful verdict there still falls through to the
        wake word, whatever it says of an aside."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="medium", aside=True, reasoning="unsure"))

        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert "what time is it" in _accepted_query(listener)
        listener.state_manager.stop()
