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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis.config import get_default_config
from jarvis.listening.wake_detection import is_wake_word_detected
from jarvis.tools.confirmation import CHANNEL_PAROLE
from test_hot_window_input import (
    _accepted_query, _console_lines, _console_says_heard_as_her_echo, _create_listener,
    _install_intent_judge, _is_beeping, _make_judgment, _prompt_shown_to,
    _simulate_tts_finish, _wait_for_hot_window_active, _wait_for_hot_window_expiry,
)

_HER_REPLY = "Pour demain, le ciel sera dégagé, Monsieur."
_TO_A_GUEST = "merci camille pour le café, tu vas bien ?"
_HIS_FOLLOW_UP = "et après-demain ?"


def _to_someone_else(**kwargs):
    return _make_judgment(directed=False, confidence="high", aside=True,
                          reasoning="said to a guest he names, not to the assistant", **kwargs)


def _in_the_window(reply=_HER_REPLY, **kwargs):
    listener, _ = _create_listener(**{"echo_tolerance": 0.02, "hot_window_seconds": 3.0,
                                      **kwargs})
    listener.echo_detector.track_tts_start(reply)
    _simulate_tts_finish(listener)
    assert _wait_for_hot_window_active(listener)
    return listener


def _hear(listener, text, **when):
    """What the audio loop does before handing the text over."""
    now = time.time()
    listener._transcript_buffer.add(text, now - 1.0, now, 0.01)
    listener._process_transcript(text, utterance_energy=0.01, **when)


def _said_not_to_her(mock_print) -> bool:
    return any("not to her" in line for line in _console_lines(mock_print))


def _a_first_name_close_to_hers(cfg) -> str:
    """A first name the wake-word check takes for a spelling of her name,
    and that is none of her names."""
    names = {cfg.wake_word, *cfg.wake_aliases}
    close = [name for name in ("boris", "chris", "doris", "harris", "mavis")
             if name not in names
             and is_wake_word_detected(name, cfg.wake_word, list(cfg.wake_aliases),
                                       cfg.wake_fuzzy_ratio)]
    assert close, "no first name here is close to her name as configured"
    return close[0]


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

        assert _accepted_query(listener) == ""

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query=_HIS_FOLLOW_UP, confidence="high"))
        _hear(listener, _HIS_FOLLOW_UP)

        assert "camille" in _prompt_shown_to(judge).lower()
        assert _accepted_query(listener) == _HIS_FOLLOW_UP
        listener.state_manager.stop()

    @pytest.mark.parametrize("confidence", ["medium", "low"])
    @patch("builtins.print")
    def test_an_aside_the_judge_is_unsure_of_is_answered(self, _print, confidence):
        """A follow-up left unanswered costs him more than an aside answered,
        and the judge's doubtful asides are mostly his words to her: only one
        it is sure of sets his speech aside."""
        listener = _in_the_window()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence=confidence, aside=True,
            reasoning="may be said to someone else, a close call"))

        _hear(listener, _HIS_FOLLOW_UP)

        assert _accepted_query(listener) == _HIS_FOLLOW_UP
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
        rejection of speech in the hot window is overridden, and answered."""
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
    def test_a_configured_spelling_of_her_name_is_his_call_too(self, _print):
        listener = _in_the_window()
        listener.cfg.wake_aliases = list(get_default_config()["wake_aliases"])
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, f"{listener.cfg.wake_aliases[0]}, quelle heure est-il ?")

        assert "quelle heure" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_first_name_close_to_hers_is_not_her_name(self, mock_print):
        """The wake-word check takes close spellings for her name, and a
        guest can be called one of them. The judge was sure the words were
        for that guest: a name that is only close to hers does not undo it."""
        listener = _in_the_window()
        listener.cfg.wake_aliases = list(get_default_config()["wake_aliases"])
        guest = _a_first_name_close_to_hers(listener.cfg)
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, f"merci {guest} pour le café, tu vas bien ?")

        assert _accepted_query(listener) == ""
        assert _said_not_to_her(mock_print)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_his_answer_to_her_question_is_never_set_aside(self, mock_print):
        """She asked before acting and waits for his yes or no. What he says
        then is his answer, whoever he names in it: it goes to the reader of
        answers, which closes the question, and is never left behind."""
        listener = _in_the_window("Voulez-vous que j'envoie le message à Hugo ?")
        asked = SimpleNamespace(channel=CHANNEL_PAROLE, origin="voix", request_id="q1")
        listener.dialogue_memory = SimpleNamespace(peek_pending=lambda: asked)
        assert listener.awaiting_spoken_answer()
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, "non, camille, laisse tomber")

        assert _accepted_query(listener) == "non, camille, laisse tomber"
        assert not _said_not_to_her(mock_print)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_what_he_adds_to_his_question_joins_it(self, mock_print):
        """His follow-up and what he adds to it both began in the window,
        and Whisper handed the first over once the window had closed. The
        addition completes his question, whatever the judge says of it."""
        listener = _in_the_window(hot_window_seconds=0.4)
        began = time.time()
        assert _wait_for_hot_window_expiry(listener)
        _install_intent_judge(listener, _make_judgment(
            directed=True, query=_HIS_FOLLOW_UP, confidence="high"))
        _hear(listener, _HIS_FOLLOW_UP,
              utterance_start_time=began, utterance_end_time=began + 0.10)
        assert listener.state_manager.is_collecting()

        _install_intent_judge(listener, _to_someone_else())
        _hear(listener, "et aussi pour lille, camille me le demande",
              utterance_start_time=began + 0.20, utterance_end_time=began + 0.35)

        assert "camille me le demande" in _accepted_query(listener)
        assert not _said_not_to_her(mock_print)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_words_that_could_be_hers_are_still_her_echo(self, mock_print):
        """Words heard as her voice stops, which nothing proves he said, are
        set aside as her echo whatever the judge calls them."""
        heard = "à demain alors, camille"
        listener = _in_the_window(echo_tolerance=0.3)
        last_word = listener.echo_detector._last_tts_finish_time
        _install_intent_judge(listener, _to_someone_else())

        _hear(listener, heard, utterance_start_time=last_word - 0.5,
              utterance_end_time=last_word + 0.1, trailing_silence=0.0)

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        assert not _said_not_to_her(mock_print)
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
