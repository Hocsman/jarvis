"""
Her own sentences at the edges of what he said.

Whisper hands back one utterance as several segments. When he speaks over
her, or the moment she stops, the first or the last of them can be a
sentence of her reply coming back through the microphone, and the rest his.
Field trace: his question went to the model with a sentence of her previous
reply stuck to its end, and she answered the mixture; another time her
sentences on both sides of his question drowned it, and he asked three times.

These tests go through the audio loop's own path with a faked Whisper, the
real echo detector and the real state manager.
"""

import time
from unittest.mock import patch

import pytest
from rapidfuzz import fuzz

from jarvis.listening.listener import _without_her_sentences_at_the_edges
from test_hot_window_input import (
    _FLOOR, _HER_VOICE, _HIS_VOICE,
    _accepted_query, _console_lines, _create_listener, _current_line_shown_to,
    _judge_that_reads_the_marker, _level, _she_says, _wait_for_hot_window_active,
    _whisper_hears,
)

_HER_REPLY = ("Le train de neuf heures est complet, Monsieur. "
              "Il reste des places dans celui de onze heures. "
              "Voulez-vous que je réserve un billet ?")
_HER_FIRST_SENTENCE = "Le train de neuf heures est complet, Monsieur."
_HER_MIDDLE_SENTENCE = "Il reste des places dans celui de onze heures."
_HER_LAST_SENTENCE = "Voulez-vous que je réserve un billet ?"
# Her middle sentence as Whisper bends it coming back: a word dropped, one
# rewritten, still one run of her reply.
_HER_MIDDLE_SENTENCE_BENT = "Il reste des glaces dans celui de onze heures."
_HIS_QUESTION = "Jarvis, combien coûte un aller simple pour Nantes ?"
_HIS_QUESTION_WITHOUT_HER_NAME = "combien coûte un aller simple pour Nantes ?"
# He corrects her, which takes her own wording.
_HIS_CORRECTION = "Non, le train de dix heures est complet, pas celui de neuf heures."
_HIS_REQUEST_AFTER_IT = "Jarvis, regarde encore."

_MIN_WORDS = 3


def _her_words_in(query: str) -> bool:
    """Whether *query* still carries a sentence of her reply."""
    return any(fuzz.partial_ratio(sentence.lower(), query.lower()) >= 85
               for sentence in (_HER_FIRST_SENTENCE, _HER_MIDDLE_SENTENCE, _HER_LAST_SENTENCE))


def _while_she_speaks():
    """A listener with no judge, so the wake-word check answers what was
    heard, and her reply being said."""
    listener, _ = _create_listener(tts_speaking=True)
    listener.echo_detector.track_tts_start(_HER_REPLY, baseline_energy=_FLOOR)
    return listener


@pytest.mark.unit
class TestHisQuestionAmongHerSentences:

    @patch("builtins.print")
    def test_her_sentence_before_his_question_is_left_out(self, _print):
        listener = _while_she_speaks()

        _whisper_hears(listener, [_HER_MIDDLE_SENTENCE, _HIS_QUESTION],
                       started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

        query = _accepted_query(listener)
        assert "nantes" in query.lower()
        assert not _her_words_in(query)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_sentence_after_his_question_is_left_out(self, _print):
        listener = _while_she_speaks()

        _whisper_hears(listener, [_HIS_QUESTION, _HER_LAST_SENTENCE],
                       started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

        query = _accepted_query(listener)
        assert "nantes" in query.lower()
        assert not _her_words_in(query)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_sentences_on_both_sides_are_left_out(self, _print):
        listener = _while_she_speaks()

        _whisper_hears(listener,
                       [_HER_FIRST_SENTENCE, _HER_MIDDLE_SENTENCE, _HIS_QUESTION,
                        _HER_LAST_SENTENCE],
                       started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

        query = _accepted_query(listener)
        assert "nantes" in query.lower()
        assert not _her_words_in(query)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_sentence_of_hers_whisper_bent_is_left_out_too(self, _print):
        listener = _while_she_speaks()

        _whisper_hears(listener, [_HER_MIDDLE_SENTENCE_BENT, _HIS_QUESTION],
                       started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

        query = _accepted_query(listener)
        assert "nantes" in query.lower()
        assert "onze heures" not in query.lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_judge_is_shown_his_words_only(self, _print):
        """Just after her last word, in the hot window: the judge reads his
        question, not her sentence with it, and she answers it."""
        listener, _ = _create_listener(echo_tolerance=0.1, hot_window_seconds=0.5)
        last_word = _she_says(listener, _HER_REPLY)
        assert _wait_for_hot_window_active(listener)
        judge = _judge_that_reads_the_marker(listener)

        _whisper_hears(listener, [_HER_LAST_SENTENCE, _HIS_QUESTION_WITHOUT_HER_NAME],
                       started=last_word - 1.0, level=_level(listener, _HIS_VOICE))

        shown = _current_line_shown_to(judge).lower()
        assert "nantes" in shown
        assert "billet" not in shown
        assert "nantes" in _accepted_query(listener).lower()
        assert not _her_words_in(_accepted_query(listener))
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_console_says_what_was_kept(self, mock_print):
        listener = _while_she_speaks()

        _whisper_hears(listener, [_HER_MIDDLE_SENTENCE, _HIS_QUESTION],
                       started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

        kept = [line for line in _console_lines(mock_print) if "✂️" in line]
        assert kept and "nantes" in kept[0].lower()
        assert "onze heures" not in kept[0].lower()
        listener.state_manager.stop()


@pytest.mark.unit
class TestWhatIsNeverCut:

    def test_his_correction_in_her_words_is_not_taken_for_hers(self):
        """Correcting her takes her wording. The correction is his, and it
        is what the request after it is about."""
        heard = [_HIS_CORRECTION, _HIS_REQUEST_AFTER_IT]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    @patch("builtins.print")
    def test_her_reply_coming_back_whole_is_still_her_echo(self, _print):
        listener = _while_she_speaks()

        _whisper_hears(listener, [_HER_FIRST_SENTENCE, _HER_MIDDLE_SENTENCE],
                       started=time.time() - 3.0, level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_word_or_two_after_her_sentences_is_not_split_from_them(self, mock_print):
        """What Whisper makes of the quiet after her voice is a word or two.
        Split from her sentences it would stand alone as words she did not
        say; left with them, the utterance is her echo."""
        listener, _ = _create_listener(echo_tolerance=0.1, hot_window_seconds=0.5)
        last_word = _she_says(listener, _HER_REPLY)
        assert _wait_for_hot_window_active(listener)
        _judge_that_reads_the_marker(listener)

        _whisper_hears(listener, [_HER_MIDDLE_SENTENCE, _HER_LAST_SENTENCE, "Merci."],
                       started=last_word - 2.0, level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert not any("✂️" in line for line in _console_lines(mock_print))
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_sentence_quoted_long_after_she_spoke_is_his(self, _print):
        """Outside her time nothing of hers comes back through the
        microphone: a sentence of hers he repeats is his."""
        listener, _ = _create_listener(echo_tolerance=0.05, hot_window_seconds=0.1)
        _she_says(listener, _HER_REPLY)
        time.sleep(0.3)

        _whisper_hears(listener, [_HER_MIDDLE_SENTENCE, "Jarvis, tu en es sûre ?"],
                       started=time.time() - 0.1, level=_level(listener, _HIS_VOICE))

        query = _accepted_query(listener).lower()
        assert "onze heures" in query
        assert "sûre" in query
        listener.state_manager.stop()


@pytest.mark.unit
class TestTheCutItself:

    def test_one_piece_is_never_cut(self):
        heard = [f"{_HER_LAST_SENTENCE} {_HIS_QUESTION}"]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_nothing_of_hers_to_compare_against_cuts_nothing(self):
        heard = [_HER_LAST_SENTENCE, _HIS_QUESTION]
        assert _without_her_sentences_at_the_edges(heard, "", _MIN_WORDS) == (heard, [])

    def test_a_sentence_of_hers_between_his_stays(self):
        """Only the edges: what she said between two sentences of his is
        left where it is."""
        heard = ["Attends une seconde s'il te plaît.", _HER_MIDDLE_SENTENCE, _HIS_QUESTION]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_a_short_piece_in_her_words_is_his(self):
        """A word or two she also said ("Monsieur.") is too little to tell."""
        heard = ["Monsieur.", _HIS_QUESTION]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_it_says_what_it_took_out(self):
        heard = [_HER_FIRST_SENTENCE, _HIS_QUESTION, _HER_LAST_SENTENCE]
        his, hers = _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS)
        assert his == [_HIS_QUESTION]
        assert hers == [_HER_FIRST_SENTENCE, _HER_LAST_SENTENCE]
