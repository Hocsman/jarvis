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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.listening.echo_detection import EchoDetector
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
# Her middle sentence as Whisper bends it coming back: one word rewritten,
# still one run of her reply.
_HER_MIDDLE_SENTENCE_BENT = "Il reste des glaces dans celui de onze heures."
# The same sentence decimated, too far from her reply to be known for one of
# its sentences, and every word of it hers.
_HER_MIDDLE_SENTENCE_DECIMATED = "reste des places celui onze heures"
_HIS_QUESTION = "Jarvis, combien coûte un aller simple pour Nantes ?"
_HIS_QUESTION_WITHOUT_HER_NAME = "combien coûte un aller simple pour Nantes ?"
# He corrects her, which takes her own wording.
_HIS_CORRECTION = "Non, le train de dix heures est complet, pas celui de neuf heures."
_HIS_REQUEST_AFTER_IT = "Jarvis, regarde encore."
# A reply of one short sentence, and one segment Whisper made of it and of
# his first words.
_HER_SHORT_REPLY = "Le billet est réservé, Monsieur."
_HER_SHORT_REPLY_THEN_HIS_WORDS = "Le billet est réservé, Monsieur. Et pour le retour alors ?"
# What Whisper makes of a television, or of nothing: no word of hers, and
# not her name.
_WORDS_NOBODY_SAID_TO_HER = "la bouilloire siffle depuis ce matin"

# The fewest words a remainder needs, as every salvage of the listener has it.
_MIN_WORDS = EchoDetector().min_salvage_words
# Words only her sentences carry, one for each.
_ONLY_HERS = ("complet", "onze heures", "billet")


def _her_words_in(text: str) -> bool:
    """Whether *text* still carries a sentence of her reply."""
    return any(words in text.lower() for words in _ONLY_HERS)


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
        assert not _her_words_in(query)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_reply_said_sentence_by_sentence_is_one_reply(self, _print):
        """Her first sentence comes back while she says her third: the
        reference is the reply, not the sentence being said."""
        listener, _ = _create_listener(tts_speaking=True)
        listener.echo_detector.track_tts_start(_HER_FIRST_SENTENCE, baseline_energy=_FLOOR)
        for sentence in (_HER_MIDDLE_SENTENCE, _HER_LAST_SENTENCE):
            listener.echo_detector.track_tts_start(sentence, continues=True)

        _whisper_hears(listener, [_HER_FIRST_SENTENCE, _HIS_QUESTION],
                       started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

        query = _accepted_query(listener)
        assert "nantes" in query.lower()
        assert not _her_words_in(query)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_mlx_whisper_segments_are_read_the_same(self, _print):
        listener = _while_she_speaks()
        cfg = listener.cfg
        cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
        cfg.whisper_no_speech_threshold = 0.5
        listener._whisper_backend = "mlx"
        listener.echo_detector.track_utterance_timing(time.time() - 3.0, 0.0)
        listener._frame_captured_at = time.time()
        listener._utterance_frames = [
            np.full(listener._frame_samples, _level(listener, _HIS_VOICE), dtype=np.float32)
            for _ in range(50)]
        listener.is_speech_active = True
        mlx = MagicMock()
        mlx.transcribe.return_value = {
            "text": f"{_HER_MIDDLE_SENTENCE} {_HIS_QUESTION}", "language": "fr",
            "segments": [{"text": f" {piece}", "avg_logprob": -0.1, "no_speech_prob": 0.01}
                         for piece in (_HER_MIDDLE_SENTENCE, _HIS_QUESTION)]}

        with patch("jarvis.listening.listener.mlx_whisper", mlx, create=True):
            listener._finalize_utterance()

        query = _accepted_query(listener)
        assert "nantes" in query.lower()
        assert not _her_words_in(query)
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

        shown = _current_line_shown_to(judge)
        assert "nantes" in shown.lower()
        assert not _her_words_in(shown)
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
        assert not _her_words_in(kept[0])
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
    def test_words_nobody_said_to_her_after_her_sentences_are_not_answered(self, _print):
        """While she speaks, her sentences and then a television, or a
        phrase Whisper made of the room. Once her sentences are out, the
        rest stands as words she did not say; it carries no wake word, and
        nothing is made of it."""
        listener = _while_she_speaks()

        _whisper_hears(listener,
                       [_HER_FIRST_SENTENCE, _HER_MIDDLE_SENTENCE, _WORDS_NOBODY_SAID_TO_HER],
                       started=time.time() - 3.0, level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
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
        """A word she also said ("Monsieur.") is too little to tell."""
        heard = ["Monsieur.", _HIS_QUESTION]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_three_words_of_hers_in_a_row_are_too_little_to_tell(self):
        """After a long reply, three words she said in that order are words
        he can say too."""
        heard = ["de onze heures.", _HIS_QUESTION]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_her_short_reply_and_his_first_words_in_one_piece_stay(self):
        """Whisper made one segment of her whole reply and the start of his
        question. Her reply is a run of that piece, not the piece a run of
        her reply: taken out, his question would go with it."""
        heard = [_HER_SHORT_REPLY_THEN_HIS_WORDS, "Réponds-moi vite s'il te plaît."]
        assert (_without_her_sentences_at_the_edges(heard, _HER_SHORT_REPLY, _MIN_WORDS)
                == (heard, []))

    def test_her_short_reply_alone_in_a_piece_is_taken_out(self):
        heard = [_HER_SHORT_REPLY, _HIS_QUESTION]
        assert (_without_her_sentences_at_the_edges(heard, _HER_SHORT_REPLY, _MIN_WORDS)
                == ([_HIS_QUESTION], [_HER_SHORT_REPLY]))

    def test_what_is_left_must_carry_words_she_did_not_say(self):
        """Her sentences around words of hers Whisper decimated: nothing of
        his is in it, and the utterance stays whole for the echo checks."""
        heard = [_HER_FIRST_SENTENCE, _HER_MIDDLE_SENTENCE_DECIMATED, _HER_LAST_SENTENCE]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_what_is_left_needs_as_many_words_as_any_salvage(self):
        words = ["Jarvis,", "regarde", "encore", "une", "fois", "ce", "soir"]
        enough = " ".join(words[:_MIN_WORDS])
        too_few = " ".join(words[:_MIN_WORDS - 1])

        assert (_without_her_sentences_at_the_edges(
                    [_HER_MIDDLE_SENTENCE, enough], _HER_REPLY, _MIN_WORDS)
                == ([enough], [_HER_MIDDLE_SENTENCE]))
        heard = [_HER_MIDDLE_SENTENCE, too_few]
        assert _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS) == (heard, [])

    def test_it_says_what_it_took_out(self):
        heard = [_HER_FIRST_SENTENCE, _HIS_QUESTION, _HER_LAST_SENTENCE]
        his, hers = _without_her_sentences_at_the_edges(heard, _HER_REPLY, _MIN_WORDS)
        assert his == [_HIS_QUESTION]
        assert hers == [_HER_FIRST_SENTENCE, _HER_LAST_SENTENCE]
