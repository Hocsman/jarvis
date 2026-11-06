"""
Tests for wake word detection and query extraction.
"""

import re
import unicodedata

import pytest

from jarvis.listening.wake_detection import (
    is_wake_word_detected,
    extract_query_after_wake,
    is_stop_command,
)


def _words(text: str) -> list:
    """The words of *text*, its punctuation left out."""
    return re.findall(r"\w+", text)


@pytest.mark.unit
class TestWakeWordDetection:
    """Tests for is_wake_word_detected."""

    def test_exact_match(self):
        assert is_wake_word_detected("hey jarvis", "jarvis", []) is True

    def test_alias_match(self):
        assert is_wake_word_detected("hey computer", "jarvis", ["computer"]) is True

    def test_no_match(self):
        assert is_wake_word_detected("hello world", "jarvis", []) is False

    def test_empty_text(self):
        assert is_wake_word_detected("", "jarvis", []) is False

    def test_fuzzy_match(self):
        """Fuzzy matching catches slight transcription errors."""
        assert is_wake_word_detected("hey jarvas", "jarvis", [], fuzzy_ratio=0.78) is True

    def test_fuzzy_below_threshold(self):
        """Completely different word doesn't fuzzy-match."""
        assert is_wake_word_detected("hey banana", "jarvis", [], fuzzy_ratio=0.78) is False


@pytest.mark.unit
class TestExtractQueryAfterWake:
    """Tests for extract_query_after_wake."""

    def test_extracts_query(self):
        result = extract_query_after_wake("jarvis what time is it", "jarvis", [])
        assert result == "what time is it"

    def test_extracts_query_with_alias(self):
        result = extract_query_after_wake("hey computer what time is it", "jarvis", ["hey computer"])
        assert result == "what time is it"

    def test_wake_word_only_returns_empty(self):
        """When only the wake word is said, return empty string (no hardcoded fallback)."""
        result = extract_query_after_wake("jarvis", "jarvis", [])
        assert result == ""

    def test_wake_word_with_punctuation_only_returns_empty(self):
        """Wake word followed by just punctuation returns empty string."""
        result = extract_query_after_wake("jarvis,", "jarvis", [])
        assert result == ""

    def test_empty_text(self):
        result = extract_query_after_wake("", "jarvis", [])
        assert result == ""

    def test_strips_leading_punctuation(self):
        result = extract_query_after_wake("jarvis, tell me a joke", "jarvis", [])
        assert result == "tell me a joke"


@pytest.mark.unit
class TestHerNameIsAWord:
    """The wake word and each alias are names, found only as whole words of
    what was heard: never inside a longer word, whatever the alphabet.
    Taking her name out of a query takes out the words that matched, and
    leaves every other word as it was heard."""

    def test_an_alias_inside_a_longer_word_is_not_her_name(self):
        assert is_wake_word_detected("est-elle charismatique ?", "jarvis", ["charis"]) is False

    def test_the_wake_word_inside_a_longer_word_is_not_her_name(self):
        assert is_wake_word_detected("quel est le maximum ?", "max", []) is False

    def test_taking_her_name_out_leaves_a_word_holding_an_alias_whole(self):
        assert (extract_query_after_wake("jarvis, est-elle charismatique ?", "jarvis", ["charis"])
                == "est-elle charismatique ?")

    def test_taking_her_name_out_leaves_a_word_holding_it_whole(self):
        assert extract_query_after_wake("max, quel est le maximum ?", "max", []) == "quel est le maximum ?"

    @pytest.mark.parametrize("heard", ["jar is, quelle heure est-il ?",
                                       "jar  is, quelle heure est-il ?"],
                             ids=["one-space", "two-spaces"])
    def test_a_name_of_two_words_is_found_and_taken_out(self, heard):
        assert is_wake_word_detected(heard, "jarvis", ["jar is"]) is True
        assert extract_query_after_wake(heard, "jarvis", ["jar is"]) == "quelle heure est-il ?"

    def test_a_name_of_two_words_is_not_found_when_its_last_word_starts_a_longer_one(self):
        assert is_wake_word_detected("le jar isolé crie.", "jarvis", ["jar is"]) is False

    @pytest.mark.parametrize("form", ["NFC", "NFD"])
    def test_a_letter_of_any_alphabet_makes_the_word_longer(self, form):
        """"Mañana" holds "ana" after an "ñ", a letter, whether it is written
        as one character or as an "n" and a combining tilde."""
        question = unicodedata.normalize(form, "¿qué tiempo hará mañana?")

        assert is_wake_word_detected(question, "ana", []) is False
        assert extract_query_after_wake(f"ana, {question}", "ana", []) == question

    def test_her_name_in_another_alphabet_is_found_and_taken_out(self):
        heard = "джарвис, который час?"

        assert is_wake_word_detected(heard, "джарвис", []) is True
        assert extract_query_after_wake(heard, "джарвис", []) == "который час?"

    @pytest.mark.parametrize("heard, name, query", [
        ("贾维斯现在几点", "贾维斯", "现在几点"),
        ("ジャービス今何時", "ジャービス", "今何時"),
        ("จาร์วิสตอนนี้กี่โมง", "จาร์วิส", "ตอนนี้กี่โมง"),
        ("jarvis现在几点", "jarvis", "现在几点"),
    ], ids=["chinese", "japanese", "thai", "her-latin-name-in-chinese"])
    def test_in_a_script_written_without_spaces_her_name_is_found_among_the_words(
            self, heard, name, query):
        """Chinese, Japanese and Thai write a sentence without spaces between
        its words, so nothing in the text says where a word ends: her name
        is found among the words around it, as it always was."""
        assert is_wake_word_detected(heard, name, []) is True
        assert extract_query_after_wake(heard, name, []) == query

    @pytest.mark.parametrize("heard", [
        "{name}, quelle heure est-il ?",
        "quelle heure est-il, {name} ?",
        "«{name}», quelle heure est-il ?",
        "{name}! quelle heure est-il ?",
        "quelle heure est-il ({name}) ?",
        "{name}, quelle heure est-il, {name} ?",
    ], ids=["first", "last", "quoted", "exclaimed", "in-brackets", "twice"])
    @pytest.mark.parametrize("name", ["jarvis", "jervis"], ids=["wake-word", "alias"])
    def test_punctuation_around_her_name_leaves_it_her_name(self, heard, name):
        heard = heard.format(name=name)

        assert is_wake_word_detected(heard, "jarvis", ["jervis"]) is True
        assert _words(extract_query_after_wake(heard, "jarvis", ["jervis"])) == _words(
            "quelle heure est-il")

    def test_an_empty_name_is_no_name(self):
        """A name configured empty is found nowhere, rather than everywhere."""
        assert is_wake_word_detected("quelle heure est-il", "", [""]) is False
        assert extract_query_after_wake("quelle heure est-il ?", "", [" "]) == "quelle heure est-il ?"

    @pytest.mark.parametrize("heard", [
        "quelle heure est-il ?",
        "il fait beau ...",
        "bon - alors !",
        "« oui »",
    ], ids=["question-mark", "ellipsis", "dash", "guillemets"])
    @pytest.mark.parametrize("wake_word, aliases", [("", []), (" ", [""]), ("jarvis", [""])],
                             ids=["empty-wake-word", "blank-names", "empty-alias"])
    def test_an_empty_name_is_no_close_spelling_of_a_lone_mark(self, heard, wake_word, aliases):
        """French writes a space before "?" and "!", so a mark stands alone as
        a token; stripped of its punctuation it is nothing, and nothing is no
        close spelling of a name configured empty."""
        assert is_wake_word_detected(heard, wake_word, aliases) is False


@pytest.mark.unit
class TestCloseSpellings:
    """Failing a whole-word match, a token stripped of the punctuation at its
    ends is weighed against each name, whatever the punctuation's script."""

    @pytest.mark.parametrize("heard", [
        "¿jarvas, qué hora es?",
        "«jarvas», quelle heure est-il ?",
        "„jarvas“, wie spät ist es?",
        "jarvas… quelle heure est-il ?",
        "「jarvas」",
    ], ids=["inverted-question-mark", "guillemets", "german-quotes", "ellipsis", "corner-brackets"])
    def test_punctuation_of_any_script_around_a_close_spelling_leaves_it_her_name(self, heard):
        assert is_wake_word_detected(heard, "jarvis", []) is True

    def test_a_word_far_from_every_name_stays_a_word_once_stripped(self):
        assert is_wake_word_detected("«banane», quelle heure est-il ?", "jarvis", []) is False


@pytest.mark.unit
class TestStopCommand:
    """Tests for is_stop_command."""

    def test_exact_stop_command(self):
        assert is_stop_command("stop", ["stop", "quiet"]) is True

    def test_stop_command_in_phrase(self):
        assert is_stop_command("please stop talking", ["stop", "quiet"]) is True

    def test_no_stop_command(self):
        assert is_stop_command("what is the weather", ["stop", "quiet"]) is False

    def test_empty_text(self):
        assert is_stop_command("", ["stop", "quiet"]) is False

    def test_fuzzy_stop_command(self):
        """Short input fuzzy-matches stop commands."""
        assert is_stop_command("stob", ["stop", "quiet"], fuzzy_ratio=0.7) is True
