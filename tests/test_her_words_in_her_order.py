"""Her words coming back as she said them, or his answer made of them.

His answer to the question she asks before acting is made of that question's
words: "Je m'apprête à lancer archivePhotos. Tu valides ?" is answered "je
valide", and both its words are hers. What sets it apart from her question
coming back is their order: her voice coming back keeps her order, a word
Whisper drops here and there aside, where his answer picks her words out of
it ("je", then "valide", five of hers later).

The tool names here are invented.
"""

from __future__ import annotations

import pytest

from jarvis.listening.echo_detection import ORDER_GAP, comes_back_in_her_order
from jarvis.tools.confirmation import describe_action
from jarvis.tools.policy import RISK_ACTION


_HER_QUESTION = describe_action("archivePhotos", {}, RISK_ACTION).spoken
_HER_QUESTION_EN = "I'm about to empty the recycle bin. Do you approve?"


@pytest.mark.parametrize("heard, spoken", [
    (_HER_QUESTION, _HER_QUESTION),
    ("Je m'apprête à lancer archive photos, tu valides ?", _HER_QUESTION),
    ("tu valides", _HER_QUESTION),
    ("lancer archive photos", _HER_QUESTION),
    ("Lance.", _HER_QUESTION),
    ("Do you approve?", _HER_QUESTION_EN),
    ("Approve.", _HER_QUESTION_EN),
], ids=["whole", "as-whisper-writes-it-heard", "her-last-words", "the-name-in-parts",
        "her-verb-alone", "english-tail", "english-verb-alone"])
def test_her_words_in_her_order_come_back_as_she_said_them(heard, spoken):
    assert comes_back_in_her_order(heard, spoken)


def test_a_word_whisper_dropped_between_two_of_hers_still_comes_back_as_hers():
    """Whisper drops a short word of hers now and then. Her order is kept
    with up to `ORDER_GAP` of her words missing between two it kept."""
    words = "Je m'apprête à lancer archive photos, tu valides ?".split()
    dropped = " ".join(words[:2] + words[2 + ORDER_GAP:])

    assert comes_back_in_her_order(dropped, _HER_QUESTION)


@pytest.mark.parametrize("heard, in_her_order", [
    ("Je m'apprête lancer archive photos, tu valides ?", True),
    ("Je m'apprête archive photos, tu valides ?", False),
], ids=["one-of-hers-dropped", "two-of-hers-dropped"])
def test_one_of_her_words_may_go_missing_and_no_more(heard, in_her_order):
    """Whisper drops a short word of hers now and then ("à"): that much of
    her order is given. Two missing side by side is already more than her
    voice coming back loses."""
    assert comes_back_in_her_order(heard, _HER_QUESTION) is in_her_order


@pytest.mark.parametrize("heard, spoken", [
    ("Je valide.", _HER_QUESTION),
    ("Je lance.", _HER_QUESTION),
    ("Valides tu ?", _HER_QUESTION),
    ("I approve.", _HER_QUESTION_EN),
], ids=["je-valide", "je-lance", "reversed", "english"])
def test_her_words_picked_out_of_her_order_are_not_her_words_coming_back(heard, spoken):
    """Every word is hers, and they are not in her order: his answer taking
    them up, not her voice coming back."""
    assert not comes_back_in_her_order(heard, spoken)


def test_more_of_her_words_missing_than_whisper_drops_is_not_her_order():
    words = "Je m'apprête à lancer archive photos, tu valides ?".split()
    skipped = " ".join(words[:1] + words[2 + ORDER_GAP:])

    assert not comes_back_in_her_order(skipped, _HER_QUESTION)


def test_a_word_she_did_not_say_is_not_her_words_coming_back():
    assert not comes_back_in_her_order("Oui, tu valides.", _HER_QUESTION)


def test_nothing_heard_is_nothing_of_his():
    assert comes_back_in_her_order("", _HER_QUESTION)


def test_with_nothing_of_hers_everything_is_his():
    """Fails open, as the word test does: with nothing of hers to compare
    against, nothing heard can be her coming back."""
    assert not comes_back_in_her_order("Je valide.", "")
