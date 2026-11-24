"""Her words coming back as she said them, or his answer made of them.

His answer to the question she asks before acting is made of that question's
words: "Je m'apprête à lancer archivePhotos. Tu valides ?" is answered "je
valide", and both its words are hers. What sets it apart from her question
coming back is their order: her voice coming back keeps her order, a word
Whisper drops here and there aside, where his answer picks her words out of
it ("je", then "valide", five of hers later). And it is a few words: her
question coming back loses words, sometimes enough to break her order, but
keeps most of them, the name of what she is about to do among them.

The tool names here are invented.
"""

from __future__ import annotations

import re

import pytest

from jarvis.listening.echo_detection import (
    ANSWER_IN_HER_WORDS, ORDER_GAP, comes_back_in_her_order, picks_her_words_out,
)
from jarvis.tools.confirmation import describe_action
from jarvis.tools.policy import RISK_ACTION


_HER_ACTION = "archivePhotos"
_HER_QUESTION = describe_action(_HER_ACTION, {}, RISK_ACTION).spoken
_HER_QUESTION_EN = "I'm about to empty the recycle bin. Do you approve?"


def _her_tokens(spoken: str) -> list:
    """Her words as the order test counts them."""
    return re.findall(r"\w+", spoken, re.UNICODE)


# ── Her order ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("heard, spoken", [
    (_HER_QUESTION, _HER_QUESTION),
    ("Je m'apprête à lancer archive photos, tu valides ?", _HER_QUESTION),
    ("tu valides", _HER_QUESTION),
    ("lancer archive photos", _HER_QUESTION),
    ("Lance.", _HER_QUESTION),
    ("Tu valides, valides ?", _HER_QUESTION),
    ("Do you approve?", _HER_QUESTION_EN),
    ("Approve.", _HER_QUESTION_EN),
], ids=["whole", "as-whisper-writes-it-heard", "her-last-words", "the-name-in-parts",
        "her-verb-alone", "a-word-heard-twice", "english-tail", "english-verb-alone"])
def test_her_words_in_her_order_come_back_as_she_said_them(heard, spoken):
    assert comes_back_in_her_order(heard, spoken)


def test_a_short_word_whisper_dropped_is_still_her_order():
    """Whisper drops a short word of hers now and then ("à")."""
    assert comes_back_in_her_order("Je m'apprête lancer archive photos, tu valides ?",
                                   _HER_QUESTION)


def test_her_order_allows_order_gap_of_her_words_missing_and_no_more():
    tokens = _her_tokens(_HER_QUESTION)
    missing = " ".join(tokens[:2] + tokens[2 + ORDER_GAP:])
    one_more = " ".join(tokens[:2] + tokens[3 + ORDER_GAP:])

    assert comes_back_in_her_order(missing, _HER_QUESTION)
    assert not comes_back_in_her_order(one_more, _HER_QUESTION)


@pytest.mark.parametrize("heard, spoken", [
    ("Je valide.", _HER_QUESTION),
    ("Je lance.", _HER_QUESTION),
    ("Valides tu ?", _HER_QUESTION),
    ("I approve.", _HER_QUESTION_EN),
], ids=["je-valide", "je-lance", "reversed", "english"])
def test_her_words_picked_out_of_her_order_are_not_her_order(heard, spoken):
    assert not comes_back_in_her_order(heard, spoken)


def test_a_word_she_did_not_say_is_not_her_words_coming_back():
    assert not comes_back_in_her_order("Oui, tu valides.", _HER_QUESTION)


def test_nothing_heard_is_nothing_of_his():
    assert comes_back_in_her_order("", _HER_QUESTION)


def test_with_nothing_of_hers_everything_is_his():
    """Fails open, as the word test does: with nothing of hers to compare
    against, nothing heard can be her coming back."""
    assert not comes_back_in_her_order("Je valide.", "")


# ── His answer picking her words out ──────────────────────────────────


@pytest.mark.parametrize("heard, spoken, action", [
    ("Je valide.", _HER_QUESTION, _HER_ACTION),
    ("Je lance.", _HER_QUESTION, _HER_ACTION),
    ("Je valide, lance.", _HER_QUESTION, _HER_ACTION),
    ("I approve.", _HER_QUESTION_EN, ""),
    ("I do approve.", _HER_QUESTION_EN, ""),
], ids=["je-valide", "je-lance", "both", "english", "english-stressed"])
def test_a_few_of_her_words_out_of_her_order_are_his_answer(heard, spoken, action):
    assert picks_her_words_out(heard, spoken, action=action)


@pytest.mark.parametrize("heard", [
    "Je à lancer archive photos.",
    "Je lancer archive photos.",
    "Je lance archive photos.",
    "Lancer archive photos lancer.",
], ids=["her-two-word-verb-dropped", "two-words-dropped", "bent-and-dropped", "repeated"])
def test_her_question_coming_back_out_of_her_order_is_not_his_answer(heard):
    """Whisper drops or repeats words of her question as it comes back, and
    her order breaks; most of her question is still there, the name of what
    she is about to do with it."""
    assert not comes_back_in_her_order(heard, _HER_QUESTION)
    assert not picks_her_words_out(heard, _HER_QUESTION, action=_HER_ACTION)


def test_naming_what_she_is_about_to_do_is_not_picking_her_words_out():
    """Her action's name is what her question says; an answer of a few of
    her words that names it could be her question coming back short."""
    heard = "Je lancer archive."
    assert len(_her_tokens(heard)) <= ANSWER_IN_HER_WORDS
    assert not comes_back_in_her_order(heard, _HER_QUESTION)

    assert not picks_her_words_out(heard, _HER_QUESTION, action=_HER_ACTION)


def test_her_question_s_tail_heard_twice_is_not_his_answer():
    """Whisper repeats a phrase of her question as it comes back: four of
    her words, none naming what she is about to do, out of her order, are
    more than his answer takes."""
    heard = "Tu valides, tu valides ?"
    assert not comes_back_in_her_order(heard, _HER_QUESTION)

    assert not picks_her_words_out(heard, _HER_QUESTION, action=_HER_ACTION)


def test_more_of_her_words_than_an_answer_takes_are_not_his_answer():
    tokens = _her_tokens(_HER_QUESTION)
    reversed_words = " ".join(reversed(tokens[:ANSWER_IN_HER_WORDS + 1]))
    assert not comes_back_in_her_order(reversed_words, _HER_QUESTION)

    assert not picks_her_words_out(reversed_words, _HER_QUESTION, action=_HER_ACTION)


@pytest.mark.parametrize("heard", ["Tu valides ?", "Lance.", "lancer archive photos", ""],
                         ids=["her-tail", "her-verb", "in-her-order", "nothing"])
def test_her_words_in_her_order_are_not_his_answer(heard):
    assert not picks_her_words_out(heard, _HER_QUESTION, action=_HER_ACTION)


def test_a_word_she_did_not_say_is_not_an_answer_made_of_her_words():
    """That proof is the word test's (`carries_a_word_she_did_not_say`)."""
    assert not picks_her_words_out("Oui, je valide.", _HER_QUESTION, action=_HER_ACTION)


def test_his_answer_after_her_question_coming_back_in_one_transcript():
    """Whisper merges her question's tail and his answer into one
    transcript; his sentence in it picks her words out."""
    assert picks_her_words_out("Tu valides ? Je valide.", _HER_QUESTION, action=_HER_ACTION)


def test_a_one_word_action_named_is_not_picking_her_words_out():
    """The tool her question names may be one word, which nothing in her
    question's spelling sets apart from the rest: it is named as her
    question names it."""
    spoken = describe_action("scanner", {}, RISK_ACTION).spoken

    assert not picks_her_words_out("Je lancer scanner.", spoken, action="scanner")
    assert picks_her_words_out("Je valide.", spoken, action="scanner")
