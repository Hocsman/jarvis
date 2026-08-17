"""Whisper loops on phrases, not only on words.

Field trace (2026-08-17, air conditioning running). Three times in one
session, on non-speech audio:

    📝 Heard: "There we go.  There we go.  There we go.  …"  ×13

The existing guard counts repeated *words*. This is a repeated *phrase*,
and neither of its two rules reaches it: "There" is 13 of 39 tokens, a
share of 0.33 against a bar of 0.50, and no word ever appears twice in a
row because two others sit between each repetition. Measured, not
assumed. "don don don" is caught; "There we go. There we go." is not.

Each one that slips through costs an intent-judge call on the cloud model
to be told it is not directed, and nothing guarantees the judge always
says no.

The fix is the same idea one dimension up: a phrase repeated back to back
is the same signal as a word repeated back to back. What must not change
is the other direction — real speech repeats words and phrases all the
time, and a guard that ate "oui oui, vas-y" or a list of three items
would cost him sentences.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.jarvis.listening.listener import VoiceListener


def _hallucination(texte: str) -> bool:
    return VoiceListener._is_repetitive_hallucination(MagicMock(), texte)


# ── Ce qui doit être rejeté ────────────────────────────────────────────


def test_the_phrase_from_the_trace_is_caught():
    assert _hallucination(" ".join(["There we go."] * 13)) is True


def test_a_phrase_repeated_only_a_few_times_is_caught_too():
    assert _hallucination("Merci beaucoup. Merci beaucoup. Merci beaucoup. "
                          "Merci beaucoup.") is True


def test_a_longer_looping_phrase_is_caught():
    assert _hallucination(
        "Sous-titres réalisés par la communauté d'Amara.org "
        "Sous-titres réalisés par la communauté d'Amara.org "
        "Sous-titres réalisés par la communauté d'Amara.org") is True


def test_the_single_word_loop_it_already_caught_still_is():
    """The control for the old behaviour: widening must not narrow."""
    assert _hallucination("don don don don don don") is True
    assert _hallucination(" ".join(["Z"] * 120)) is True


# ── Ce qui doit passer ─────────────────────────────────────────────────


def test_ordinary_speech_survives():
    assert _hallucination(
        "Est-ce que si j'ai des lunettes connectées du style Meta, "
        "Ray-Ban ou autre, tu pourrais accéder à la caméra ?") is False


def test_a_person_repeating_themselves_is_not_a_hallucination():
    """He says things twice. That is speech, not a loop."""
    assert _hallucination("oui oui, vas-y") is False
    assert _hallucination("attends attends, répète") is False


def test_a_list_of_similar_items_survives():
    """Enumeration repeats structure without being a loop."""
    assert _hallucination(
        "j'ai acheté du pain, du lait, du beurre et du café") is False


def test_a_short_phrase_said_twice_is_not_enough():
    """Two is emphasis; a loop is more than that. Set the bar too low and
    it eats him mid-sentence."""
    assert _hallucination("c'est bon, c'est bon") is False


def test_an_empty_or_tiny_utterance_is_not_judged():
    assert _hallucination("") is False
    assert _hallucination("oui") is False
