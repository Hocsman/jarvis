"""Cutting a stream of tokens into sentences worth speaking.

She writes the whole reply, then starts speaking it. On a three-sentence
answer that is several seconds of silence where a person would already be
talking, and the point of the whole exercise is that the exchange should
feel like one with someone.

This is the piece that decides where a chunk ends. It sees token deltas
arriving and returns each sentence as it closes.

Two rules, and the second is what keeps it usable. A sentence closes on
terminal punctuation followed by whitespace — orthographic, not lexical,
so no word list and no language detection. And a chunk shorter than a
handful of characters waits for the next one, which costs nothing and
quietly absorbs the abbreviation problem: "M." on its own is two
characters, so it merges forward instead of becoming a chunk that says
"M".

See `src/jarvis/output/streaming.spec.md`.
"""

from __future__ import annotations

import pytest

from src.jarvis.output.streaming import SentenceStreamer


def _tout(morceaux, taille=7):
    """Feed a text in fixed-size slices, the way deltas actually arrive."""
    s = SentenceStreamer()
    sorties = []
    texte = "".join(morceaux) if isinstance(morceaux, list) else morceaux
    for i in range(0, len(texte), taille):
        sorties.extend(s.feed(texte[i:i + taille]))
    reste = s.flush()
    if reste:
        sorties.append(reste)
    return sorties


# ── Ce qui sort, et quand ──────────────────────────────────────────────


def test_three_sentences_come_out_as_three_chunks():
    phrases = _tout("Il fait beau à Bagneux aujourd'hui. "
                    "La température atteint vingt-quatre degrés. "
                    "Prends une veste quand même.")

    assert len(phrases) == 3
    assert phrases[0].startswith("Il fait beau")
    assert phrases[2].startswith("Prends une veste")


def test_a_sentence_is_emitted_before_the_rest_arrives():
    """The whole point: the first chunk must leave before the model has
    finished writing the second."""
    s = SentenceStreamer()

    sorties = s.feed("Il fait beau à Bagneux aujourd'hui. Et demain")

    assert sorties == ["Il fait beau à Bagneux aujourd'hui."]


def test_nothing_comes_out_of_an_unfinished_sentence():
    s = SentenceStreamer()

    assert s.feed("Il fait beau à Bagneux") == []


def test_the_tail_is_spoken_even_without_final_punctuation():
    phrases = _tout("Il fait beau à Bagneux aujourd'hui. Et demain aussi")

    assert phrases[-1] == "Et demain aussi"


# ── Ce qui ne doit pas être coupé ──────────────────────────────────────


def test_an_abbreviation_does_not_become_its_own_chunk():
    """A chunk of "M." would be synthesised as a letter. Too short to be
    worth a call, so it merges forward."""
    phrases = _tout("M. Dupont arrive à huit heures.")

    assert phrases == ["M. Dupont arrive à huit heures."]


def test_a_decimal_number_survives():
    phrases = _tout("Il fait 24.5 degrés dehors en ce moment.")

    assert phrases == ["Il fait 24.5 degrés dehors en ce moment."]


# ── Le texte livré est le texte écrit ──────────────────────────────────


@pytest.mark.parametrize("taille", [1, 3, 7, 50])
def test_the_assembled_text_is_the_text_whatever_the_delta_size(taille):
    """The control that matters most. Streaming is a delivery detail; a
    single character lost or added would make what she said differ from
    what was written down."""
    texte = ("Il fait beau à Bagneux. La température atteint 24.5 degrés ! "
             "Tu sors ? Prends une veste…")

    assert " ".join(_tout(texte, taille)) == " ".join(texte.split())


def test_it_does_not_key_on_one_script():
    """Full-width punctuation closes a sentence too, so the rule does not
    privilege a writing system."""
    phrases = _tout("今日はいい天気です。散歩に行きましょう。")

    assert len(phrases) == 2


def test_an_empty_stream_says_nothing():
    s = SentenceStreamer()

    assert s.feed("") == []
    assert s.flush() == ""
