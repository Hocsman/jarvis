"""The rewrite may drop a figure. It may not invent one.

Measured on his graph, 2026-08-17. A summary said the Kestrel M3 lasts
19 hours on battery; the extractor got it right; the merge wrote it down
as 26. Three repeats of the same merge came back correct, so it is not
systematic — but it happened, and nothing saw it.

The existing hallucination guard cannot: it bounds how many *lines* the
rewrite may return against how many went in. A digit changed inside a
line passes it by construction, and so does a fact quietly dropped.

Numbers are where a rewrite does the most damage per character. A
reworded sentence is still roughly the fact; a wrong figure is a
different fact wearing the same words, and it is the shape a later
answer will quote with confidence.

So: every digit run in the rewrite must already appear somewhere in what
went in. Dropping is allowed — consolidation is meant to shrink. Adding
is not.

Rejection is safe. A refused merge falls back to plain append, so the
cost of a false positive is less consolidation, never a lost fact.
"""

from __future__ import annotations

import pytest

from src.jarvis.memory.graph_ops import _numbers_are_grounded


ENTREE = [
    "The Kestrel M3 case weighs 412 grams",
    "The Kestrel M3 case has a battery life of 19 hours",
]


def test_the_figure_the_merge_invented_is_caught():
    """The observed failure, stated as a test."""
    reecriture = ["The Kestrel M3 case weighs 412 grams and has a battery "
                  "life of 26 hours."]

    assert _numbers_are_grounded(reecriture, ENTREE) is False


def test_a_faithful_consolidation_passes():
    reecriture = ["The Kestrel M3 case weighs 412 grams and has a 19-hour "
                  "battery life."]

    assert _numbers_are_grounded(reecriture, ENTREE) is True


def test_dropping_a_figure_is_allowed():
    """Consolidation is meant to shrink. Only invention is refused."""
    reecriture = ["The Kestrel M3 case weighs 412 grams."]

    assert _numbers_are_grounded(reecriture, ENTREE) is True


def test_reformatting_around_the_digits_is_allowed():
    """Punctuation, units and word order move freely; the digits do not."""
    reecriture = ["Kestrel M3: 412 g, 19 h autonomie."]

    assert _numbers_are_grounded(reecriture, ENTREE) is True


def test_a_rewrite_with_no_numbers_at_all_passes():
    """The control that keeps the guard out of the way of ordinary text."""
    assert _numbers_are_grounded(
        ["Le Kestrel M3 est un boîtier compact."], ENTREE) is True


def test_a_date_already_present_is_not_an_invention():
    """Existing lines carry their provenance suffix, dates included, and
    the rewrite may echo them."""
    entree = ["Le Vela X2 sort en mars 2026 · web · 2026-08-20"]

    assert _numbers_are_grounded(["Le Vela X2 sort en mars 2026."],
                                 entree) is True


def test_a_date_that_was_never_there_is_an_invention():
    assert _numbers_are_grounded(["Le Vela X2 sort en mars 2027."],
                                 ["Le Vela X2 sort en mars 2026"]) is False


def test_a_digit_hidden_inside_a_word_still_counts():
    """`gpt-oss-120b` and `24.5` are figures too; the check reads digit
    runs, not tokens that happen to look numeric."""
    assert _numbers_are_grounded(["Le modèle gpt-oss-120b répond vite."],
                                 ["Le modèle gpt-oss-120b est rapide"]) is True
    assert _numbers_are_grounded(["Le modèle gpt-oss-200b répond vite."],
                                 ["Le modèle gpt-oss-120b est rapide"]) is False


def test_nothing_in_means_nothing_to_ground_against():
    """No input, no opinion — the caller's other guards decide."""
    assert _numbers_are_grounded(["42 lignes"], []) is False
    assert _numbers_are_grounded(["aucun chiffre"], []) is True
