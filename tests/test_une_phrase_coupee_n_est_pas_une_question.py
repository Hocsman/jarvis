"""Half a sentence should wait for its other half.

Field trace (2026-08-16). The voice activity detector cut mid-sentence
and produced two utterances:

    "Ah d'accord le 23 août ça marche.  Quel est pour toi..."
    "baisser les prix de la rame"

The judge read the first as directed and returned "Quel est pour toi" —
four words, no object — and a whole reply turn ran on it. She answered by
asking what on earth he meant, which was the only honest thing to do with
half a question.

A minimum-length guard cannot fix this, and not merely imperfectly: it is
the wrong axis. The fragment is 17 characters and four words. Every
legitimate short query is shorter — "quelle heure" is 12, "et demain ?"
is 11, "tell me more" is 12 and the judge's own prompt names it as a
valid follow-up. A threshold catching the fragment kills all three.

The signal is not length, it is truncation, and Whisper writes it down:
the transcript ends in an ellipsis. Paired with a judge query that is
just the tail of that transcript — meaning the judge echoed the fragment
rather than resolving it into a question — that is a sentence still being
spoken.

Held, not dropped: the segment stays unprocessed so the next utterance's
judge call sees both halves and can compose the whole question.
"""

from __future__ import annotations

import pytest

from src.jarvis.listening.listener import _query_is_an_unfinished_fragment


COUPEE = "Ah d'accord le 23 août ça marche.  Quel est pour toi..."


def test_the_trailing_fragment_of_a_cut_sentence_is_held():
    assert _query_is_an_unfinished_fragment(COUPEE, "Quel est pour toi") is True


def test_a_unicode_ellipsis_counts_too():
    assert _query_is_an_unfinished_fragment("Quel est pour toi…",
                                            "Quel est pour toi") is True


def test_a_finished_sentence_is_never_held():
    """The control that matters most: this runs on every hot-window turn,
    and holding a real question costs him the turn."""
    assert _query_is_an_unfinished_fragment("Yuba, quelle heure est-il ?",
                                            "quelle heure est-il") is False


@pytest.mark.parametrize("court", ["quelle heure", "et demain ?",
                                   "tell me more", "stop"])
def test_the_short_queries_a_length_guard_would_have_killed_survive(court):
    assert _query_is_an_unfinished_fragment(court, court) is False


def test_a_judge_that_resolved_the_question_is_not_held():
    """An ellipsis alone is not enough. When the judge names the topic
    from earlier segments — which its prompt tells it to do — the query
    is no longer the tail of the transcript and the turn is real."""
    assert _query_is_an_unfinished_fragment(
        "et le prix, tu en penses quoi...",
        "que penses-tu du prix du Kestrel M3") is False


def test_case_and_spacing_do_not_defeat_the_comparison():
    assert _query_is_an_unfinished_fragment("Bon.  QUEL EST POUR TOI...",
                                            "quel est   pour toi") is True


def test_an_empty_query_is_not_a_fragment_decision():
    """Empty is another branch's problem; this guard must not claim it."""
    assert _query_is_an_unfinished_fragment(COUPEE, "") is False
