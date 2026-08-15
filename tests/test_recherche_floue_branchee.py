"""The flexible query builder is actually reachable.

`_normalize_fts_query` imported `jarvis.memory.fuzzy_search`. The module
is `jarvis.utils.fuzzy_search` and always has been, so the import raised
`ImportError` on every single call, `except ImportError: pass` swallowed
it, and the fallback — a bare alphanumeric tokenise — ran every time.

Nothing failed. Searching his diary simply worked less well than it was
built to, for as long as the file has existed, and the only trace was a
`pass`.

The tests below pin behaviour rather than the import: a query with a
possessive or a plural must reach rows the bare tokenise misses. That is
what the builder is for, and pinning it that way means a future move of
the module fails here rather than falling back for another year.
"""

from __future__ import annotations

import pytest


def _db(tmp_path, resumes):
    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    for date, texte in resumes:
        db.upsert_conversation_summary(date_utc=date, summary=texte, topics="")
    return db


def test_the_module_it_imports_is_the_one_that_exists():
    """The whole defect in one line."""
    import importlib

    importlib.import_module("jarvis.utils.fuzzy_search")
    with pytest.raises(ImportError):
        importlib.import_module("jarvis.memory.fuzzy_search")


def test_the_builder_is_reached_rather_than_the_fallback():
    """The fallback returns bare tokens joined by spaces. The builder
    returns something else — whatever it is, it must not be that."""
    from src.jarvis.memory.db import _normalize_fts_query
    from src.jarvis.utils.fuzzy_search import generate_flexible_fts_query

    assert _normalize_fts_query("boxing club") == generate_flexible_fts_query(
        "boxing club")


def test_a_query_still_works_when_the_builder_gives_nothing():
    """It is allowed to decline. The fallback is the safety net, not the
    normal path."""
    from src.jarvis.memory.db import _normalize_fts_query

    assert _normalize_fts_query("   ") == ""


def test_punctuation_does_not_break_the_query(tmp_path):
    """The bare tokenise dropped everything but alphanumerics, which is
    what made it safe. Whatever replaces it must stay safe: an apostrophe
    or a quote reaching FTS5 raw is a syntax error, and a search that
    raises is worse than one that finds little."""
    from src.jarvis.memory.db import _normalize_fts_query

    db = _db(tmp_path, [("2026-08-04", "he asked about the boxing club's hours")])

    for requete in ["club's", 'say "hello"', "a - b", "*", "NEAR(", "x OR y"]:
        # The builder's output goes straight into FTS5. A search that
        # raises is worse than one that finds little.
        db.search_hybrid(_normalize_fts_query(requete), None, top_k=3)


def test_the_caller_hands_over_words_not_a_query(tmp_path):
    """The contract, pinned where it broke.

    `_normalize_fts_query` builds the FTS5 expression. A caller that
    pre-joins its keywords with " OR " is handing it syntax instead of
    words, and the builder lowercases the operator into a search term:
    "boxing OR club" became the phrase "boxing or club" and matched
    nothing at all.

    Nobody noticed for as long as the import was dead, because the
    fallback passed the operator through untouched and the accident
    worked. Turning the builder on is what surfaced the mismatch.
    """
    from src.jarvis.memory.db import _normalize_fts_query

    db = _db(tmp_path, [("2026-08-04",
                         "The user asked about the boxing club near Mare Street.")])

    assert len(db.search_hybrid("boxing club", None, top_k=5)) == 1
    assert "or " not in _normalize_fts_query("boxing club").lower().replace(" or ", " ")


def test_the_enrichment_path_finds_what_it_just_saved(tmp_path):
    """The end-to-end shape the flow tests cover, kept here too so the
    contract and its consumer are pinned in one place."""
    db = _db(tmp_path, [("2026-08-04",
                         "The user asked about the boxing club near Mare Street.")])

    for mots in (["boxing", "club"], ["mare", "street"], ["boxing"]):
        assert db.search_hybrid(" ".join(mots), None, top_k=5), mots
