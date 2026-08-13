"""The diary's search index holds one document per day, or it lies.

A day's summary row is rewritten on every flush. Resolving the UNIQUE
clash with `INSERT OR REPLACE` deletes and re-inserts, which hands the
row a new id each time, and SQLite skips the DELETE trigger for a REPLACE
unless `recursive_triggers` is on. The old terms stay in the index under
an id no row carries any more.

Nothing on the surface shows it. `COUNT(*)` on the index reads the
content table and answers with the number of days. FTS5's own
integrity-check passes. Only bm25 knows, and what it does with the
knowledge is put the wrong day first: the day he mentioned something once
ahead of the day he talked about it for hours.
"""

from __future__ import annotations

import pytest


def _db(tmp_path):
    from src.jarvis.memory.db import Database

    return Database(str(tmp_path / "journal.db"))


def _jour(db, date: str, texte: str, sujets: str = "") -> int:
    return db.upsert_conversation_summary(
        date_utc=date, summary=texte, topics=sujets,
    )


def test_rewriting_a_day_does_not_change_its_identity(tmp_path):
    """Everything keyed on that id — the index, the search join, the
    embedding row — is pointed at the day it describes."""
    db = _db(tmp_path)

    ids = [_jour(db, "2026-08-10", f"il a mangé des sushis, version {n}")
           for n in range(5)]

    assert len(set(ids)) == 1


def test_the_index_holds_no_more_documents_than_there_are_days(tmp_path):
    """`summaries_fts_docsize` carries one row per indexed document,
    stale ones included — the count anyone can run on their own file."""
    db = _db(tmp_path)
    for n in range(30):
        _jour(db, "2026-08-10", f"il a mangé des ramen, version {n}")
    _jour(db, "2026-08-11", "il a mangé du quinoa")

    jours = db.conn.execute(
        "SELECT COUNT(*) FROM conversation_summaries").fetchone()[0]
    indexes = db.conn.execute(
        "SELECT COUNT(*) FROM summaries_fts_docsize").fetchone()[0]

    assert indexes == jours


def test_the_day_he_talked_about_it_comes_before_the_day_he_mentioned_it(tmp_path):
    """The damage as he meets it. bm25 is the only ranking on a default
    install, and it is computed over the polluted index."""
    db = _db(tmp_path)
    for j in range(1, 29):
        _jour(db, f"2026-06-{j:02d}", "journée ordinaire, rien de particulier")

    _jour(db, "2026-07-01", "il a goûté du quinoa au déjeuner")

    for n in range(30):
        _jour(db, "2026-07-02",
              f"il a parlé de ramen toute la journée, ramen le midi, ramen le soir "
              f"et encore des ramen, version {n}")

    resultats = db.search_hybrid("quinoa ramen", None, top_k=8)

    assert resultats, "la recherche doit rendre quelque chose"
    # Rows come back as (id, score, text, result_type), the text prefixed
    # with the day it describes.
    assert "[2026-07-02]" in resultats[0]["text"]


# ── And a file written by the old shape is repaired, out loud ──────────


def _pollue(chemin, jours: int = 6):
    """Write a day the way the previous shape did, so the index inherits
    one stale entry per flush."""
    from src.jarvis.memory.db import Database

    db = Database(str(chemin))
    for n in range(jours):
        db.conn.execute(
            "INSERT OR REPLACE INTO conversation_summaries"
            "(date_utc, ts_utc, summary, topics, source_app) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-08-10", "2026-08-10T12:00:00+00:00",
             f"il a mangé des sushis, version {n}", "", "jarvis"),
        )
    db.conn.commit()
    db.close()


def test_an_inherited_index_is_repaired_at_startup_and_he_is_told(tmp_path, capsys):
    from src.jarvis.memory.db import Database

    chemin = tmp_path / "journal.db"
    _pollue(chemin)
    capsys.readouterr()

    db = Database(str(chemin))

    sortie = capsys.readouterr().out
    assert "📓" in sortie
    jours = db.conn.execute(
        "SELECT COUNT(*) FROM conversation_summaries").fetchone()[0]
    indexes = db.conn.execute(
        "SELECT COUNT(*) FROM summaries_fts_docsize").fetchone()[0]
    assert indexes == jours


def test_a_repair_that_cannot_run_is_said_and_does_not_stop_startup(tmp_path, capsys):
    """The half that keeps this honest: an index that cannot be repaired
    must not look like one that never needed it."""
    from src.jarvis.memory.db import Database

    chemin = tmp_path / "journal.db"
    _pollue(chemin)

    import sqlite3

    brut = sqlite3.connect(str(chemin))
    brut.execute("DROP TABLE summaries_fts_data")
    brut.commit()
    brut.close()
    capsys.readouterr()

    db = Database(str(chemin))

    assert db is not None
    assert "⚠️" in capsys.readouterr().out
