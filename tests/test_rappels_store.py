"""Where a promise is kept between now and then.

A pending confirmation is deliberately never written to disk: an approval
that outlives the process was given without the context that produced it.
A reminder is the exact opposite. If a restart loses it, it was never a
reminder — so this is the one part of the assistant that must survive the
machine going down, and the table is the whole reason it can.

Reaching existing installs costs no migration code: every statement in
the schema is CREATE TABLE IF NOT EXISTS and the whole script runs on
every open, so a new table arrives by the same path on a fresh database
and on one written months ago. `_migrate` stays untouched, which matters
because it is wired to one table and carries the project's only migration
test.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.memory.db import Database


def _iso(offset_minutes: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).isoformat()


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _add(db, **kw):
    payload = dict(
        texte="appeler le comptable",
        due_utc=_iso(-1),
        due_local="2026-08-01T09:00",
        tz="Europe/Paris",
        origin="voix",
        query="rappelle-moi d'appeler le comptable",
    )
    payload.update(kw)
    return db.add_rappel(**payload)


# ── It survives the machine ───────────────────────────────────────────


def test_a_reminder_outlives_the_process(tmp_path):
    """The property the whole subsystem rests on."""
    first = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    rid = _add(first)
    first.close()

    second = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    try:
        assert [r["id"] for r in second.pending_rappels()] == [rid]
    finally:
        second.close()


def test_the_table_reaches_a_database_written_before_it_existed(tmp_path):
    """No migration code: the schema script runs on every open and every
    statement in it is IF NOT EXISTS."""
    path = tmp_path / "ancien.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE action_log (
          id INTEGER PRIMARY KEY, ts_utc TEXT NOT NULL, origin TEXT,
          tool TEXT NOT NULL, args TEXT, risk TEXT NOT NULL,
          verdict TEXT NOT NULL, outcome TEXT NOT NULL,
          duration_ms INTEGER, query TEXT, request_id TEXT
        );
        PRAGMA user_version = 2;
        """
    )
    conn.execute(
        "INSERT INTO action_log (ts_utc, tool, risk, verdict, outcome)"
        " VALUES ('2026-07-20T10:00:00+00:00', 'webSearch', 'lecture',"
        " 'libre', 'ok')"
    )
    conn.commit()
    conn.close()

    db = Database(str(path), sqlite_vss_path=None)
    try:
        assert _add(db)
        assert len(db.recent_actions()) == 1, "an existing row was lost"
    finally:
        db.close()


def test_the_version_stamp_moves(tmp_path):
    from src.jarvis.memory.db import _SCHEMA_VERSION

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    try:
        assert db.conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
        assert _SCHEMA_VERSION >= 3
    finally:
        db.close()


def test_opening_twice_changes_nothing(tmp_path):
    first = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    _add(first)
    first.close()

    second = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    try:
        assert len(second.pending_rappels()) == 1
    finally:
        second.close()


# ── What is due ───────────────────────────────────────────────────────


def test_only_what_is_due_comes_back(db):
    _add(db, texte="passé", due_utc=_iso(-5))
    _add(db, texte="futur", due_utc=_iso(+60))

    assert [r["texte"] for r in db.due_rappels(_iso())] == ["passé"]


def test_the_oldest_due_comes_first(db):
    _add(db, texte="deuxième", due_utc=_iso(-5))
    _add(db, texte="premier", due_utc=_iso(-50))

    assert [r["texte"] for r in db.due_rappels(_iso())] == ["premier", "deuxième"]


def test_a_settled_reminder_stops_being_due(db):
    rid = _add(db)

    db.settle_rappel(rid, said_utc=_iso())

    assert db.due_rappels(_iso()) == []


def test_a_cancelled_reminder_stops_being_due(db):
    rid = _add(db)

    db.cancel_rappel(rid)

    assert db.due_rappels(_iso()) == []


def test_a_cancelled_reminder_leaves_the_pending_list(db):
    rid = _add(db)

    db.cancel_rappel(rid)

    assert db.pending_rappels() == []


def test_cancelling_an_unknown_id_changes_nothing(db):
    _add(db)

    db.cancel_rappel("pas-un-id")

    assert len(db.pending_rappels()) == 1


# ── Counting attempts, so a failure is bounded ────────────────────────


def test_an_attempt_is_recorded(db):
    rid = _add(db)

    db.mark_rappel_tried(rid, _iso())

    assert db.due_rappels(_iso())[0]["attempts"] == 1


def test_attempts_accumulate(db):
    rid = _add(db)

    db.mark_rappel_tried(rid, _iso())
    db.mark_rappel_tried(rid, _iso())

    assert db.due_rappels(_iso())[0]["attempts"] == 2


def test_a_settled_reminder_records_when_it_was_said(db):
    rid = _add(db)
    said = _iso()

    db.settle_rappel(rid, said_utc=said)

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["said_utc"] == said
    assert row["etat"] == "fini"


# ── What it holds ─────────────────────────────────────────────────────


def test_the_local_time_and_zone_are_kept_alongside_the_instant(db):
    """The instant is what fires it; the local reading is what she says
    back and what the user recognises. A timezone change between creating
    and firing must not silently rewrite either."""
    rid = _add(db, due_local="2026-08-01T09:00", tz="Europe/Paris")

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["due_local"] == "2026-08-01T09:00"
    assert row["tz"] == "Europe/Paris"


def test_the_originating_query_is_stored_redacted(db):
    """Same rule as the ledger: a reminder carries whatever the user just
    said."""
    rid = _add(db, query="rappelle-moi d'écrire à hocsman92@gmail.com")

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert "hocsman92@gmail.com" not in (row["query"] or "")


def test_the_reminder_text_is_kept_as_the_user_said_it(db):
    """Unlike the query, this is read back aloud. Redacting it would have
    her say a placeholder to the user's face."""
    rid = _add(db, texte="appeler le docteur Nguyen")

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["texte"] == "appeler le docteur Nguyen"


def test_an_id_is_handed_back_and_is_unique(db):
    first = _add(db)
    second = _add(db)

    assert first and second and first != second


def test_the_id_can_carry_into_the_ledger(db):
    """One reminder, one episode: the same id ties the firing row to the
    reminder it came from."""
    rid = _add(db)

    db.record_action(
        tool="rappel", args={}, risk="lecture", verdict="libre", outcome="ok",
        duration_ms=None, origin="rappel", query=None, request_id=rid,
    )

    assert db.recent_actions()[0]["request_id"] == rid


# ── Keeping it bounded ────────────────────────────────────────────────


def test_old_settled_reminders_are_dropped(db):
    rid = _add(db)
    db.settle_rappel(rid, said_utc=_iso())
    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    db.conn.execute("UPDATE rappels SET created_utc = ?", (old,))
    db.conn.commit()

    db.prune_rappels(max_age_days=90)

    assert db.all_rappels() == []


def test_an_old_but_unfired_reminder_is_never_dropped(db):
    """Pruning something still owed is losing a promise."""
    _add(db, due_utc=_iso(+60))
    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    db.conn.execute("UPDATE rappels SET created_utc = ?", (old,))
    db.conn.commit()

    db.prune_rappels(max_age_days=90)

    assert len(db.pending_rappels()) == 1
