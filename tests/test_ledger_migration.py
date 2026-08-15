"""The ledger's shape can change without costing the user their rows.

`action_log` is the table Yuba tells the user they can trust: it is what
she did, and the Activity tab presents it as a record. A migration that
drops rows when the shape surprises it would make that claim false the
first time the shape changes — and the shape changes here, because a
confirmation needs to correlate the question with its answer.

`CREATE TABLE IF NOT EXISTS` does not add a column to a table that
already exists, so a release that only ships new DDL would leave every
existing install writing into the old shape and failing on the new
column. The migration is what makes the version stamp mean something.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.jarvis.memory.db import Database


def _open(path):
    db = Database(str(path), sqlite_vss_path=None)
    return db


@pytest.fixture
def previous_release(tmp_path):
    """A database in the shape the previous release wrote."""
    path = tmp_path / "ancien.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE action_log (
          id          INTEGER PRIMARY KEY,
          ts_utc      TEXT NOT NULL,
          origin      TEXT,
          tool        TEXT NOT NULL,
          args        TEXT,
          risk        TEXT NOT NULL,
          verdict     TEXT NOT NULL,
          outcome     TEXT NOT NULL,
          duration_ms INTEGER,
          query       TEXT
        );
        PRAGMA user_version = 1;
        """
    )
    conn.execute(
        "INSERT INTO action_log (ts_utc, origin, tool, args, risk, verdict, outcome)"
        " VALUES ('2026-07-20T10:00:00+00:00', 'voix', 'webSearch', '{}',"
        " 'lecture', 'libre', 'ok')"
    )
    conn.commit()
    conn.close()
    return path


# ── The user keeps what they had ──────────────────────────────────────


def test_rows_written_by_the_previous_release_survive(previous_release):
    db = _open(previous_release)
    try:
        rows = db.recent_actions()
        assert len(rows) == 1
        assert rows[0]["tool"] == "webSearch"
        assert rows[0]["outcome"] == "ok"
    finally:
        db.close()


def test_an_old_row_reads_back_with_no_request_id(previous_release):
    """Absent, not invented. A row from before the feature was never part
    of a question."""
    db = _open(previous_release)
    try:
        assert db.recent_actions()[0]["request_id"] is None
    finally:
        db.close()


def test_the_new_column_is_writable_on_a_migrated_database(previous_release):
    db = _open(previous_release)
    try:
        db.record_action(
            tool="localFiles", args={"operation": "delete"}, risk="destructif",
            verdict="demande", outcome="demandé", duration_ms=None,
            origin="voix", query="supprime", request_id="cf_abc123",
        )
        assert db.recent_actions()[0]["request_id"] == "cf_abc123"
    finally:
        db.close()


def test_migrating_twice_is_harmless(previous_release):
    """Startup runs this every time, not only once."""
    for _ in range(3):
        db = _open(previous_release)
        db.close()

    db = _open(previous_release)
    try:
        assert len(db.recent_actions()) == 1
    finally:
        db.close()


# ── A fresh database arrives in the current shape ─────────────────────


def test_a_new_database_has_the_column_from_the_start(tmp_path):
    db = _open(tmp_path / "neuve.db")
    try:
        columns = {
            r[1] for r in db.conn.execute("PRAGMA table_info(action_log)").fetchall()
        }
        assert "request_id" in columns
    finally:
        db.close()


def test_the_version_stamp_moves_with_the_shape(tmp_path):
    from src.jarvis.memory.db import _SCHEMA_VERSION

    db = _open(tmp_path / "neuve.db")
    try:
        stamped = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert stamped == _SCHEMA_VERSION
        assert _SCHEMA_VERSION >= 2
    finally:
        db.close()


def test_a_question_and_its_answer_share_one_id(tmp_path):
    """The pair is the point: one row when she asks, one when it settles,
    and nothing but the id ties them together."""
    db = _open(tmp_path / "neuve.db")
    try:
        db.record_action(
            tool="localFiles", args={"operation": "delete"}, risk="destructif",
            verdict="demande", outcome="demandé", duration_ms=None,
            origin="chat", query="supprime", request_id="cf_paire",
        )
        db.record_action(
            tool="localFiles", args={"operation": "delete"}, risk="destructif",
            verdict="demande", outcome="décliné", duration_ms=None,
            origin="chat", query="supprime", request_id="cf_paire",
        )

        rows = [r for r in db.recent_actions() if r["request_id"] == "cf_paire"]
        assert {r["outcome"] for r in rows} == {"demandé", "décliné"}
    finally:
        db.close()
