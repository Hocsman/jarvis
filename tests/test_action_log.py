"""The record of what the assistant actually did.

One line per tool call, written from the gate so it covers every route
into execution. The invariant that keeps this small and safe to read:
**it records what was done, never what was seen.** No tool output, ever.
A ledger that captured results would accumulate the contents of every
page fetched and every file read, which is a different and much larger
thing than a list of actions.

Arguments are stored redacted, because a tool call carries whatever the
user just said.
"""

from __future__ import annotations

import pytest

from src.jarvis.memory.db import Database


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "test.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _record(db, **kw):
    payload = dict(
        tool="webSearch",
        args={"query": "météo"},
        risk="lecture",
        verdict="libre",
        outcome="ok",
        duration_ms=12,
        origin="chat",
        query="quel temps fait-il",
    )
    payload.update(kw)
    return db.record_action(**payload)


# ── What it keeps ─────────────────────────────────────────────────────


def test_a_call_is_recorded(db):
    _record(db)

    rows = db.recent_actions()
    assert len(rows) == 1
    assert rows[0]["tool"] == "webSearch"


def test_the_record_carries_what_the_user_needs_to_judge_it(db):
    _record(db, tool="localFiles", risk="destructif", verdict="demande", outcome="refusé")

    row = db.recent_actions()[0]
    assert row["risk"] == "destructif"
    assert row["verdict"] == "demande"
    assert row["outcome"] == "refusé"
    assert row["ts_utc"]


def test_a_refusal_is_recorded_as_readily_as_a_success(db):
    """Refusals are the interesting half: they are how the user notices
    the policy is too tight, or that something tried what it should not."""
    _record(db, outcome="refusé")

    assert db.recent_actions()[0]["outcome"] == "refusé"


def test_the_newest_call_comes_first(db):
    _record(db, tool="webSearch")
    _record(db, tool="getWeather")

    assert [r["tool"] for r in db.recent_actions()] == ["getWeather", "webSearch"]


# ── What it must never keep ───────────────────────────────────────────


def test_no_column_exists_for_tool_output(db):
    """Structural, not a promise. A ledger with nowhere to put results
    cannot accumulate the contents of every page fetched and file read."""
    cur = db.conn.execute("PRAGMA table_info(action_log)")
    columns = {row[1] for row in cur.fetchall()}

    for forbidden in ("output", "result", "reply", "reply_text", "content"):
        assert forbidden not in columns


def test_arguments_are_stored_redacted(db):
    _record(db, args={"to": "hocsman92@gmail.com"})

    stored = db.recent_actions()[0]["args"]
    assert "hocsman92@gmail.com" not in stored
    assert "REDACTED" in stored


def test_the_query_is_stored_redacted(db):
    _record(db, query="envoie ça à hocsman92@gmail.com")

    stored = db.recent_actions()[0]["query"]
    assert "hocsman92@gmail.com" not in stored


def test_unserialisable_arguments_do_not_break_the_call(db):
    """The ledger is bookkeeping. It must never be the reason a tool call
    fails."""
    _record(db, args={"handle": object()})

    assert len(db.recent_actions()) == 1


# ── Keeping it bounded ────────────────────────────────────────────────


def test_old_entries_are_dropped(db):
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    _record(db, tool="ancien")
    db.conn.execute("UPDATE action_log SET ts_utc = ?", (old,))
    db.conn.commit()
    _record(db, tool="récent")

    db.prune_actions(max_age_days=90)

    assert [r["tool"] for r in db.recent_actions()] == ["récent"]


def test_the_user_can_clear_the_whole_ledger(db):
    _record(db)
    _record(db)

    db.clear_actions()

    assert db.recent_actions() == []


def test_recent_actions_is_bounded(db):
    for i in range(10):
        _record(db, tool=f"outil{i}")

    assert len(db.recent_actions(limit=3)) == 3


# ── The schema can be migrated later ──────────────────────────────────


def test_the_schema_carries_a_version(db):
    """Three tables are coming, one of which will hold the user's
    reminders. The only migration precedent in this project wipes its
    table when the shape surprises it; that is not an acceptable
    inheritance for data the user asked to keep."""
    version = db.conn.execute("PRAGMA user_version").fetchone()[0]

    assert version >= 1
