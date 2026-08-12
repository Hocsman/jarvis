"""The morning write-up says what she reached, and what stopped her.

The engine hands back text and nothing else, so the write-up learns what
a run used by reading the ledger back. That read-back was written against
dicts and the ledger returns `sqlite3.Row`, which has no `.get` — so the
first line of the loop raised `AttributeError`, a bare `except` swallowed
it, and every write-up has claimed no tools were used and nothing was
turned away.

Confirmed against the real database rather than inferred: the row type is
`sqlite3.Row` and `.get` raises on it.

Nobody would have noticed. An empty list is exactly what a quiet morning
looks like, and the only trace was one debug line in a channel nobody
reads. It is the same shape as every other defect this project has found
in a week: it fails in the direction that looks like success.

What the write-up is *for* is the half nobody sees — the tools the gate
turned away. A routine that silently stopped reaching half its envelope
reads, in the journal, exactly like one that had a quiet morning.
"""

from __future__ import annotations

import sqlite3

import pytest


def _db(tmp_path, lignes):
    """A real ledger, with real rows, because the row type is the defect."""
    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    for tool, outcome, ts in lignes:
        db.record_action(tool=tool, args={}, risk="lecture", verdict="libre",
                         outcome=outcome, origin="routine", query="x")
        with db._lock:
            db.conn.execute(
                "UPDATE action_log SET ts_utc = ? WHERE id = "
                "(SELECT MAX(id) FROM action_log)", (ts,))
            db.conn.commit()
    return db


def _runner(db):
    from src.jarvis.routines.runner import RoutineRunner

    r = RoutineRunner.__new__(RoutineRunner)
    r._db = db
    return r


def test_every_ledger_reader_hands_back_the_same_shape(tmp_path):
    """The root cause, pinned. `recent_actions` alone returned raw
    `sqlite3.Row` while `pending_rappels` beside it returned dicts, and a
    reader written against one shape raised on the other. One shape, and
    the trap cannot be laid again."""
    db = _db(tmp_path, [("getWeather", "ok", "2026-08-12T09:00:00")])

    ligne = db.recent_actions(1)[0]

    assert isinstance(ligne, dict)
    assert ligne.get("tool") == "getWeather"
    assert ligne.get("colonne_absente") is None


def test_what_the_run_used_comes_back(tmp_path):
    db = _db(tmp_path, [("getWeather", "ok", "2026-08-12T09:00:00"),
                        ("webSearch", "ok", "2026-08-12T09:00:05")])

    outils, ecartes = _runner(db)._tool_calls_since("2026-08-12T08:00:00")

    assert set(outils) == {"getWeather", "webSearch"}
    assert ecartes == []


def test_what_the_gate_turned_away_comes_back_with_its_reason(tmp_path):
    """This is the half the write-up exists for. Without it, a routine
    silently blocked from half its envelope reads like a quiet morning."""
    db = _db(tmp_path, [("localFiles", "refusé", "2026-08-12T09:00:00")])

    outils, ecartes = _runner(db)._tool_calls_since("2026-08-12T08:00:00")

    assert outils == []
    assert len(ecartes) == 1 and ecartes[0][0] == "localFiles"
    assert ecartes[0][1]


def test_a_tool_used_twice_is_named_once(tmp_path):
    db = _db(tmp_path, [("webSearch", "ok", "2026-08-12T09:00:00"),
                        ("webSearch", "ok", "2026-08-12T09:00:09")])

    outils, _ = _runner(db)._tool_calls_since("2026-08-12T08:00:00")

    assert outils == ["webSearch"]


def test_calls_from_before_the_run_are_not_claimed(tmp_path):
    """The window is what stops one morning taking credit for another."""
    db = _db(tmp_path, [("getWeather", "ok", "2026-08-11T09:00:00"),
                        ("webSearch", "ok", "2026-08-12T09:00:00")])

    outils, _ = _runner(db)._tool_calls_since("2026-08-12T08:00:00")

    assert outils == ["webSearch"]


def test_the_routines_own_bookkeeping_row_is_not_a_tool(tmp_path):
    from src.jarvis.routines.runner import LEDGER_PREFIX

    db = _db(tmp_path, [(f"{LEDGER_PREFIX}actusWebMatin", "ok", "2026-08-12T09:00:00"),
                        ("webSearch", "ok", "2026-08-12T09:00:02")])

    outils, _ = _runner(db)._tool_calls_since("2026-08-12T08:00:00")

    assert outils == ["webSearch"]


def test_an_attended_call_is_not_claimed_by_a_routine(tmp_path):
    """He may be talking to her while a routine runs."""
    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    db.record_action(tool="webSearch", args={}, risk="lecture", verdict="libre",
                     outcome="ok", origin="voix", query="x")
    with db._lock:
        db.conn.execute("UPDATE action_log SET ts_utc = '2026-08-12T09:00:00'")
        db.conn.commit()

    outils, _ = _runner(db)._tool_calls_since("2026-08-12T08:00:00")

    assert outils == []


def test_no_window_means_no_claim(tmp_path):
    db = _db(tmp_path, [("webSearch", "ok", "2026-08-12T09:00:00")])

    assert _runner(db)._tool_calls_since(None) == ([], [])


def test_an_unreadable_ledger_costs_the_detail_and_not_the_write_up(tmp_path):
    """Bookkeeping never breaks the morning. But it must fail for a real
    reason, not because the row type surprised it."""
    from unittest.mock import MagicMock

    db = MagicMock()
    db.recent_actions.side_effect = RuntimeError("disque")

    assert _runner(db)._tool_calls_since("2026-08-12T08:00:00") == ([], [])
