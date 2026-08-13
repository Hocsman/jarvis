"""The next occurrence has to land after the run that produced it.

A routine row carries its own zone, frozen when the user set it. The
advance computed the next occurrence from the machine's clock and then
stored it under the row's zone, so a row set in Paris and a daemon
running in New York produced an occurrence belonging to neither.

Six hours west, that occurrence lands in the past. A row due in the past
is due again on the very next tick, and on every tick after it: one
morning becomes hundreds of engine turns, hundreds of model calls, and
hundreds of ledger rows, with nothing announced. From the outside it
looks like a morning that worked.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest


PARIS = "Europe/Paris"


def _iso(delta_sec: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_sec)).isoformat()


def _ligne(db, rid: str) -> dict:
    for r in db.all_rappels():
        if r.get("id") == rid:
            return r
    raise AssertionError(f"ligne {rid} introuvable")


# ── The clock a row's zone reads ───────────────────────────────────────


def test_the_wall_clock_is_read_off_the_row_zone_not_the_machine(monkeypatch):
    """`now_in` is the exact counterpart of `to_utc_iso`: that one takes a
    wall-clock reading to an instant, this one brings the instant back to
    a reading."""
    from src.jarvis.utils.time_context import now_in

    monkeypatch.setenv("TZ", "America/New_York")

    a_paris = now_in(PARIS)
    reference = datetime.now(timezone.utc).astimezone(
        __import__("zoneinfo").ZoneInfo(PARIS)
    ).replace(tzinfo=None)

    assert abs((a_paris - reference).total_seconds()) < 5


def test_with_no_zone_it_reads_the_machine():
    """Symmetrical with `to_utc_iso`, which falls back the same way."""
    from src.jarvis.utils.time_context import now_in

    assert abs((now_in("") - datetime.now()).total_seconds()) < 5


# ── The row must move forward ──────────────────────────────────────────


def test_a_row_cannot_be_advanced_onto_a_moment_already_past(tmp_path):
    """Backwards is not a smaller step forward. A row moved onto an
    instant already gone is owed again immediately, so it is no move at
    all wearing the shape of one."""
    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"))
    rid = db.add_rappel(
        texte="le point du matin", due_utc=_iso(60),
        due_local="2026-08-13T07:00:00", tz=PARIS,
    )
    avant = _ligne(db, rid)["due_utc"]

    with pytest.raises(Exception):
        db.advance_rappel(rid, due_utc=_iso(-300),
                          due_local="2026-08-13T07:00:00", tz=PARIS)

    assert _ligne(db, rid)["due_utc"] == avant


def test_a_row_advanced_forward_still_moves(tmp_path):
    """The guard refuses a step backwards, not the ordinary step."""
    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"))
    rid = db.add_rappel(
        texte="le point du matin", due_utc=_iso(60),
        due_local="2026-08-13T07:00:00", tz=PARIS,
    )
    avant = _ligne(db, rid)["due_utc"]

    db.advance_rappel(rid, due_utc=_iso(86400),
                      due_local="2026-08-14T07:00:00", tz=PARIS)

    assert _ligne(db, rid)["due_utc"] != avant


# ── End to end: one occurrence is one run ──────────────────────────────


def _regle_quotidienne(heure: int = 7):
    from src.jarvis.routines.recurrence import Regle

    return Regle(kind="daily", hour=heure, minute=0, weekday=None)


def _ligne_due(db, tz: str, heure: int = 7):
    """A routine row owed right now, carrying `tz` as its own zone."""
    regle = _regle_quotidienne(heure)
    return db.add_rappel(
        texte="fais le point du matin",
        due_utc=_iso(-1),
        due_local=f"2026-08-13T{heure:02d}:00:00",
        tz=tz,
        kind="routine",
        payload={"nom": "matin", "regle": regle.to_json()},
    )


@pytest.fixture
def horloge_new_york(monkeypatch):
    """The daemon's own clock six hours behind the row's zone."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_the_next_occurrence_lands_ahead_of_the_run_it_follows(tmp_path, horloge_new_york):
    """The measured case: a row whose zone sits east of the machine came
    back with a due time *before* the tick that fired it.

    The hour is read off the machine clock so the window the defect lives
    in is entered on purpose rather than by luck: an occurrence an hour
    ahead of the machine's own reading is six hours behind the row's.
    """
    from src.jarvis.memory.db import Database
    from src.jarvis.routines.runner import RoutineRunner

    heure = (datetime.now() + timedelta(hours=1)).hour

    db = Database(str(tmp_path / "t.db"))
    rid = _ligne_due(db, PARIS, heure)
    row = _ligne(db, rid)

    runner = RoutineRunner(db, _cfg_muet(tmp_path))
    runner._advance(dict(row), json.loads(row["payload"]))

    apres = datetime.fromisoformat(_ligne(db, rid)["due_utc"])
    assert apres > datetime.now(timezone.utc)


def _cfg_muet(tmp_path):
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.routine_max_steriles = 5
    return cfg


def test_a_morning_that_cannot_be_rescheduled_is_skipped_out_loud(tmp_path):
    """A row that did not move is owed again on the very next tick, and on
    every tick after it. Running it now buys one morning and a loop; one
    morning missed, said out loud, is the cheaper of the two."""
    from src.jarvis.memory.db import Database
    from src.jarvis.routines.runner import RoutineRunner

    db = Database(str(tmp_path / "t.db"))
    rid = db.add_rappel(
        texte="fais le point du matin", due_utc=_iso(-1),
        due_local="2026-08-13T07:00:00", tz=PARIS, kind="routine",
        payload={"nom": "matin", "regle": {"kind": "jamais"}},
    )

    dits = []
    tours = []
    runner = RoutineRunner(
        db, _cfg_muet(tmp_path),
        engine=lambda *a, **k: tours.append(1) or "ok",
        announce=lambda nom, pourquoi, **kw: dits.append((nom, pourquoi)),
    )
    runner._run(dict(_ligne(db, rid)))

    assert dits, "une matinée sautée sans un mot est une matinée qui a l'air d'avoir marché"
    assert not tours
