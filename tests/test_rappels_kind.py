"""One table, two kinds, and no crossing between them.

`rappels` was built with a `kind` column and a default so that routines
would need no migration. Now that routines exist, the column has to
actually separate them: a routine due at 07:00 must never be spoken aloud
by the reminder scheduler, and must never appear in the Rappels tab —
where the user would try to cancel it and find it back the next morning.

`advance_rappel` is the other half. A reminder settles once; a routine
moves to its next occurrence and keeps going, which means resetting the
attempt count. A routine that failed twice in March must not arrive in
December one failure from being switched off.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.memory.db import Database


def _iso(minutes: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _add(db, kind="rappel", **kw):
    payload = dict(
        texte="quelque chose", due_utc=_iso(-1), due_local="2026-08-01T07:00",
        tz="Europe/Paris", origin="voix", query=None, kind=kind,
    )
    payload.update(kw)
    return db.add_rappel(**payload)


# ── The two kinds do not mix ──────────────────────────────────────────


def test_due_rows_can_be_asked_for_by_kind(db):
    _add(db, kind="rappel", texte="un rappel")
    _add(db, kind="routine", texte="une routine")

    assert [r["texte"] for r in db.due_rappels(_iso(), kind="rappel")] == ["un rappel"]
    assert [r["texte"] for r in db.due_rappels(_iso(), kind="routine")] == ["une routine"]


def test_asking_for_neither_returns_both(db):
    """The scheduler asks by kind; a maintenance sweep should not have
    to."""
    _add(db, kind="rappel")
    _add(db, kind="routine")

    assert len(db.due_rappels(_iso())) == 2


def test_the_pending_list_can_be_asked_for_by_kind(db):
    _add(db, kind="rappel", texte="un rappel")
    _add(db, kind="routine", texte="une routine")

    assert [r["texte"] for r in db.pending_rappels(kind="rappel")] == ["un rappel"]


def test_a_routine_is_never_spoken_by_the_reminder_scheduler(db):
    """It would be read out as a sentence — which is not what a routine
    is — and settled, so it would never run again."""
    from src.jarvis.reminders.scheduler import ReminderScheduler

    class _Cfg:
        reminders_enabled = True
        reminder_tick_sec = 5.0
        reminder_late_grace_sec = 900.0
        reminder_max_attempts = 60
        voice_debug = False

    spoken = []
    _add(db, kind="routine", texte="résumer les mails")
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(), speak=lambda t, on_spoken=None: spoken.append(t),
        busy=lambda: False, announce=lambda a, o: None,
    )

    scheduler.tick()

    assert spoken == []
    assert len(db.pending_rappels(kind="routine")) == 1


def test_a_routine_does_not_show_up_in_the_reminders_tab(db, tmp_path):
    """The user would try to cancel it there and find it back the next
    morning."""
    import json

    from src.desktop_app import memory_viewer

    _add(db, kind="routine", texte="résumer les mails")
    _add(db, kind="rappel", texte="sortir le plat")
    memory_viewer._activity_db = db
    try:
        listed = json.loads(memory_viewer.app.test_client().get("/api/rappels").data)
    finally:
        memory_viewer._activity_db = None

    assert [r["texte"] for r in listed["rappels"]] == ["sortir le plat"]


# ── Advancing, rather than settling ───────────────────────────────────


def test_advancing_moves_the_row_to_its_next_occurrence(db):
    rid = _add(db, kind="routine")
    later = _iso(+60)

    db.advance_rappel(rid, due_utc=later, due_local="2026-08-02T07:00",
                      tz="Europe/Paris")

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["due_utc"] == later
    assert row["due_local"] == "2026-08-02T07:00"


def test_an_advanced_row_is_still_owed(db):
    rid = _add(db, kind="routine")

    db.advance_rappel(rid, due_utc=_iso(+60), due_local="x", tz="Europe/Paris")

    assert len(db.pending_rappels(kind="routine")) == 1


def test_advancing_forgets_old_failures(db):
    """A routine that failed twice in March must not arrive in December
    one failure from being switched off."""
    rid = _add(db, kind="routine")
    for _ in range(59):
        db.mark_rappel_tried(rid, _iso())

    db.advance_rappel(rid, due_utc=_iso(+60), due_local="x", tz="Europe/Paris")

    assert next(r for r in db.all_rappels() if r["id"] == rid)["attempts"] == 0


def test_advancing_something_cancelled_does_not_revive_it(db):
    """Cancelling has to be final, or the user cannot stop a routine."""
    rid = _add(db, kind="routine")
    db.cancel_rappel(rid)

    db.advance_rappel(rid, due_utc=_iso(+60), due_local="x", tz="Europe/Paris")

    assert db.pending_rappels(kind="routine") == []


def test_a_reminder_still_defaults_to_its_own_kind(db):
    """Existing callers pass no kind and must keep working."""
    rid = db.add_rappel(
        texte="x", due_utc=_iso(-1), due_local="x", tz="Europe/Paris",
    )

    assert next(r for r in db.all_rappels() if r["id"] == rid)["kind"] == "rappel"
