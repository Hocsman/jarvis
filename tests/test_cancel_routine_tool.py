"""Stopping one out loud, the way it was started.

The tab already has a button, and that is the honest surface: it shows
the name, the hour and the envelope, so the user stops the thing they can
see. This exists because a capability easier to grant than to revoke is
one people stop granting at all.

It stops the row and leaves the block. The block is the record of what
the routine was allowed to do — exactly what someone wants to read after
switching it off — and leaving it is what makes saying the sentence again
cheap, since `setRoutine` re-arms a block whose row has gone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.routines.recurrence import Regle
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.cancel_routine import CancelRoutineTool


class _Cfg:
    voice_debug = False

    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


@pytest.fixture
def cfg(tmp_path):
    return _Cfg(tmp_path)


def _add(db, *, nom="matin", texte="résumer mes mails"):
    return db.add_rappel(
        texte=texte, kind="routine", origin="voix",
        due_utc=(datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(),
        due_local="2026-08-03T07:00", tz="Europe/Paris", query=texte,
        payload={"nom": nom,
                 "regle": Regle(kind="daily", hour=7, minute=0,
                                weekday=None).to_json()},
    )


def _run(db, cfg, args):
    return CancelRoutineTool().run(args, ToolContext(
        db=db, cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1, user_print=lambda m: None,
        origin="voix",
    ))


def _live(db):
    return list(db.pending_rappels(kind="routine"))


# ── Stopping ──────────────────────────────────────────────────────────


def test_a_named_routine_stops(db, cfg):
    _add(db)

    result = _run(db, cfg, {"nom": "matin"})

    assert result.success is True
    assert _live(db) == []


def test_stopping_leaves_the_others_alone(db, cfg):
    _add(db, nom="matin")
    _add(db, nom="bilan")

    _run(db, cfg, {"nom": "matin"})

    assert [r["texte"] for r in _live(db)] == ["résumer mes mails"]


def test_a_reminder_is_not_a_routine(db, cfg):
    """They share a table and nothing else. Cancelling the user's dinner
    reminder because they asked to stop a morning digest would be the
    worst kind of near-miss."""
    _add(db, nom="matin")
    db.add_rappel(texte="sortir le plat", kind="rappel",
                  due_utc=(datetime.now(timezone.utc)
                           + timedelta(hours=1)).isoformat(),
                  due_local="x", tz="Europe/Paris")

    result = _run(db, cfg, {"nom": "sortir le plat"})

    assert result.success is False
    assert len(db.pending_rappels(kind="rappel")) == 1
    assert len(_live(db)) == 1


# ── A name that is not there ──────────────────────────────────────────


def test_an_unknown_name_stops_nothing_and_says_what_exists(db, cfg):
    """A near-match silently stopped is the wrong routine silently
    stopped, and the user finds out a morning later at best."""
    _add(db, nom="matin")

    result = _run(db, cfg, {"nom": "matinale"})

    assert result.success is False
    assert len(_live(db)) == 1
    assert "matin" in result.reply_text


def test_a_cancel_that_did_not_take_is_reported_as_a_failure(db, cfg):
    """Saying it is stopped while it still fires is the worst answer
    available here, so the tool reports what the write returned rather
    than what it attempted."""
    _add(db, nom="matin")

    class _Stubborn:
        def pending_rappels(self, kind=None):
            return db.pending_rappels(kind=kind)

        def cancel_rappel(self, rappel_id):
            return False

    result = CancelRoutineTool().run({"nom": "matin"}, ToolContext(
        db=_Stubborn(), cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1, user_print=lambda m: None,
        origin="voix",
    ))

    assert result.success is False
    assert len(_live(db)) == 1


def test_a_database_that_raises_is_not_reported_as_a_stop(db, cfg):
    _add(db, nom="matin")

    class _Broken:
        def pending_rappels(self, kind=None):
            return db.pending_rappels(kind=kind)

        def cancel_rappel(self, rappel_id):
            raise RuntimeError("base verrouillée")

    result = CancelRoutineTool().run({"nom": "matin"}, ToolContext(
        db=_Broken(), cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1, user_print=lambda m: None,
        origin="voix",
    ))

    assert result.success is False
    assert len(_live(db)) == 1


# ── Listing ───────────────────────────────────────────────────────────


def test_no_name_lists_what_is_running(db, cfg):
    _add(db, nom="matin")

    result = _run(db, cfg, {})

    assert result.success is True
    assert "matin" in result.reply_text
    assert len(_live(db)) == 1


def test_listing_stops_nothing(db, cfg):
    _add(db, nom="matin")

    _run(db, cfg, {})

    assert len(_live(db)) == 1


def test_nothing_running_is_said_rather_than_treated_as_an_error(db, cfg):
    result = _run(db, cfg, {"nom": "matin"})

    assert result.success is True


# ── It cannot be turned on itself ─────────────────────────────────────


def test_a_routine_cannot_silence_another_routine(db, cfg):
    """It writes no row of its own, but it reaches into Yuba's own
    bookkeeping, and nothing running unattended should be able to switch
    off what the user set up."""
    assert CancelRoutineTool().writes_own_state is True
