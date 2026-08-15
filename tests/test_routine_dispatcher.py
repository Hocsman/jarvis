"""Who decides that a morning has arrived, and who decides it has passed.

The dispatcher is one indexed SELECT and a hand-off, on its own thread.
Its own, deliberately: the reminder scheduler is the only thing keeping
spoken promises, and it must not share a failure surface with a newer and
more complicated feature.

Three decisions live here, and each one leans the same way.

**Late is not the same as missed.** A digest two hours late is still the
thing that was asked for; the same digest at 18:00 is not. Past the
window the occurrence is skipped and the row moves on, because a routine
that fires eleven hours late teaches the user to ignore it.

**A run it cannot start is not a run it drops.** Busy, or a run already
in flight, leaves the row exactly as it was, so the next tick finds it
again. The staleness window is what eventually ends that, rather than a
counter.

**A routine that has produced nothing for days is broken, and stops.**
Loudly: a page in the journal saying so, because the failure this guards
against is a morning digest that silently stopped arriving in October
and was noticed in December.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.routines.dispatcher import RoutineDispatcher
from src.jarvis.routines.journal import read_day
from src.jarvis.routines.recurrence import Regle


ROUTINES_FILE = """# Routines

## matin
phrase: résume-moi mes mails
quand: tous les jours à 07:00
outils:
- webSearch
"""


class _Cfg:
    routines_enabled = True
    routine_tick_sec = 30.0
    routine_late_grace_sec = 14400.0
    routine_max_steriles = 5
    voice_debug = False

    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


class _Runner:
    """A runner whose slot we drive by hand."""

    def __init__(self, *, accepts=True):
        self.submitted = []
        self._accepts = accepts

    def busy(self):
        return not self._accepts

    def submit(self, row):
        if not self._accepts:
            return False
        self.submitted.append(dict(row))
        return True


@pytest.fixture
def cfg(tmp_path):
    config = _Cfg(tmp_path)
    core = tmp_path / "yuba"
    core.mkdir(parents=True, exist_ok=True)
    (core / "routines.md").write_text(ROUTINES_FILE, encoding="utf-8")
    return config


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "jarvis.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _add(db, *, nom="matin", minutes_late=1.0, regle=None, steriles=0,
         texte="résume-moi mes mails"):
    due = datetime.now(timezone.utc) - timedelta(minutes=minutes_late)
    payload = {
        "nom": nom,
        "regle": (regle or Regle(kind="daily", hour=7, minute=0,
                                 weekday=None)).to_json(),
    }
    if steriles:
        payload["steriles"] = steriles
    return db.add_rappel(
        texte=texte, kind="routine", origin="voix",
        due_utc=due.isoformat(), due_local=due.isoformat(), tz="Europe/Paris",
        query=texte, payload=payload,
    )


def _row(db, rid):
    return next(r for r in db.all_rappels() if r["id"] == rid)


def _dispatcher(db, cfg, runner, *, busy=False):
    return RoutineDispatcher(db=db, cfg=cfg, runner=runner, busy=lambda: busy)


# ── A morning that arrived ────────────────────────────────────────────


def test_a_due_routine_is_handed_to_the_runner(db, cfg):
    runner = _Runner()
    rid = _add(db)

    _dispatcher(db, cfg, runner).tick()

    assert [r["id"] for r in runner.submitted] == [rid]


def test_a_routine_not_yet_due_is_left_alone(db, cfg):
    runner = _Runner()
    _add(db, minutes_late=-60)

    _dispatcher(db, cfg, runner).tick()

    assert runner.submitted == []


def test_a_reminder_is_not_a_routine(db, cfg):
    """They share a table and nothing else. A reminder handed to the
    runner would be answered by the reply engine instead of spoken."""
    runner = _Runner()
    db.add_rappel(
        texte="sortir le plat", kind="rappel",
        due_utc=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        due_local="x", tz="Europe/Paris",
    )

    _dispatcher(db, cfg, runner).tick()

    assert runner.submitted == []


def test_nothing_fires_when_routines_are_switched_off(db, cfg):
    cfg.routines_enabled = False
    runner = _Runner()
    _add(db)

    _dispatcher(db, cfg, runner).tick()

    assert runner.submitted == []


# ── A morning that has passed ─────────────────────────────────────────


def test_a_routine_too_late_to_be_useful_is_skipped(db, cfg):
    """A digest eleven hours late is not the thing that was asked for,
    and a routine that keeps arriving at the wrong time is a routine the
    user learns to ignore."""
    runner = _Runner()
    _add(db, minutes_late=60 * 11)

    _dispatcher(db, cfg, runner).tick()

    assert runner.submitted == []


def test_a_skipped_morning_still_moves_the_row(db, cfg):
    """Skipping is not dropping. Left where it was, it would be found
    late again on every tick for the rest of the day."""
    runner = _Runner()
    rid = _add(db, minutes_late=60 * 11)
    was = _row(db, rid)["due_utc"]

    _dispatcher(db, cfg, runner).tick()

    assert _row(db, rid)["due_utc"] != was


def test_a_skipped_morning_is_written_down(db, cfg):
    """Otherwise the page for that day is simply missing, which reads as
    "it never fired" and sends the user to look at the schedule."""
    runner = _Runner()
    _add(db, minutes_late=60 * 11)

    _dispatcher(db, cfg, runner).tick()

    assert "matin" in read_day(cfg, datetime.now())


def test_a_late_but_still_useful_morning_runs(db, cfg):
    """The laptop was shut at 07:00 and opened at 09:00. That digest is
    still wanted."""
    runner = _Runner()
    _add(db, minutes_late=120)

    _dispatcher(db, cfg, runner).tick()

    assert len(runner.submitted) == 1


# ── A run it cannot start now ─────────────────────────────────────────


def test_nothing_starts_while_the_user_is_mid_query(db, cfg):
    """The runner never holds the query lock, so this is not about
    deadlock. It is about the user's own reply getting slower because
    something they did not ask for is competing for the same model."""
    runner = _Runner()
    _add(db)

    _dispatcher(db, cfg, runner, busy=True).tick()

    assert runner.submitted == []


def test_a_routine_deferred_because_the_user_was_busy_is_still_owed(db, cfg):
    runner = _Runner()
    rid = _add(db)
    was = _row(db, rid)["due_utc"]

    _dispatcher(db, cfg, runner, busy=True).tick()

    assert _row(db, rid)["due_utc"] == was
    assert _row(db, rid)["etat"] == db.ETAT_PENDING


def test_a_routine_the_runner_refused_is_still_owed(db, cfg):
    """One at a time. The second one waits for the next tick rather than
    being lost."""
    runner = _Runner(accepts=False)
    rid = _add(db)
    was = _row(db, rid)["due_utc"]

    _dispatcher(db, cfg, runner).tick()

    assert _row(db, rid)["due_utc"] == was


# ── A routine that has stopped working ────────────────────────────────


def test_a_routine_that_has_produced_nothing_for_days_is_stopped(db, cfg):
    """Failing every morning forever costs a rate limit and a wallet, and
    the write-up it produces is not one anybody reads."""
    runner = _Runner()
    rid = _add(db, steriles=5)

    _dispatcher(db, cfg, runner).tick()

    assert runner.submitted == []
    assert _row(db, rid)["etat"] == db.ETAT_CANCELLED


def test_stopping_a_routine_is_said_out_loud_in_the_journal(db, cfg):
    """The failure this guards against is a morning digest that silently
    stopped arriving in October and was noticed in December."""
    runner = _Runner()
    _add(db, steriles=5)

    _dispatcher(db, cfg, runner).tick()

    day = read_day(cfg, datetime.now())
    assert "matin" in day
    assert "5" in day


def test_one_short_of_the_limit_still_runs(db, cfg):
    runner = _Runner()
    _add(db, steriles=4)

    _dispatcher(db, cfg, runner).tick()

    assert len(runner.submitted) == 1


# ── A row that can never work ─────────────────────────────────────────


def test_a_routine_whose_rule_cannot_be_read_is_stopped(db, cfg):
    """It can never be advanced, so it is due forever: every tick, for
    the life of the process. A routine that cannot say when it runs must
    not exist."""
    runner = _Runner()
    rid = db.add_rappel(
        texte="x", kind="routine",
        due_utc=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        due_local="x", tz="Europe/Paris",
        payload={"nom": "matin", "regle": {"kind": "toutes les minutes"}},
    )

    _dispatcher(db, cfg, runner).tick()

    assert runner.submitted == []
    assert _row(db, rid)["etat"] == db.ETAT_CANCELLED


# ── The thread ────────────────────────────────────────────────────────


def test_a_bad_row_does_not_stop_the_ones_after_it(db, cfg):
    runner = _Runner()
    db.add_rappel(
        texte="x", kind="routine",
        due_utc=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        due_local="x", tz="Europe/Paris", payload={"nom": "cassé"},
    )
    good = _add(db)

    _dispatcher(db, cfg, runner).tick()

    assert [r["id"] for r in runner.submitted] == [good]


def test_a_broken_tick_does_not_kill_the_thread(db, cfg):
    class _Explosive(_Runner):
        def submit(self, row):
            raise RuntimeError("boum")

    _add(db)
    dispatcher = _dispatcher(db, cfg, _Explosive())

    dispatcher.tick()  # must not raise


def test_it_stops_promptly(db, cfg):
    dispatcher = _dispatcher(db, cfg, _Runner())
    dispatcher.start()

    began = time.monotonic()
    dispatcher.stop()

    assert time.monotonic() - began < 2.0


def test_a_new_routine_wakes_it_early(db, cfg):
    """A routine created for two minutes' time must not wait out a tick
    that was already half spent."""
    runner = _Runner()
    dispatcher = _dispatcher(db, cfg, runner)
    dispatcher.start()
    try:
        _add(db)
        dispatcher.nudge()
        deadline = time.monotonic() + 3.0
        while not runner.submitted and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        dispatcher.stop()

    assert len(runner.submitted) == 1


# ── What reaches the tray ─────────────────────────────────────────────


def test_stopping_a_routine_reaches_the_user_not_only_the_journal(db, cfg):
    """A journal page alone would be found weeks later, and by then the
    digest has been missing the whole time."""
    told = []
    runner = _Runner()
    _add(db, steriles=5)

    RoutineDispatcher(
        db=db, cfg=cfg, runner=runner, busy=lambda: False,
        announce=lambda nom, pourquoi, stopped: told.append((nom, stopped)),
    ).tick()

    assert told == [("matin", True)]


def test_a_skipped_morning_does_not_reach_the_tray(db, cfg):
    """It is one occurrence, it is written down, and the routine is
    fine. A balloon for it is a balloon people learn to dismiss."""
    told = []
    runner = _Runner()
    _add(db, minutes_late=60 * 11)

    RoutineDispatcher(
        db=db, cfg=cfg, runner=runner, busy=lambda: False,
        announce=lambda nom, pourquoi, stopped: told.append(nom),
    ).tick()

    assert told == []


def test_an_announcer_that_raises_does_not_take_the_tick_down(db, cfg):
    runner = _Runner()
    _add(db, steriles=5)

    def _boom(nom, pourquoi, stopped):
        raise RuntimeError("pas de zone de notification")

    RoutineDispatcher(
        db=db, cfg=cfg, runner=runner, busy=lambda: False, announce=_boom,
    ).tick()  # must not raise
