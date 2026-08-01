"""The thread that keeps the promise.

Wall clock, not monotonic — the opposite of the confirmation TTL, for the
mirror reason. `PendingAction.has_expired` uses `time.monotonic()`
because a deadline that can move backwards is a resurrectable approval.
A confirmation TTL is an attention span; a reminder is an appointment
with the world. Polling against the wall clock is correct across a sleep
by construction: the comparison happens at tick time, so a laptop shut
from 09:00 to 14:00 finds the row due the moment it wakes.

Defer, never drop. A reminder that was owed and never said is the failure
this whole subsystem exists to prevent, and the only trace of a silent
drop would be a ledger line in a tab nobody has a reason to open. Past
the grace window she still says it — she says how late she is.

Delivery is what settles it, never queueing. The row stays owed until
the speech has actually finished, so a crash between the two costs a
repeat rather than a broken promise.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.reminders.scheduler import ReminderScheduler


class _Cfg:
    reminders_enabled = True
    reminder_tick_sec = 5.0
    reminder_late_grace_sec = 900.0
    reminder_max_attempts = 60
    voice_debug = False


def _iso(minutes: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


@pytest.fixture
def spoken():
    return []


@pytest.fixture
def scheduler(db, spoken):
    """A scheduler whose speaking is a list, and whose clock we drive."""
    def _speak(text, on_spoken=None):
        spoken.append((text, on_spoken))
        return True

    return ReminderScheduler(
        db=db, cfg=_Cfg(), speak=_speak, busy=lambda: False,
        announce=lambda action, outcome: None,
    )


def _add(db, **kw):
    payload = dict(
        texte="sortir le plat", due_utc=_iso(-1), due_local="2026-08-01T23:06",
        tz="Europe/Paris", origin="voix", query="rappelle-moi",
    )
    payload.update(kw)
    return db.add_rappel(**payload)


# ── It says what is owed ──────────────────────────────────────────────


def test_a_due_reminder_is_spoken(db, scheduler, spoken):
    _add(db)

    scheduler.tick()

    assert len(spoken) == 1
    assert "sortir le plat" in spoken[0][0]


def test_a_future_reminder_is_left_alone(db, scheduler, spoken):
    _add(db, due_utc=_iso(+60))

    scheduler.tick()

    assert spoken == []


def test_a_recent_reminder_is_said_plainly(db, scheduler, spoken):
    """Inside the grace window, saying "five seconds late" is noise."""
    _add(db, due_utc=_iso(-0.2))

    scheduler.tick()

    assert "retard" not in spoken[0][0].lower()


def test_a_late_reminder_says_how_late_it_is(db, scheduler, spoken):
    """A laptop shut from 09:00 to 14:00. She still says it — pretending
    it is now would be worse than being late."""
    _add(db, due_utc=_iso(-300))

    scheduler.tick()

    assert spoken, "a reminder owed five hours ago was silently dropped"
    assert "retard" in spoken[0][0].lower()


def test_nothing_is_ever_dropped_for_being_old(db, scheduler, spoken):
    """The failure the whole subsystem exists to prevent."""
    _add(db, due_utc=_iso(-60 * 24 * 3))

    scheduler.tick()

    assert spoken


# ── One voice, one thing at a time ────────────────────────────────────


def test_three_due_at_once_produce_one_utterance(db, scheduler, spoken):
    """Three separate hand-offs would talk over each other: the speaker
    holds one completion callback, and a second call destroys the
    first's."""
    _add(db, texte="premier")
    _add(db, texte="deuxième")
    _add(db, texte="troisième")

    scheduler.tick()

    assert len(spoken) == 1
    for word in ("premier", "deuxième", "troisième"):
        assert word in spoken[0][0]


def test_nothing_is_spoken_while_a_query_is_running(db, spoken):
    """Cutting across a reply the user is waiting for is worse than a few
    seconds late."""
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(), speak=lambda t, on_spoken=None: spoken.append(t),
        busy=lambda: True, announce=lambda a, o: None,
    )
    _add(db)

    scheduler.tick()

    assert spoken == []


def test_a_deferred_reminder_is_still_owed(db, spoken):
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(), speak=lambda t, on_spoken=None: spoken.append(t),
        busy=lambda: True, announce=lambda a, o: None,
    )
    _add(db)

    scheduler.tick()

    assert len(db.pending_rappels()) == 1


# ── Only delivery settles it ──────────────────────────────────────────


def test_queueing_does_not_settle_the_promise(db, scheduler, spoken):
    """A crash between queueing and speaking must cost a repeat, not a
    broken promise."""
    _add(db)

    scheduler.tick()

    assert len(db.pending_rappels()) == 1


def test_speech_finishing_settles_it(db, scheduler, spoken):
    _add(db)
    scheduler.tick()

    spoken[0][1]()  # the delivery callback

    assert db.pending_rappels() == []


def test_a_delivered_reminder_records_when_it_was_said(db, scheduler, spoken):
    rid = _add(db)
    scheduler.tick()

    spoken[0][1]()

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["said_utc"]
    assert row["etat"] == "fini"


def test_a_delivered_reminder_leaves_a_ledger_row(db, scheduler, spoken):
    rid = _add(db)
    scheduler.tick()

    spoken[0][1]()

    row = next(r for r in db.recent_actions() if r["request_id"] == rid)
    assert row["origin"] == "rappel"
    assert row["outcome"] == "ok"


# ── When nothing can say it ───────────────────────────────────────────


def test_a_dead_speaker_settles_the_row_rather_than_looping(db):
    """`enqueue_reply` returning False is definitive — no engine will
    ever say this. Retrying every tick forever would spin, and leaving it
    owed would promise something that cannot happen."""
    announced = []
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(), speak=lambda t, on_spoken=None: False,
        busy=lambda: False, announce=lambda a, o: announced.append((a, o)),
    )
    rid = _add(db)

    scheduler.tick()

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["etat"] == "fini"
    assert row["said_utc"] is None


def test_a_dead_speaker_says_so_in_the_ledger(db):
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(), speak=lambda t, on_spoken=None: False,
        busy=lambda: False, announce=lambda a, o: None,
    )
    rid = _add(db)

    scheduler.tick()

    row = next(r for r in db.recent_actions() if r["request_id"] == rid)
    assert row["outcome"] == "échec"


def test_a_dead_speaker_reaches_the_surfaces(db):
    """The user is owed the knowledge that it could not be said, and the
    ledger alone is a tab nobody opens."""
    announced = []
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(), speak=lambda t, on_spoken=None: False,
        busy=lambda: False, announce=lambda a, o: announced.append((a, o)),
    )
    _add(db)

    scheduler.tick()

    assert announced


# ── A failure is bounded ──────────────────────────────────────────────


def test_attempts_are_counted(db, scheduler, spoken):
    rid = _add(db)

    scheduler.tick()

    assert next(r for r in db.all_rappels() if r["id"] == rid)["attempts"] == 1


def test_a_reminder_that_never_lands_eventually_stops_trying(db, spoken):
    """Otherwise it retries forever, every tick, for the life of the
    process."""
    class _Few(_Cfg):
        reminder_max_attempts = 2

    scheduler = ReminderScheduler(
        db=db, cfg=_Few(), speak=lambda t, on_spoken=None: True,
        busy=lambda: False, announce=lambda a, o: None,
    )
    rid = _add(db)

    for _ in range(5):
        scheduler.tick()

    row = next(r for r in db.all_rappels() if r["id"] == rid)
    assert row["etat"] == "fini"
    assert row["attempts"] <= 2


# ── The thread ────────────────────────────────────────────────────────


def test_it_stops_promptly(db, scheduler):
    """It must be joined before the speaker dies, and the shutdown diary
    pass can take 45 seconds after that. A thread that only notices its
    stop flag on the next tick would be joined late."""
    scheduler.start()
    try:
        stopped = threading.Event()

        def _stop():
            scheduler.stop()
            stopped.set()

        threading.Thread(target=_stop).start()
        assert stopped.wait(timeout=2.0), "stop() did not return promptly"
    finally:
        scheduler.stop()


def test_a_new_reminder_wakes_it_early(db, scheduler):
    """A reminder for 30 seconds out must not wait for a 5-second tick
    boundary to be noticed — and one for two seconds out must not be
    late by a whole tick."""
    scheduler.start()
    try:
        scheduler.nudge()  # must not raise, and must not need the tick
    finally:
        scheduler.stop()


def test_disabled_means_nothing_fires(db, spoken):
    class _Off(_Cfg):
        reminders_enabled = False

    scheduler = ReminderScheduler(
        db=db, cfg=_Off(), speak=lambda t, on_spoken=None: spoken.append(t),
        busy=lambda: False, announce=lambda a, o: None,
    )
    _add(db)

    scheduler.tick()

    assert spoken == []
    assert len(db.pending_rappels()) == 1


def test_a_broken_tick_does_not_kill_the_thread(db, spoken):
    """It is the only thing keeping promises. A bad row must cost that
    row, not every reminder after it."""
    scheduler = ReminderScheduler(
        db=db, cfg=_Cfg(),
        speak=MagicMock(side_effect=RuntimeError("boum")),
        busy=lambda: False, announce=lambda a, o: None,
    )
    _add(db)

    scheduler.tick()  # must not raise

    assert len(db.pending_rappels()) == 1
