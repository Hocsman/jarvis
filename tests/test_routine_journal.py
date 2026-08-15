"""The morning after.

A routine runs while the user is asleep. There is no TTS, no chat
bubble, nobody to say "hm?" to. The journal is the whole delivery: if it
is not written there, the routine happened to nobody.

Which means it carries the write-up itself, and not merely a note that
something ran. That is the one place this diverges from the action
ledger, which deliberately records what was *done* and never what was
*seen*. The ledger is an audit trail; this is the letter waiting on the
kitchen table. It never leaves the machine — same directory as the
user's own profile — so there is nothing here to redact.

Everything else is the usual discipline: one file per day so a crash
costs one morning, append rather than rewrite, and never raise. A
routine that finished its work and then died writing the diary would be
the worst possible failure, because the work is gone and there is no
record it was ever done.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.jarvis.routines.journal import (
    Entree,
    append_run,
    journal_path,
    prune_journal,
    read_day,
)


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


def _entree(**kw):
    base = dict(
        nom="matin",
        moment=datetime(2026, 8, 2, 7, 0),
        demande="résume-moi mes mails",
        texte="Trois mails, rien d'urgent.",
        outils=["webSearch"],
        ecartes=[],
        duree_sec=4.2,
        erreur=None,
    )
    base.update(kw)
    return Entree(**base)


# ── It lands where the user will look ─────────────────────────────────


def test_a_run_is_written_under_its_own_day(tmp_path):
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree())

    assert journal_path(cfg, datetime(2026, 8, 2)).exists()


def test_the_day_is_the_filename(tmp_path):
    """Sorting a directory listing should sort the mornings."""
    assert journal_path(_Cfg(tmp_path), datetime(2026, 8, 2)).name == "2026-08-02.md"


def test_the_write_up_is_in_it(tmp_path):
    """The point of the whole feature. Without this the routine ran for
    nobody."""
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(texte="Trois mails, rien d'urgent."))

    assert "Trois mails, rien d'urgent." in read_day(cfg, datetime(2026, 8, 2))


def test_what_was_asked_for_is_in_it(tmp_path):
    """A month later, "matin" alone does not say what it was for."""
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(demande="résume-moi mes mails"))

    assert "résume-moi mes mails" in read_day(cfg, datetime(2026, 8, 2))


def test_the_hour_it_actually_ran_is_in_it(tmp_path):
    """Not the hour it was due. A run deferred by two hours because the
    machine was asleep is a different fact."""
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(moment=datetime(2026, 8, 2, 9, 13)))

    assert "09:13" in read_day(cfg, datetime(2026, 8, 2))


# ── Appending, not rewriting ──────────────────────────────────────────


def test_a_second_run_joins_the_first(tmp_path):
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(nom="matin", texte="premier"))
    append_run(cfg, _entree(nom="revue", moment=datetime(2026, 8, 2, 9, 0),
                            texte="deuxième"))

    day = read_day(cfg, datetime(2026, 8, 2))
    assert "premier" in day and "deuxième" in day


def test_two_runs_do_not_run_into_each_other(tmp_path):
    """A line break is not a paragraph break. Without a blank line the
    second entry's heading lands directly under the first one's timing
    note, on screen and in anything that renders the Markdown, and this
    page is the delivery rather than a log of it."""
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(nom="matin", duree_sec=6.4))
    append_run(cfg, _entree(nom="revue", moment=datetime(2026, 8, 2, 9, 0)))

    assert "\n\n## 09:00 — revue" in read_day(cfg, datetime(2026, 8, 2))


def test_the_day_heading_is_written_once(tmp_path):
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree())
    append_run(cfg, _entree(nom="revue"))

    assert read_day(cfg, datetime(2026, 8, 2)).count("# 2026-08-02") == 1


def test_another_day_is_another_file(tmp_path):
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(moment=datetime(2026, 8, 2, 7, 0), texte="lundi"))
    append_run(cfg, _entree(moment=datetime(2026, 8, 3, 7, 0), texte="mardi"))

    assert "mardi" not in read_day(cfg, datetime(2026, 8, 2))
    assert "lundi" not in read_day(cfg, datetime(2026, 8, 3))


# ── What did not happen is also news ──────────────────────────────────


def test_a_refusal_is_recorded_with_its_reason(tmp_path):
    """"She did nothing" and "she was stopped" are different facts, and
    only one of them means the envelope needs widening."""
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(
        ecartes=[("localFiles", "il ne se contente pas de lire")],
    ))

    day = read_day(cfg, datetime(2026, 8, 2))
    assert "localFiles" in day
    assert "il ne se contente pas de lire" in day


def test_a_run_that_failed_says_so(tmp_path):
    """A silent gap in the journal reads as "it didn't fire", which sends
    the user looking at the schedule instead of at the error."""
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(texte=None, erreur="le modèle n'a pas répondu"))

    assert "le modèle n'a pas répondu" in read_day(cfg, datetime(2026, 8, 2))


def test_a_run_that_produced_nothing_still_leaves_a_line(tmp_path):
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(texte=None, erreur=None))

    assert "matin" in read_day(cfg, datetime(2026, 8, 2))


def test_the_tools_it_reached_for_are_named(tmp_path):
    cfg = _Cfg(tmp_path)

    append_run(cfg, _entree(outils=["webSearch", "fetchWebPage"]))

    day = read_day(cfg, datetime(2026, 8, 2))
    assert "webSearch" in day and "fetchWebPage" in day


# ── It never takes the routine down with it ───────────────────────────


def test_an_unwritable_directory_does_not_raise(tmp_path):
    """The work is already done by the time this runs. Losing the letter
    is bad; losing the letter *and* raising into the runner is worse."""
    cfg = _Cfg(tmp_path)
    blocked = tmp_path / "yuba" / "journal"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("je suis un fichier, pas un dossier", encoding="utf-8")

    append_run(cfg, _entree())  # must not raise


def test_reading_a_day_that_was_never_written_is_empty_not_an_error(tmp_path):
    assert read_day(_Cfg(tmp_path), datetime(2026, 8, 2)) == ""


# ── Ninety days, and not a line the user wrote ────────────────────────


def test_old_mornings_are_dropped(tmp_path):
    cfg = _Cfg(tmp_path)
    old = datetime.now() - timedelta(days=200)

    append_run(cfg, _entree(moment=old))
    prune_journal(cfg, 90)

    assert not journal_path(cfg, old).exists()


def test_recent_mornings_are_kept(tmp_path):
    cfg = _Cfg(tmp_path)
    recent = datetime.now() - timedelta(days=3)

    append_run(cfg, _entree(moment=recent))
    prune_journal(cfg, 90)

    assert journal_path(cfg, recent).exists()


def test_anything_that_is_not_a_dated_page_is_left_alone(tmp_path):
    """The folder is in the user's own directory. They may well drop
    their own notes in it, and a sweep that deletes those is a sweep that
    gets the whole feature turned off."""
    cfg = _Cfg(tmp_path)
    old = datetime.now() - timedelta(days=200)
    append_run(cfg, _entree(moment=old))
    mine = journal_path(cfg, old).parent / "mes-notes.md"
    mine.write_text("à moi", encoding="utf-8")

    # The sweep really did run: it took the page next to this one.
    assert prune_journal(cfg, 90) == 1
    assert mine.exists()


def test_pruning_nothing_is_not_an_error(tmp_path):
    assert prune_journal(_Cfg(tmp_path), 90) == 0


def test_the_sweep_that_prunes_reminders_prunes_the_journal_too(tmp_path):
    """One retention answer for everything Yuba writes down about
    herself. A journal that grew forever because nothing called the
    sweep would be the same class of bug as a ledger that did."""
    from src.jarvis.memory.db import Database
    from src.jarvis.reminders import scheduler as sched

    cfg = _Cfg(tmp_path)
    cfg.reminders_enabled = True
    cfg.reminder_tick_sec = 5.0
    cfg.reminder_late_grace_sec = 900.0
    cfg.reminder_max_attempts = 60
    cfg.voice_debug = False

    append_run(cfg, _entree(moment=datetime.now() - timedelta(days=200)))

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    try:
        runner = sched.ReminderScheduler(
            db=db, cfg=cfg, speak=lambda t, on_spoken=None: True,
            busy=lambda: False, announce=lambda action, outcome: None,
        )
        for _ in range(sched._PRUNE_EVERY_TICKS):
            runner.tick()
    finally:
        db.close()

    assert not journal_path(cfg, datetime.now() - timedelta(days=200)).exists()
