"""One routine at a time, and never in the user's way.

The runner is where a row in `rappels` becomes a turn of the reply
engine. Almost everything interesting about it is a thing it refuses to
touch.

It does not take the query lock. Holding it would mean a routine that
reached a slow MCP server at 07:00 leaves Yuba unresponsive until it
finishes — the user talks and nothing happens. The dispatcher *checks*
whether a query is running and defers, but the runner never holds
anything the user needs.

It does not use the shared dialogue memory. A routine is not a
conversation: its turn must not land in the user's history, must not
move the hot window, and above all must not disturb a confirmation card
already waiting there for an answer.

It advances the row *before* running, which is the opposite of what the
reminder scheduler does. A reminder that fails is still owed and stays
owed. A routine that kills the process mid-run and is still owed comes
back on the next tick, kills the process again, and does that forever.
So a crash costs one morning.

And it runs one at a time. Two routines at 07:00 sharing one small
model, one rate limit and one machine is not twice the work; it is two
slower runs and a much better chance that neither finishes.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.routines.journal import read_day
from src.jarvis.routines.recurrence import Regle
from src.jarvis.routines.runner import RoutineRunner
from src.jarvis.routines.scope import RoutineScope


ROUTINES_FILE = """# Routines

## matin
phrase: résume-moi mes mails
quand: tous les jours à 07:00
outils:
- webSearch
"""


class _Cfg:
    voice_debug = False
    routine_timeout_sec = 120.0

    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


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


def _add(db, *, nom="matin", texte="résume-moi mes mails", due_minutes=-1):
    due = datetime.now(timezone.utc) + timedelta(minutes=due_minutes)
    return db.add_rappel(
        texte=texte, kind="routine", origin="voix",
        due_utc=due.isoformat(), due_local=due.isoformat(), tz="Europe/Paris",
        query=texte,
        payload={"nom": nom, "regle": Regle(kind="daily", hour=7, minute=0, weekday=None).to_json()},
    )


def _row(db, rid):
    return next(r for r in db.all_rappels() if r["id"] == rid)


def _runner(db, cfg, *, engine=None, calls=None):
    """A runner whose engine is a stand-in, run to completion inline."""
    def _default(**kw):
        if calls is not None:
            calls.append(kw)
        return "Trois mails, rien d'urgent."

    runner = RoutineRunner(db=db, cfg=cfg, engine=engine or _default)
    return runner


def _run_now(runner, db, rid):
    """Submit and wait for the slot to clear."""
    assert runner.submit(_row(db, rid)) is True
    deadline = time.monotonic() + 5.0
    while runner.busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not runner.busy(), "the run never finished"


# ── What reaches the engine ───────────────────────────────────────────


def test_the_routine_runs_its_own_sentence(db, cfg):
    calls = []
    runner = _runner(db, cfg, calls=calls)
    rid = _add(db, texte="résume-moi mes mails")

    _run_now(runner, db, rid)
    runner.stop()

    assert calls[0]["text"] == "résume-moi mes mails"


def test_it_runs_unattended(db, cfg):
    """No speech: there is nobody to speak to, and a routine that woke
    the house at 07:00 would be uninstalled by 07:01."""
    calls = []
    runner = _runner(db, cfg, calls=calls)

    _run_now(runner, db, _add(db))
    runner.stop()

    assert calls[0]["tts"] is None
    assert calls[0]["origin"] == "routine"


def test_it_carries_its_envelope(db, cfg):
    calls = []
    runner = _runner(db, cfg, calls=calls)

    _run_now(runner, db, _add(db))
    runner.stop()

    scope = calls[0]["scope"]
    assert isinstance(scope, RoutineScope)
    assert scope.nom == "matin"
    assert scope.outils == ["webSearch"]


# ── The user's conversation is not touched ────────────────────────────


def test_the_turn_does_not_land_in_the_user_s_history(db, cfg):
    """A routine is not a conversation. Its turn appearing in the hot
    window would have the next thing the user says answered in the
    context of a mail summary they never read."""
    from src.jarvis.memory.conversation import DialogueMemory

    shared = DialogueMemory()
    shared.add_message("user", "salut")
    before = len(shared.get_recent_messages())

    calls = []
    runner = _runner(db, cfg, calls=calls)
    _run_now(runner, db, _add(db))
    runner.stop()

    assert calls[0]["dialogue_memory"] is not shared
    assert len(shared.get_recent_messages()) == before


def test_a_question_already_waiting_for_an_answer_survives(db, cfg):
    """The worst case: the user clicked nothing, went to bed with a
    confirmation card pending, and a routine cleared it overnight. They
    wake to an action that silently expired."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.tools.confirmation import PendingAction

    shared = DialogueMemory()
    pending = PendingAction.create(
        tool="localFiles", args={}, risk="destructif", channel="geste",
        origin="voix", query_redacted="supprime", raised_at_turn=1,
        ttl_sec=180.0,
    )
    shared.raise_pending(pending)

    runner = _runner(db, cfg)
    _run_now(runner, db, _add(db))
    runner.stop()

    assert shared.peek_pending() is not None


# ── The row moves before the work starts ──────────────────────────────


def test_the_next_occurrence_is_set_before_the_run(db, cfg):
    """Opposite of a reminder, deliberately. A reminder that fails is
    still owed. A routine that takes the process down with it and is
    still owed comes back next tick and does it again, forever."""
    seen = {}
    rid = _add(db)
    was = _row(db, rid)["due_utc"]

    def _engine(**kw):
        seen["pendant"] = _row(db, rid)["due_utc"]
        return "fait"

    runner = RoutineRunner(db=db, cfg=cfg, engine=_engine)

    _run_now(runner, db, rid)
    runner.stop()

    assert seen["pendant"] != was


def test_a_run_that_blows_up_still_moved_the_row(db, cfg):
    def _boom(**kw):
        raise RuntimeError("le modèle n'a pas répondu")

    rid = _add(db)
    was = _row(db, rid)["due_utc"]
    runner = RoutineRunner(db=db, cfg=cfg, engine=_boom)

    _run_now(runner, db, rid)
    runner.stop()

    assert _row(db, rid)["due_utc"] != was


def test_a_cancelled_routine_does_not_run(db, cfg):
    calls = []
    rid = _add(db)
    db.cancel_rappel(rid)
    runner = _runner(db, cfg, calls=calls)

    row = _row(db, rid)
    runner.submit(row)
    deadline = time.monotonic() + 2.0
    while runner.busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()

    assert calls == []


# ── One at a time ─────────────────────────────────────────────────────


def test_a_second_routine_is_not_started_while_one_is_running(db, cfg):
    """Two runs sharing one small model, one rate limit and one machine
    is not twice the work. It is two slower runs and a good chance
    neither finishes."""
    started = threading.Event()
    release = threading.Event()
    overlap = []
    running = []

    def _slow(**kw):
        running.append(1)
        overlap.append(len(running))
        started.set()
        release.wait(timeout=5.0)
        running.pop()
        return "fait"

    runner = RoutineRunner(db=db, cfg=cfg, engine=_slow)
    first, second = _add(db), _add(db, nom="matin")

    assert runner.submit(_row(db, first)) is True
    assert started.wait(timeout=5.0)
    assert runner.submit(_row(db, second)) is False

    release.set()
    deadline = time.monotonic() + 5.0
    while runner.busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()

    assert max(overlap) == 1


def test_the_user_is_never_locked_out_while_a_routine_runs(db, cfg):
    """The lock voice and text share. A routine holding it would mean a
    slow server at 07:00 leaves Yuba unresponsive until it finishes: the
    user talks and nothing happens. The dispatcher asks whether a query
    is running and defers; the runner holds nothing they need."""
    from src.jarvis import daemon

    held = []

    def _look(**kw):
        held.append(daemon._chat_query_lock.locked())
        return "fait"

    runner = RoutineRunner(db=db, cfg=cfg, engine=_look)
    _run_now(runner, db, _add(db))
    runner.stop()

    assert held == [False]


def test_the_slot_frees_up_after_a_failure(db, cfg):
    """Otherwise one bad morning suspends every routine until restart."""
    def _boom(**kw):
        raise RuntimeError("non")

    runner = RoutineRunner(db=db, cfg=cfg, engine=_boom)
    _run_now(runner, db, _add(db))

    assert runner.submit(_row(db, _add(db))) is True
    deadline = time.monotonic() + 5.0
    while runner.busy() and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.stop()


# ── No block, no run ──────────────────────────────────────────────────


def test_a_routine_whose_block_was_deleted_does_not_run(db, cfg):
    """Deleting the block is the off switch a user can reach with a text
    editor. It has to actually switch things off."""
    calls = []
    from src.jarvis.routines.scope import routines_path

    routines_path(cfg).write_text("# Routines\n", encoding="utf-8")
    runner = _runner(db, cfg, calls=calls)

    _run_now(runner, db, _add(db))
    runner.stop()

    assert calls == []


def test_a_suspended_routine_says_so_in_the_journal(db, cfg):
    """Silence would read as "it never fired", which sends the user to
    look at the schedule rather than at the block they deleted."""
    from src.jarvis.routines.scope import routines_path

    routines_path(cfg).write_text("# Routines\n", encoding="utf-8")
    runner = _runner(db, cfg)

    _run_now(runner, db, _add(db))
    runner.stop()

    assert "matin" in read_day(cfg, datetime.now())


# ── The morning after ─────────────────────────────────────────────────


def test_the_write_up_reaches_the_journal(db, cfg):
    runner = _runner(db, cfg, engine=lambda **kw: "Trois mails, rien d'urgent.")

    _run_now(runner, db, _add(db))
    runner.stop()

    assert "Trois mails, rien d'urgent." in read_day(cfg, datetime.now())


def test_a_failure_reaches_the_journal(db, cfg):
    def _boom(**kw):
        raise RuntimeError("le modèle n'a pas répondu")

    runner = RoutineRunner(db=db, cfg=cfg, engine=_boom)

    _run_now(runner, db, _add(db))
    runner.stop()

    assert "le modèle n'a pas répondu" in read_day(cfg, datetime.now())


# ── The ledger ────────────────────────────────────────────────────────


def _outcomes(db):
    return [r["outcome"] for r in db.recent_actions(20)][::-1]


def test_a_run_is_bracketed_in_the_ledger(db, cfg):
    """The opening row is the crash marker: a run that took the process
    down with it leaves a `démarré` with nothing after it, which is the
    only trace that would exist at all."""
    from src.jarvis.tools.policy import OUTCOME_OK, OUTCOME_STARTED

    runner = _runner(db, cfg)
    _run_now(runner, db, _add(db))
    runner.stop()

    assert _outcomes(db) == [OUTCOME_STARTED, OUTCOME_OK]


def test_a_failed_run_closes_with_a_failure(db, cfg):
    from src.jarvis.tools.policy import OUTCOME_FAILED, OUTCOME_STARTED

    runner = RoutineRunner(db=db, cfg=cfg, engine=MagicMock(side_effect=RuntimeError))
    _run_now(runner, db, _add(db))
    runner.stop()

    assert _outcomes(db) == [OUTCOME_STARTED, OUTCOME_FAILED]


def test_the_ledger_names_the_routine_not_a_tool(db, cfg):
    """The Activity tab lists tool calls. A run is not one, and a row
    reading `matin` next to `webSearch` invents a tool the user has no
    way to look up."""
    runner = _runner(db, cfg)
    _run_now(runner, db, _add(db))
    runner.stop()

    assert all(r["tool"] == "routine:matin" for r in db.recent_actions(20))


# ── Counting the mornings that produced nothing ───────────────────────


def _steriles(db, rid):
    return json.loads(_row(db, rid)["payload"]).get("steriles", 0)


def test_a_run_that_produced_nothing_is_counted(db, cfg):
    """Not for its own sake: a routine that has failed every morning for
    a week is broken, and something has to be able to notice."""
    runner = RoutineRunner(db=db, cfg=cfg, engine=lambda **kw: "")
    rid = _add(db)

    _run_now(runner, db, rid)
    runner.stop()

    assert _steriles(db, rid) == 1


def test_a_quiet_morning_is_not_a_sterile_one(db, cfg):
    """"Rien à signaler" is the routine working. Counting it would
    switch off exactly the routines that are doing their job."""
    runner = RoutineRunner(db=db, cfg=cfg, engine=lambda **kw: "Rien à signaler.")
    rid = _add(db)

    _run_now(runner, db, rid)
    runner.stop()

    assert _steriles(db, rid) == 0


def test_one_good_morning_clears_the_count(db, cfg):
    rid = _add(db)
    empty = RoutineRunner(db=db, cfg=cfg, engine=lambda **kw: "")
    _run_now(empty, db, rid)
    _run_now(empty, db, rid)
    empty.stop()
    assert _steriles(db, rid) == 2

    good = RoutineRunner(db=db, cfg=cfg, engine=lambda **kw: "voilà")
    _run_now(good, db, rid)
    good.stop()

    assert _steriles(db, rid) == 0


def test_a_run_that_raised_counts_as_nothing_produced(db, cfg):
    runner = RoutineRunner(db=db, cfg=cfg, engine=MagicMock(side_effect=RuntimeError))
    rid = _add(db)

    _run_now(runner, db, rid)
    runner.stop()

    assert _steriles(db, rid) == 1
