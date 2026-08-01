"""
⏱️ The thread that keeps the promise.

Wall clock, not monotonic — the opposite of the confirmation TTL, for the
mirror reason. `PendingAction.has_expired` uses `time.monotonic()`
because a deadline that can move backwards is a resurrectable approval. A
confirmation TTL is an attention span; a reminder is an appointment with
the world. Polling against the wall clock is correct across a sleep by
construction: the comparison happens at tick time, so a laptop shut from
09:00 to 14:00 finds the row due the moment it wakes.

Its own thread rather than the daemon's main loop, which calls the diary
pass synchronously and can block for up to 45 seconds — a 09:00 reminder
would land anywhere inside that, and never during shutdown.

The tick body is deliberately trivial: one indexed SELECT, a state
change, a hand-off. Everything slow stays on the paths that already own
it.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..debug import debug_log

# How often to sweep old settled rows, in ticks. Here rather than on a
# tab being opened: the ledger's own 90-day promise is only kept for
# users who open the Activity tab, and this table does not inherit that.
_PRUNE_EVERY_TICKS = 720


class ReminderScheduler:
    """Says what is owed, once, and only counts it once it was heard."""

    def __init__(self, *, db, cfg, speak: Callable[..., Any],
                 busy: Callable[[], bool],
                 announce: Callable[[Any, str], None]):
        self._db = db
        self._cfg = cfg
        self._speak = speak
        self._busy = busy
        self._announce = announce
        self._wake = threading.Event()
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self._ticks = 0

    # ── The thread ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="jarvis-reminders", daemon=True,
        )
        self._thread.start()
        debug_log("reminder scheduler started", "tools")

    def stop(self) -> None:
        """Stop and join.

        Called before the speaker dies. The shutdown diary pass can take
        45 seconds after that and the database outlives both, so a
        reminder firing in that window would be marked said with nothing
        able to say it.
        """
        self._stopping = True
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        debug_log("reminder scheduler stopped", "tools")

    def nudge(self) -> None:
        """Wake early — a reminder closer than one tick was just set."""
        self._wake.set()

    def _run(self) -> None:
        tick_sec = float(getattr(self._cfg, "reminder_tick_sec", 5.0))
        while not self._stopping:
            self._wake.wait(tick_sec)
            self._wake.clear()
            if self._stopping:
                return
            self.tick()

    # ── One sweep ─────────────────────────────────────────────────────

    def tick(self) -> None:
        """Say whatever is owed. Never raises: it is the only thing
        keeping promises, and a bad row must cost that row rather than
        every reminder after it."""
        try:
            self._tick()
        except Exception as e:
            debug_log(f"reminder tick failed, continuing: {e}", "tools")

    def _tick(self) -> None:
        if not bool(getattr(self._cfg, "reminders_enabled", True)):
            return

        self._ticks += 1
        if self._ticks % _PRUNE_EVERY_TICKS == 0:
            try:
                self._db.prune_rappels(90)
            except Exception as e:
                debug_log(f"reminder prune skipped: {e}", "tools")
            # The journal keeps the same window as the ledger and the
            # settled rows. One retention answer for everything Yuba
            # writes down about herself is one thing to explain, and one
            # thing to change.
            try:
                from ..routines.journal import prune_journal

                prune_journal(self._cfg, 90)
            except Exception as e:
                debug_log(f"journal prune skipped: {e}", "tools")

        now = datetime.now(timezone.utc)
        # Reminders only. A routine read out here would be spoken as a
        # sentence — which is not what a routine is — and then settled,
        # so it would never run again.
        due = self._db.due_rappels(now.isoformat(), kind="rappel")
        if not due:
            return

        if self._busy():
            # Cutting across a reply the user is waiting for is worse
            # than a few seconds late. It stays owed.
            debug_log(f"{len(due)} reminder(s) deferred: a query is running", "tools")
            return

        max_attempts = int(getattr(self._cfg, "reminder_max_attempts", 60))
        ready, exhausted = [], []
        for row in due:
            (exhausted if row["attempts"] >= max_attempts else ready).append(row)

        for row in exhausted:
            self._give_up(row, "trop de tentatives")
        if not ready:
            return

        for row in ready:
            self._db.mark_rappel_tried(row["id"], now.isoformat())

        # One utterance for all of them. Three hand-offs would talk over
        # each other: the speaker holds a single completion callback, and
        # a second call destroys the first's.
        text = _compose(ready, now, float(getattr(
            self._cfg, "reminder_late_grace_sec", 900.0,
        )))
        ids = [row["id"] for row in ready]
        taken = self._speak(text, on_spoken=lambda: self._settle(ids))

        if taken is False:
            # Definitive: no engine will ever say this. Retrying every
            # tick would spin, and leaving it owed would promise
            # something that cannot happen.
            for row in ready:
                self._give_up(row, "rien ne peut le prononcer")

    def _settle(self, ids: list) -> None:
        """Delivery, not queueing, is what settles a promise.

        A crash between the hand-off and the speech costs a repeat rather
        than a broken promise, which is the right way round.
        """
        said = datetime.now(timezone.utc).isoformat()
        for rappel_id in ids:
            try:
                row = _row(self._db, rappel_id)
                self._db.settle_rappel(rappel_id, said_utc=said)
                self._record(row, "ok")
            except Exception as e:
                debug_log(f"reminder not settled: {e}", "tools")

    def _give_up(self, row: dict, why: str) -> None:
        """Close a row nothing can deliver, loudly."""
        try:
            self._db.settle_rappel(row["id"], said_utc=None)
            self._record(row, "échec")
            self._announce(row, why)
            debug_log(f"    ⏰ reminder abandoned ({why}): {row['id']}", "tools")
        except Exception as e:
            debug_log(f"reminder not abandoned cleanly: {e}", "tools")

    def _record(self, row: Optional[dict], outcome: str) -> None:
        if row is None:
            return
        try:
            self._db.record_action(
                tool="rappel", args={}, risk="lecture", verdict="libre",
                outcome=outcome, duration_ms=None, origin="rappel",
                query=None, request_id=row["id"],
            )
        except Exception as e:
            debug_log(f"reminder not recorded: {e}", "tools")


def _row(db, rappel_id: str) -> Optional[dict]:
    for row in db.all_rappels():
        if row.get("id") == rappel_id:
            return row
    return None


def _compose(rows: list, now: datetime, grace_sec: float) -> str:
    """What she says. Written by code, like every other promise-bearing
    sentence in this project.

    Past the grace window it carries the lateness rather than pretending
    the moment is now — a reminder owed five hours ago and delivered as
    if it were due is worse than one that admits it.
    """
    parts = []
    for row in rows:
        late = _lateness(row, now, grace_sec)
        parts.append(f"{row['texte']}{late}")
    if len(parts) == 1:
        return f"Rappel : {parts[0]}."
    joined = " ; ".join(parts)
    return f"Rappels : {joined}."


def _lateness(row: dict, now: datetime, grace_sec: float) -> str:
    try:
        due = datetime.fromisoformat(row["due_utc"])
    except Exception:
        return ""
    behind = (now - due).total_seconds()
    if behind <= grace_sec:
        return ""
    hours = int(behind // 3600)
    if hours >= 24:
        return f" (en retard de {hours // 24} jour(s))"
    if hours >= 1:
        return f" (en retard de {hours} heure(s))"
    return f" (en retard de {int(behind // 60)} minute(s))"
