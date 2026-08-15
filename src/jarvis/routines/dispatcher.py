"""
🕰️ Who decides that a morning has arrived, and who decides it has passed.

One indexed SELECT and a hand-off, on its own thread. Its own
deliberately: the reminder scheduler is the only thing keeping spoken
promises, and it must not share a failure surface with a newer and more
complicated feature.

Three decisions live here, and each leans the same way.

**Late is not missed.** A digest two hours late is still the thing that
was asked for; the same digest at 18:00 is not. Past the window the
occurrence is skipped and the row moves on, because a routine that keeps
arriving at the wrong time is one the user learns to ignore.

**A run it cannot start is not a run it drops.** Busy, or a run already
in flight, leaves the row exactly where it was, so the next tick finds it
again. The staleness window is what eventually ends that, rather than a
counter — the decision stays "is this still worth doing" rather than
becoming "how many times have I tried".

**A routine that has produced nothing for days is broken, and stops.**
Loudly, on a journal page, because the failure being guarded against is a
morning digest that silently stopped arriving in October and was noticed
in December.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..debug import debug_log
from ..tools.policy import FREE, OUTCOME_EXPIRED, OUTCOME_REFUSED, RISK_READ
from .journal import Entree, append_run
from .recurrence import RUN, Regle, next_occurrence, should_run
from .runner import LEDGER_PREFIX, ORIGIN, payload_of


class RoutineDispatcher:
    """Finds what is owed this minute and hands it over, one at a time."""

    def __init__(self, *, db, cfg, runner,
                 busy: Optional[Callable[[], bool]] = None,
                 announce: Optional[Callable[..., None]] = None) -> None:
        self._db = db
        self._cfg = cfg
        self._runner = runner
        self._busy = busy or (lambda: False)
        self._announce = announce
        self._wake = threading.Event()
        self._stopping = False
        self._thread: Optional[threading.Thread] = None

    # ── The thread ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="jarvis-routines", daemon=True,
        )
        self._thread.start()
        debug_log("routine dispatcher started", "tools")

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        debug_log("routine dispatcher stopped", "tools")

    def nudge(self) -> None:
        """Wake early — a routine closer than one tick was just created."""
        self._wake.set()

    def _loop(self) -> None:
        tick_sec = float(getattr(self._cfg, "routine_tick_sec", 30.0))
        while not self._stopping:
            self._wake.wait(tick_sec)
            self._wake.clear()
            if self._stopping:
                return
            self.tick()

    # ── One sweep ─────────────────────────────────────────────────────

    def tick(self) -> None:
        """Never raises. A bad row costs that row, not every routine
        after it."""
        try:
            self._tick()
        except Exception as e:
            debug_log(f"routine tick failed, continuing: {e}", "tools")

    def _tick(self) -> None:
        if not bool(getattr(self._cfg, "routines_enabled", True)):
            return

        now = datetime.now(timezone.utc).isoformat()
        for row in self._db.due_rappels(now, kind="routine"):
            try:
                self._consider(dict(row), now)
            except Exception as e:
                debug_log(f"routine row skipped: {e}", "tools")

    def _consider(self, row: dict, now: str) -> None:
        rid = row.get("id") or ""
        payload = payload_of(row)
        nom = str(payload.get("nom") or "?")

        try:
            regle = Regle.from_json(payload.get("regle"))
        except Exception as e:
            # It can never be advanced, so it is due forever: every tick,
            # for the life of the process. A routine that cannot say when
            # it runs must not exist.
            self._disarm(row, nom, f"je n'ai pas su relire sa récurrence ({e})")
            return

        cap = int(getattr(self._cfg, "routine_max_steriles", 5))
        steriles = int(payload.get("steriles", 0) or 0)
        if steriles >= cap:
            self._disarm(
                row, nom,
                f"{steriles} passages de suite sans rien produire",
                outcome=OUTCOME_REFUSED,
            )
            return

        if should_run(
            str(row.get("due_utc") or ""), now, regle,
            cap=float(getattr(self._cfg, "routine_late_grace_sec", 14400.0)),
        ) != RUN:
            self._pass_over(row, regle, nom)
            return

        if self._busy():
            # Not about deadlock — the runner never holds the query lock.
            # It is about the user's own reply getting slower because
            # something they did not ask for is competing for the model.
            debug_log(f"routine {nom} deferred: a query is running", "tools")
            return

        # Refused means the slot is taken. The row keeps its due time, so
        # the next tick finds it again.
        self._runner.submit(row)

    # ── The two ways a morning ends without running ───────────────────

    def _pass_over(self, row: dict, regle: Regle, nom: str) -> None:
        """Too late to be worth doing. Move on and say so."""
        if not self._advance(row, regle):
            # The same reasoning already applied to a rule that cannot be
            # read: a row that cannot be moved is due forever.
            self._disarm(row, nom, "je n'ai pas su placer son prochain passage")
            return
        self._write(
            row, nom,
            "ce passage était trop en retard pour être encore utile",
            OUTCOME_EXPIRED,
        )

    def _disarm(self, row: dict, nom: str, pourquoi: str,
                outcome: str = OUTCOME_EXPIRED) -> None:
        """Stop it for good, loudly.

        Cancelling rather than suspending: the block stays in
        `routines.md`, so saying the sentence again costs one breath,
        while a row that keeps being reconsidered every tick costs
        forever.
        """
        try:
            self._db.cancel_rappel(row.get("id"))
        except Exception as e:
            debug_log(f"routine not cancelled: {e}", "tools")
        debug_log(f"    🕰️ routine {nom} stopped: {pourquoi}", "tools")
        self._write(row, nom, f"routine arrêtée : {pourquoi}", outcome)
        if self._announce is not None:
            # A journal page alone would be found weeks later, and by
            # then the digest has been missing the whole time.
            try:
                self._announce(nom, pourquoi, stopped=True)
            except Exception as e:
                debug_log(f"routine stop not announced: {e}", "tools")

    def _write(self, row: dict, nom: str, pourquoi: str, outcome: str) -> None:
        append_run(self._cfg, Entree(
            nom=nom, moment=datetime.now(),
            demande=str(row.get("texte") or ""), texte=None, erreur=pourquoi,
        ))
        try:
            self._db.record_action(
                tool=f"{LEDGER_PREFIX}{nom}", args=None, risk=RISK_READ,
                verdict=FREE, outcome=outcome, origin=ORIGIN,
                query=str(row.get("texte") or ""), request_id=row.get("id"),
            )
        except Exception as e:
            debug_log(f"routine ledger row not written: {e}", "tools")

    def _advance(self, row: dict, regle: Regle) -> bool:
        """False when the row did not move, and so is owed forever."""
        from ..utils.time_context import now_in, to_utc_iso

        tz = str(row.get("tz") or "")
        try:
            # The row's own zone on both sides — see the runner's twin.
            nxt = next_occurrence(regle, now_in(tz), tz)
            self._db.advance_rappel(
                row.get("id"), due_utc=to_utc_iso(nxt, tz),
                due_local=nxt.isoformat(), tz=tz,
            )
        except Exception as e:
            debug_log(f"routine not advanced: {e}", "tools")
            return False
        return True
