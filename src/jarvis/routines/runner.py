"""
🌅 Where a row becomes a morning.

The runner takes one due `routine` row and turns it into a turn of the
reply engine. Almost everything worth saying about it is a thing it
deliberately does not touch.

It never takes the query lock. Holding it would mean a routine that
reached a slow server at 07:00 leaves Yuba unresponsive until it
finishes: the user talks and nothing happens. The dispatcher *asks*
whether a query is running and defers; the runner holds nothing the user
needs.

It never uses the shared dialogue memory. A routine is not a
conversation. Its turn must not land in the user's history, must not move
the hot window, and above all must not disturb a confirmation card
already waiting there — the user going to bed with a question pending and
waking to find it silently expired is the worst thing this feature could
do to them.

It advances the row *before* the work, which is the exact opposite of the
reminder scheduler. A reminder that fails is still owed and stays owed. A
routine that takes the process down with it and is still owed comes back
on the next tick, does it again, and does that forever. So here a crash
costs one morning.

And it runs one at a time. Two routines at 07:00 sharing one small model,
one rate limit and one machine is not twice the work; it is two slower
runs and a good chance neither finishes.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from ..debug import debug_log
from ..tools.policy import (
    FREE,
    OUTCOME_FAILED,
    OUTCOME_OK,
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_STARTED,
    RISK_READ,
)
from .journal import Entree, append_run
from .recurrence import Regle, next_occurrence
from .scope import block_for

ORIGIN = "routine"

# The ledger lists tool calls, and a run is not one. The prefix keeps a
# row reading `matin` from inventing a tool the user cannot look up.
LEDGER_PREFIX = "routine:"

# What the ledger's outcome means, said in a sentence, because "refusé"
# in a morning write-up tells the user nothing about which file to open.
_POURQUOI = {
    OUTCOME_OUT_OF_SCOPE: "hors du périmètre de cette routine",
    "refusé": "tu as demandé à ne jamais le lancer",
    "demandé": "il aurait fallu te le demander, et tu dormais",
    OUTCOME_FAILED: "il n'a pas abouti",
}


class RoutineRunner:
    """One routine at a time, on its own thread."""

    def __init__(self, db, cfg, *, engine: Optional[Callable] = None,
                 announce: Optional[Callable[..., None]] = None) -> None:
        self._db = db
        self._cfg = cfg
        self._engine = engine
        self._announce = announce
        self._slot = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()

    # ── The slot ──────────────────────────────────────────────────────

    def busy(self) -> bool:
        return self._slot.locked()

    def submit(self, row: dict) -> bool:
        """Start a run, or say the slot is taken.

        Refusing is not dropping: the row keeps its due time, so the
        dispatcher finds it again on the next tick.
        """
        if self._stopping.is_set():
            return False
        if not self._slot.acquire(blocking=False):
            debug_log("routine deferred: one is already running", "tools")
            return False

        self._thread = threading.Thread(
            target=self._run_and_release, args=(dict(row),),
            name="routine-runner", daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    # ── One run ───────────────────────────────────────────────────────

    def _run_and_release(self, row: dict) -> None:
        try:
            self._run(row)
        except Exception as e:
            # The slot matters more than this run. A runner that stayed
            # busy after one bad morning would suspend every routine
            # until the next restart.
            debug_log(f"routine run failed outright: {e}", "tools")
        finally:
            self._slot.release()

    def _run(self, row: dict) -> None:
        rid = row.get("id") or ""
        if row.get("etat") != getattr(self._db, "ETAT_PENDING", "prévu"):
            # Cancelled between the dispatcher reading it and this
            # thread starting. Cancelling has to be final.
            debug_log(f"routine {rid} no longer owed, skipping", "tools")
            return

        payload = payload_of(row)
        nom = str(payload.get("nom") or "").strip()
        phrase = str(row.get("texte") or "").strip()
        started_at = datetime.now()
        started_iso = datetime.now(timezone.utc).isoformat()
        began = time.monotonic()

        self._record(nom, OUTCOME_STARTED, rid, phrase)
        self._advance(row, payload)

        bloc = block_for(self._cfg, nom) if nom else None
        # The file is the control surface, for the sentence as much as
        # for the envelope. `routines.md` shows `phrase:` beside
        # `outils:`, and a field that changes nothing when edited is
        # worse than no field at all: the user corrects a mis-heard word,
        # watches the same wrong summary arrive tomorrow, and cannot tell
        # which of the two they got wrong. The row keeps what was
        # originally asked, and is the fallback for a half-written block.
        if bloc is not None and bloc.phrase.strip():
            phrase = bloc.phrase.strip()

        # The schedule line is a mirror of the rule, and an editor buffer
        # saved after a rewrite puts the two out of step with nothing
        # anywhere able to see it. Both strings are composed by this code,
        # so comparing them is not a language pattern.
        desaccord = None
        if bloc is not None:
            attendu = quand_fichier_pour(payload)
            if attendu and bloc.quand.strip() != attendu:
                desaccord = (
                    f"routines.md dit « {bloc.quand.strip()} », "
                    f"elle tourne sur « {attendu} »"
                )

        scope = bloc.scope() if bloc is not None else None
        if scope is None:
            # The block was deleted, which is the off switch a user can
            # reach with a text editor. Journalled rather than silent:
            # silence reads as "it never fired", which sends them to look
            # at the schedule instead of at the block.
            #
            # `count=False` is the whole point. A suspended routine has
            # not failed, it is switched off, and counting these mornings
            # would have the dispatcher cancel the row after five of
            # them. A cancelled row is final, so a fortnight away would
            # destroy the routine under a message blaming it for
            # producing nothing.
            self._settle(
                nom, rid, phrase, started_at, began, payload,
                erreur=(
                    f"aucun bloc « {nom} » dans routines.md — elle est "
                    f"suspendue jusqu'à ce qu'il revienne"
                ),
                outcome=OUTCOME_OUT_OF_SCOPE, texte=None, count=False,
            )
            return

        texte, erreur = self._ask_the_engine(phrase, scope)
        self._settle(
            nom, rid, phrase, started_at, began, payload,
            erreur=erreur, texte=texte,
            outcome=OUTCOME_OK if texte else OUTCOME_FAILED,
            since=started_iso, absents=missing_from_catalogue(scope),
            desaccord=desaccord,
        )

    def _ask_the_engine(self, phrase: str, scope) -> tuple:
        from ..memory.conversation import DialogueMemory

        engine = self._engine
        if engine is None:
            from ..reply.engine import run_reply_engine

            engine = run_reply_engine

        try:
            texte = engine(
                db=self._db, cfg=self._cfg, tts=None, text=phrase,
                # Its own, every time. A routine is not a conversation,
                # and this is what keeps its turn out of the user's.
                dialogue_memory=DialogueMemory(),
                origin=ORIGIN, scope=scope,
            )
            return (texte or "").strip(), None
        except Exception as e:
            debug_log(f"routine {scope.nom} did not finish: {e}", "tools")
            return "", str(e)

    # ── The row ───────────────────────────────────────────────────────

    def _advance(self, row: dict, payload: dict) -> None:
        """Move to the next occurrence before the work, not after."""
        from ..utils.time_context import to_utc_iso

        try:
            regle = Regle.from_json(payload.get("regle"))
        except Exception as e:
            debug_log(f"routine rule unreadable, not advancing: {e}", "tools")
            return

        tz = str(row.get("tz") or "")
        try:
            nxt = next_occurrence(regle, datetime.now(), tz)
            self._db.advance_rappel(
                row.get("id"), due_utc=to_utc_iso(nxt, tz),
                due_local=nxt.isoformat(), tz=tz,
            )
        except Exception as e:
            debug_log(f"routine not advanced: {e}", "tools")

    def _count(self, rid: str, payload: dict, *, productive: bool) -> None:
        """Consecutive mornings that produced nothing.

        Not for its own sake: a routine that has failed every day for a
        week is broken, and something has to be able to notice. A quiet
        morning is not one of these — "rien à signaler" is the routine
        working, and counting it would switch off exactly the ones doing
        their job.
        """
        try:
            payload["steriles"] = 0 if productive else int(payload.get("steriles", 0)) + 1
            self._db.set_rappel_payload(rid, payload)
        except Exception as e:
            debug_log(f"routine tally not written: {e}", "tools")

    # ── The record ────────────────────────────────────────────────────

    def _record(self, nom: str, outcome: str, rid: str, phrase: str,
                duration_ms: Optional[int] = None) -> None:
        try:
            self._db.record_action(
                tool=f"{LEDGER_PREFIX}{nom}", args=None, risk=RISK_READ,
                verdict=FREE, outcome=outcome, duration_ms=duration_ms,
                origin=ORIGIN, query=phrase, request_id=rid,
            )
        except Exception as e:
            debug_log(f"routine ledger row not written: {e}", "tools")

    def _tool_calls_since(self, since: Optional[str]) -> tuple:
        """What the run reached for, read back out of the ledger.

        The engine hands back text and nothing else, so this is where the
        write-up learns what was used and what was turned away. Reading
        it back also means the journal cannot claim a call the ledger
        never recorded.
        """
        outils, ecartes = [], []
        if not since:
            return outils, ecartes
        try:
            for r in self._db.recent_actions(200):
                if r.get("origin") != ORIGIN or r.get("ts_utc", "") < since:
                    continue
                tool = str(r.get("tool") or "")
                if tool.startswith(LEDGER_PREFIX):
                    continue
                if r.get("outcome") == OUTCOME_OK:
                    if tool not in outils:
                        outils.append(tool)
                else:
                    pourquoi = _POURQUOI.get(str(r.get("outcome")), str(r.get("outcome")))
                    if (tool, pourquoi) not in ecartes:
                        ecartes.append((tool, pourquoi))
        except Exception as e:
            debug_log(f"routine tool read-back failed: {e}", "tools")
        return outils, ecartes

    def _settle(self, nom, rid, phrase, started_at, began, payload, *,
                erreur, texte, outcome, since: Optional[str] = None,
                absents: Optional[List[str]] = None, count: bool = True,
                desaccord: Optional[str] = None) -> None:
        outils, ecartes = self._tool_calls_since(since)
        # A name the envelope holds but the catalogue no longer does is
        # dropped by the engine before a single ledger row exists, so the
        # read-back above cannot know about it. Unsaid, the model answers
        # in prose anyway, the run counts as productive, and the routine
        # reports success every morning while doing a fraction of its job.
        for nom_absent in (absents or []):
            ecartes.append((nom_absent, "cet outil n'existe plus ici"))
        if desaccord:
            ecartes.append(("horaire", desaccord))
        append_run(self._cfg, Entree(
            nom=nom or "?", moment=started_at, demande=phrase,
            texte=texte or None, outils=outils, ecartes=ecartes,
            duree_sec=time.monotonic() - began, erreur=erreur,
        ))
        self._record(nom, outcome, rid, phrase,
                     duration_ms=int((time.monotonic() - began) * 1000))
        if count:
            self._count(rid, payload, productive=bool(texte))
        if not texte and count and self._announce is not None:
            # Only the mornings that did not work. One that ran and wrote
            # its page is already delivered.
            try:
                self._announce(nom or "?", erreur or "elle n'a rien produit",
                               stopped=False)
            except Exception as e:
                debug_log(f"routine trouble not announced: {e}", "tools")


def quand_fichier_pour(payload: dict) -> str:
    """The schedule line this rule would be written as, or empty."""
    from .scope import quand_fichier

    try:
        return quand_fichier(Regle.from_json(payload.get("regle")))
    except Exception:
        return ""


def missing_from_catalogue(scope) -> List[str]:
    """Names the envelope holds that no longer exist anywhere.

    An MCP server that left the config takes its tools with it. The
    engine drops those names from the turn without a word — correctly,
    there is nothing to call — but the routine then quietly does less
    than the block says it does, for months.
    """
    try:
        from ..tools.registry import BUILTIN_TOOLS, get_cached_mcp_tools

        try:
            mcp = get_cached_mcp_tools() or {}
        except Exception:
            mcp = {}
        return [n for n in scope.outils if n not in BUILTIN_TOOLS and n not in mcp]
    except Exception as e:
        debug_log(f"routine catalogue check skipped: {e}", "tools")
        return []


def payload_of(row: dict) -> dict:
    try:
        raw = json.loads(row.get("payload") or "{}")
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}
