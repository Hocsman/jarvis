"""setReminder — the only way something gets said later.

Named at a distance from `remember` on purpose. The defect this closes is
the confusion between the two: with nowhere to put "remind me to call the
dentist tomorrow", the model reached for the memory tool and filed the
task as a permanent fact about the user. `remindMe` would share a prefix
with the tool whose traffic it is taking over, in a catalogue the router
matches on.

The invariant is stronger than confirming what was written: the tool
writes the row, reads it back from the database, re-parses it, and speaks
that. The sentence the user hears is derived from the artefact that will
actually fire — not from the model's JSON, and not from what the tool
believed a moment ago. A write that did not land cannot be confirmed,
because there is nothing to read back.

Saying the time back is also the whole disambiguation strategy. A model
that read "jeudi" as Tuesday is caught by the user's ear, in the same
turn, in any language, with no word list anywhere.

Failure is noisy and immediate, which is neither of the usual directions.
`read_approval` fails closed because one extra "no" costs a turn. Here
both silent directions break a promise — and the user is standing there
having spoken two seconds ago, so a clarifying sentence costs nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ...debug import debug_log
from ...reminders.extract import (
    ExtractionFailed,
    extract_reminder_time,
    reminder_channel_available,
)
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class SetReminderTool(Tool):
    """Say something back to the user at a time they choose."""

    # Writes Yuba's own state. The risk below stays `lecture`
    # because an attended turn is correctable in the next breath;
    # this carries what that reasoning assumes, for a caller running
    # with nobody there to correct anything.
    writes_own_state = True

    def risk_for(self, args):
        """Writes, but only into Yuba's own files and database: text the
        user can reopen and correct by hand, never their machine and
        never anything outward."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "setReminder"

    @property
    def description(self) -> str:
        # Whole sentences, restriction first: the router reads only the
        # opening of this, so an invitation with the limit tacked on the
        # end would reach it as an unrestricted invitation.
        return (
            "Arrange to tell the user something at a later time, ONLY when "
            "they ask to be told later ('remind me to...', 'in twenty "
            "minutes...', 'on Thursday, tell me to...'). This is for things "
            "to be said at a moment, never for facts to be kept about the "
            "user — those are a different tool. A question about the past "
            "is not a request for a reminder."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        # One optional property, following log_meal's reasoning. It is
        # what makes the planner's fast path work: `setReminder
        # rappel='…'` is a concrete key='value', so the step dispatches
        # without the resolver — which would otherwise be asked to
        # hallucinate a timestamp, the exact failure this avoids.
        return {
            "type": "object",
            "properties": {
                "rappel": {
                    "type": "string",
                    "description": (
                        "The user's request, copied verbatim, including "
                        "when they said it should happen."
                    ),
                },
            },
        }

    def run(self, args: Optional[Dict[str, Any]],
            context: ToolContext) -> ToolExecutionResult:
        from ...utils.time_context import local_timezone_name, to_utc_iso

        cfg = context.cfg
        utterance = str((args or {}).get("rappel") or "").strip()
        if not utterance:
            utterance = (context.redacted_text or "").strip()
        if not utterance:
            return self._refuse("je n'ai pas vu ce qu'il fallait te rappeler")

        if not reminder_channel_available(cfg):
            return self._refuse(
                "je n'ai aucun modèle pour lire l'heure que tu m'as donnée"
            )

        try:
            reading = extract_reminder_time(cfg, utterance)
        except ExtractionFailed as e:
            debug_log(f"    ⏰ reminder not set: {e}", "tools")
            return self._refuse(str(e))

        tz = local_timezone_name()
        try:
            due_utc = to_utc_iso(reading.due_local, tz)
        except Exception as e:
            return self._refuse(f"je n'ai pas su placer ce moment : {e}")

        rappel_id = context.db.add_rappel(
            texte=reading.what,
            due_utc=due_utc,
            due_local=reading.due_local.strftime("%Y-%m-%dT%H:%M"),
            tz=tz,
            origin=getattr(context, "origin", None),
            query=utterance,
        )

        # The round trip. What she says next is derived from the row that
        # will fire, so a write that landed wrong cannot be confirmed as
        # right — and one that did not land cannot be confirmed at all.
        stored = _read_back(context.db, rappel_id)
        when = None
        if stored is not None:
            try:
                when = datetime.fromisoformat(stored["due_local"])
            except Exception:
                when = None

        if when is None:
            # She cannot describe it, so she must not claim it — and it
            # must not fire either. A reminder going off after she said
            # it was not set is worse than no reminder at all, so the row
            # is withdrawn rather than left to surprise them.
            context.db.cancel_rappel(rappel_id)
            debug_log("    ⏰ reminder withdrawn: unreadable round trip", "tools")
            return self._refuse("je n'ai pas réussi à le relire correctement")

        # A reminder closer than one tick must not wait for the next
        # boundary to be noticed.
        try:
            from ...daemon import nudge_reminders

            nudge_reminders()
        except Exception as e:
            debug_log(f"scheduler not nudged: {e}", "tools")

        said = _confirmation(when, stored["texte"], reading.hour_was_assumed)
        debug_log(f"    ⏰ reminder set for {stored['due_utc']}", "tools")
        return ToolExecutionResult(success=True, reply_text=said)

    def _refuse(self, reason: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            success=False,
            reply_text=(
                f"Nothing was scheduled: {reason}. Tell the user plainly, in "
                f"their language, and ask for the time again if that is what "
                f"was missing. Do not claim a reminder was set."
            ),
        )


def _read_back(db, rappel_id: str) -> Optional[dict]:
    try:
        for row in db.all_rappels():
            if row.get("id") == rappel_id:
                return row
    except Exception as e:
        debug_log(f"reminder read-back failed: {e}", "tools")
    return None


def _confirmation(when: datetime, what: str, hour_was_assumed: bool) -> str:
    """What she says back, written by code rather than by the model.

    The time is the point: it is what lets the user hear that "jeudi" was
    read as Tuesday and say so, in the same turn.
    """
    stamp = f"{when:%d/%m à %H:%M}"
    if hour_was_assumed:
        return (
            f"Reminder set for {stamp} — you did not say an hour, so it is "
            f"set for the morning. Tell the user exactly that, in their "
            f"language, including the date and time, and that they can "
            f"change it: « {what} »"
        )
    return (
        f"Reminder set for {stamp}. Tell the user, in their language, "
        f"repeating the date and time back so they can correct you: "
        f"« {what} »"
    )
