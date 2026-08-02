"""cancelRoutine — stopping one out loud, the way it was started.

The Routines tab already has a button, and that is the honest surface: it
shows the name, the hour and the envelope, so the user stops the thing
they can see. This exists because the machinery to create a routine by
speaking has no matching way to stop one, and a capability that is easier
to grant than to revoke is one people stop granting at all.

It stops the row and leaves the block. The block is the record of what
that routine was allowed to do, which is exactly what someone wants to
read after switching it off, and leaving it is what makes saying the
sentence again cheap — `setRoutine` re-arms a block whose row is gone
rather than refusing.

Its risk is `lecture`: stopping something is the safe direction, and a
routine wrongly stopped costs one sentence to restart while a routine
wrongly left running costs every morning until someone notices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...debug import debug_log
from ...routines.runner import payload_of
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


class CancelRoutineTool(Tool):
    """Stop something Yuba has been doing on her own."""

    # Not for the row it writes — it writes none — but because the thing
    # it reaches into is Yuba's own bookkeeping, and a routine reaching
    # in to silence another routine is not something anyone asked for.
    writes_own_state = True

    def risk_for(self, args):
        """Stopping is the safe direction. A routine wrongly stopped
        costs one sentence to restart; one wrongly left running costs
        every morning until somebody notices."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "cancelRoutine"

    @property
    def description(self) -> str:
        return (
            "Stop a repeating routine Yuba has been running on her own, "
            "ONLY when the user asks to stop or remove one. Called with no "
            "name it lists what is running, so it is also how to answer "
            "'what do you do on your own?'. This never creates a routine "
            "and never cancels a reminder."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nom": {
                    "type": "string",
                    "description": (
                        "The exact name of the routine to stop, as it "
                        "appears in the user's routines file. Omit to list "
                        "what is running instead."
                    ),
                },
            },
        }

    def run(self, args: Optional[Dict[str, Any]],
            context: ToolContext) -> ToolExecutionResult:
        vivantes = _live(context.db)
        nom = str((args or {}).get("nom") or "").strip()

        if not vivantes:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "There are no routines running. Tell the user that "
                    "plainly, in their language."
                ),
            )

        if not nom:
            return self._list(vivantes)

        row = vivantes.get(nom)
        if row is None:
            # Named rather than guessed. A near-match silently stopped is
            # the wrong routine silently stopped, and the user finds out
            # a morning later at best.
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f"Nothing was stopped: there is no routine named "
                    f"« {nom} ». Tell the user, in their language, and list "
                    f"what is actually running so they can pick: "
                    f"{_names(vivantes)}."
                ),
            )

        try:
            stopped = bool(context.db.cancel_rappel(row["id"]))
        except Exception as e:
            debug_log(f"routine not cancelled: {e}", "tools")
            stopped = False

        if not stopped:
            # Reported honestly. Saying it is stopped while it still fires
            # is the worst answer available here.
            return ToolExecutionResult(
                success=False,
                reply_text=(
                    f"Nothing was stopped: « {nom} » could not be cancelled. "
                    f"Tell the user plainly, in their language, and that it "
                    f"will run again as scheduled."
                ),
            )

        debug_log(f"    🌅 routine {nom} stopped by the user", "tools")
        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Routine « {nom} » stopped. Its block stays in routines.md, "
                f"so the user can still read what it was allowed to do, and "
                f"saying the same request again restarts it. Tell them "
                f"exactly that, in their language."
            ),
        )

    def _list(self, vivantes: Dict[str, dict]) -> ToolExecutionResult:
        lignes = "; ".join(
            f"{nom} ({_when(row)}): {row.get('texte') or ''}"
            for nom, row in vivantes.items()
        )
        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Routines currently running: {lignes}. Read them back to "
                f"the user in their language and ask which to stop, if that "
                f"is what they wanted. Nothing has been stopped yet."
            ),
        )


def _live(db) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    try:
        for row in db.pending_rappels(kind="routine"):
            nom = str(payload_of(row).get("nom") or "")
            if nom:
                out[nom] = row
    except Exception as e:
        debug_log(f"routine rows unreadable: {e}", "tools")
    return out


def _names(vivantes: Dict[str, dict]) -> str:
    return ", ".join(vivantes) or "none"


def _when(row: dict) -> str:
    from datetime import datetime

    try:
        return f"{datetime.fromisoformat(str(row.get('due_local'))):%d/%m à %H:%M}"
    except Exception:
        return str(row.get("due_local") or "")
