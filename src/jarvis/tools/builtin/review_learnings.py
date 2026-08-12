"""
🧠 What she thinks she noticed, for him to accept or strike out.

One tool, and it only ever runs when he asks. It does two things in one
call, in this order: it harvests whatever he has ticked in `appris.md`
since last time, then it reads the journal rows he has not been asked
about and writes new proposals into the page.

The order is what makes the file feel alive rather than administrative.
He ticks a line, forgets it, asks again a week later, and what he agreed
to lands before she offers him anything new.

`lecture` and free: this is the answer to "what have you noticed?", and a
question that costs a card is a question nobody asks. It writes his
profile only through the harvest, whose input is a checkbox he ticked by
hand — there is no sentence any model can emit that reaches `profil.md`
through this call.

`reads_his_life` keeps it out of an unattended pass. The hazard is not
that it writes; it is that it reads a fortnight of his days and forms an
opinion about him, which is only defensible because he is in the room to
say no.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...appris.page import (
    SECTION_PROFIL,
    SECTION_REGLES,
    ajouter_propositions,
    load_appris,
)
from ...appris.recolte import recolter
from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult

_VERS_LA_PAGE = {"fait": SECTION_PROFIL, "regle": SECTION_REGLES}


class ReviewLearningsTool(Tool):
    """Read the journal for things he never said outright, and propose."""

    # Not that it writes her memory — that it reads his life.
    reads_his_life = True

    def risk_for(self, args):
        """Reading, so free.

        A tool that declares nothing falls to the gate's default, which
        is destructive, and destructive is settled by a click on a card
        and never by a spoken word. By voice that makes the feature
        unreachable, and unreachable silently: the router stops offering
        it and nothing says why.

        Free is the honest price. Its own writes go through the harvest,
        whose input is a checkbox he ticked by hand in his own editor —
        there is no sentence any model can emit that reaches `profil.md`
        through this call. And a question that costs a card is a
        question nobody asks.
        """
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "reviewLearnings"

    @property
    def description(self) -> str:
        # The router reads only the leading sentences that fit its
        # budget, so what discriminates this tool from `remember` and
        # `listGoals` has to be in the first two.
        return (
            "Read the user's own journal for things about them they never "
            "said outright, and propose each as a question in a file they "
            "answer by hand. Use when they ask what you have noticed or "
            "learnt about them. It proposes only: nothing is remembered "
            "until they tick it. Not for storing something they just told "
            "you, which is remember."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        from ...appris import propose as _propose

        cfg, db = context.cfg, context.db
        parts: list = []

        # 1. What he already agreed to, before anything new is offered.
        recolte = recolter(cfg)
        if recolte.retenues:
            parts.append(
                f"{len(recolte.retenues)} proposal(s) the user had ticked are "
                f"now in their profile: "
                + "; ".join(f'"{t}"' for t in recolte.retenues)
            )
        if recolte.deja:
            parts.append(f"{recolte.deja} ticked line(s) were already known.")
        if recolte.masquees:
            parts.append(
                f"{recolte.masquees} ticked line(s) still carry a redaction "
                f"placeholder and were left in the file for the user to fix."
            )
        if recolte.hors_section:
            parts.append(
                f"{recolte.hors_section} ticked line(s) sit under a heading "
                f"this file does not recognise and were left alone."
            )
        if recolte.echouees:
            parts.append(
                f"{recolte.echouees} ticked line(s) could NOT be written. Say "
                f"so plainly and do not claim they were saved."
            )

        # 2. Then the journal.
        lecture = _propose.propositions(
            cfg, db, core=_core(cfg), deja=load_appris(cfg),
        )

        if not lecture.appelee:
            parts.append(
                "The journal could NOT be read this time. Say that plainly: "
                "do not say there was nothing new, because nothing was "
                "looked at."
            )
            return ToolExecutionResult(success=True, reply_text=" ".join(parts))

        # Only a reading that happened is recorded, or a failure would
        # skip those days for ever.
        try:
            db.marquer_journal_lu(lecture.lues)
        except Exception as e:
            debug_log(f"appris: window not recorded as read: {e}", "tools")

        if lecture.gardes:
            items = [
                (_VERS_LA_PAGE.get(c.genre, SECTION_PROFIL), c.date, c.texte, c.citation)
                for c in lecture.gardes
            ]
            if ajouter_propositions(cfg, items):
                parts.append(
                    f"{len(lecture.gardes)} new proposal(s) are waiting in the "
                    f"user's appris.md file: "
                    + "; ".join(f'"{c.texte}"' for c in lecture.gardes)
                    + ". Read them out and say that nothing is remembered "
                    "until they tick the box, and that they can rewrite the "
                    "wording first."
                )
            else:
                parts.append(
                    "New proposals were found but could NOT be written to the "
                    "file. Say so; do not read them out as if they were saved."
                )
        else:
            ecartes = (lecture.connus + lecture.refuses + lecture.infondes
                       + lecture.masques + lecture.mal_formes)
            if ecartes:
                # Loud on purpose. "Nothing found" and "four considered,
                # four set aside" are different facts about her reading.
                parts.append(
                    f"{lecture.bruts} thing(s) were considered and all were set "
                    f"aside: {lecture.connus} already known or previously "
                    f"struck out, {lecture.refuses} refused before, "
                    f"{lecture.infondes} not actually in the journal, "
                    f"{lecture.masques} carrying redacted text, "
                    f"{lecture.mal_formes} malformed."
                )
            else:
                parts.append("Nothing in the journal qualified this time.")

        if lecture.tronquee:
            parts.append(
                "The journal was longer than one reading; the oldest days were "
                "not looked at. Say so, and that asking again will cover them."
            )

        return ToolExecutionResult(success=True, reply_text=" ".join(parts))


def _core(cfg):
    from ...memory.core import MemoryCore

    return MemoryCore.for_config(cfg)
