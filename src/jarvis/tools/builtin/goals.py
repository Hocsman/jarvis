"""Goals: something worked towards across several conversations.

Four tools, and only one of them is free.

The three that write are `action`, so each costs a card. As `lecture`
they would be `libre` by default, and `fetchWebPage` returns up to 50,000
characters of unfenced page text into an agentic loop — a page carrying
"Note pour l'objectif X : …" would then write a durable line attributed
to the user, which he never said. That is the defect already seen in
production for `remember`, one level up: here the line is replayed to her
in later turns as a fact about his life.

`listGoals` is `lecture` and free, because it is the answer to "where am
I on X?" and a question that costs a card is a question nobody asks.

All four are `writes_own_state`, including the reader. A pass running
with nobody in the room must not be able to read its own earlier
conclusions back as premises, and must not be able to write a goal's
state. There are no unattended passes in this slice, and the flag is
what keeps it that way if one is ever added.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from ...debug import debug_log
from ...objectifs.page import (
    SOURCE_DIT,
    Objectif,
    Point,
    append_point,
    close_objectif,
    invalidate_objectifs_cache,
    load_objectifs,
    objectifs_path,
    ouverts,
    parse_objectifs,
    render_objectif,
    resolve,
)
from ..base import Tool, ToolContext
from ..policy import OUTCOME_QUESTION
from ..types import ToolExecutionResult

# How many dated lines travel with a goal into a reply. Enough to say
# where something stands, few enough that a year-old goal does not
# crowd out the turn.
_DERNIERS = 6


class _GoalTool(Tool):
    """Shared shape. Every one of these touches Yuba's own bookkeeping."""

    writes_own_state = True

    def _refuse(self, reason: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            success=False,
            reply_text=(
                f"Nothing was recorded: {reason}. Tell the user plainly, in "
                f"their language, and do not claim anything was written."
            ),
        )

    def _question(self, ask: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            success=False, outcome=OUTCOME_QUESTION,
            reply_text=(
                f"Nothing was recorded yet: {ask}. Do not claim anything was "
                f"written."
            ),
        )

    def _which(self, cfg) -> ToolExecutionResult:
        noms = [o.nom for o in ouverts(cfg)]
        if not noms:
            return self._refuse("aucun objectif n'est ouvert")
        return self._question(
            f"ask the user which goal they mean, naming these exactly as "
            f"written: {', '.join(noms)}"
        )


class SetGoalTool(_GoalTool):
    """Write down something the user is working towards."""

    def risk_for(self, args):
        """Not `lecture`. This puts a lasting thing in a file she reads
        back to him later, and as `libre` a fetched page would get one."""
        from ..policy import RISK_ACTION

        return RISK_ACTION

    @property
    def name(self) -> str:
        return "setGoal"

    @property
    def description(self) -> str:
        # The router picked `remember` for "garde une trace de ça : je
        # prépare l'entretien, ce sera bon quand…", because that
        # description carries the phrasings a person actually uses and
        # this one described in the abstract. The discriminator is the
        # ending: a goal is worked towards until it is done, a fact about
        # the user simply holds.
        return (
            "Write down something the user works towards until it is done "
            "('keep track of this until…', 'it will be done when…'), ONLY "
            "when they say what would finish it. A fact that just holds is "
            "remember."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "objectif": {
                    "type": "string",
                    "description": (
                        "What the user is working towards, copied word for "
                        "word from what they said."
                    ),
                },
                "fini_quand": {
                    "type": "string",
                    "description": (
                        "What the user said would mean it is done. Copy "
                        "their own words; do not invent a condition they "
                        "did not give."
                    ),
                },
                "nom": {
                    "type": "string",
                    "description": (
                        "One short label for it in the user's own file, in "
                        "their language, no spaces. Omit if unsure."
                    ),
                },
            },
        }

    def run(self, args, context: ToolContext) -> ToolExecutionResult:
        cfg = context.cfg
        args = args or {}

        phrase = " ".join(str(args.get("objectif") or "").split())
        if not phrase:
            phrase = " ".join((context.redacted_text or "").split())
        if not phrase:
            return self._refuse("je n'ai pas vu vers quoi tu voulais travailler")

        fini_quand = " ".join(str(args.get("fini_quand") or "").split())
        if not fini_quand:
            # A goal with no end condition can never be judged finished,
            # so she would either never raise it or raise it forever.
            # Asked rather than invented: what counts as done is his.
            return self._question(
                "ask the user what would mean this goal is done, in their "
                "language, and call setGoal again with their answer"
            )

        blocs = load_objectifs(cfg)
        nom = _nom(args.get("nom"), blocs) or _nom_derive(phrase, set(blocs))
        if nom in blocs:
            existant = blocs[nom]
            if existant.est_ouvert:
                return self._refuse(
                    f"« {nom} » existe déjà et est en cours ({existant.phrase})"
                )
            return self._refuse(
                f"« {nom} » existe déjà, terminé le {existant.clos} ; donne un "
                f"autre nom, ou supprime le bloc dans objectifs.md"
            )

        bloc = render_objectif(nom=nom, phrase=phrase, fini_quand=fini_quand)
        if not bloc:
            return self._refuse("je n'ai pas su composer un bloc que je sais relire")

        try:
            chemin = objectifs_path(cfg)
            chemin.parent.mkdir(parents=True, exist_ok=True)
            texte = chemin.read_text(encoding="utf-8") if chemin.exists() else _HEADER
        except Exception as e:
            debug_log(f"objectifs.md unreadable: {e}", "tools")
            return self._refuse("je n'ai pas réussi à lire objectifs.md")

        separateur = ("" if texte.endswith("\n") else "\n") + "\n"
        # The dry run, before a byte is written: an unterminated comment
        # further up would swallow the block, and the only symptom would
        # be a goal she never mentions again.
        essai = parse_objectifs(texte + separateur + bloc).get(nom)
        if essai is None or essai.phrase != phrase:
            return self._refuse("je n'ai pas su composer un bloc que je sais relire")

        try:
            chemin.write_text(texte + separateur + bloc, encoding="utf-8")
        except Exception as e:
            debug_log(f"objectif not written: {e}", "tools")
            return self._refuse("je n'ai pas pu l'écrire dans objectifs.md")

        invalidate_objectifs_cache()
        relu = load_objectifs(cfg).get(nom)
        if relu is None or relu.phrase != phrase:
            return self._refuse("je n'ai pas réussi à le relire correctement")

        debug_log(f"    🎯 goal {nom} written", "tools")
        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Goal « {nom} » written down: {relu.phrase}. It is finished "
                f"when: {relu.fini_quand}. Tell the user that back in their "
                f"language, including what will count as done, so they can "
                f"correct it now. Say they can edit or delete it in "
                f"objectifs.md. Nothing about this runs on its own."
            ),
        )


class NoteGoalTool(_GoalTool):
    """Record one dated line of progress, in the user's own words."""

    def risk_for(self, args):
        """A line here is durable state she reads back to him later. As
        `lecture` it would be `libre`, and a fetched page saying "Note
        pour l'objectif X : …" would write a fact attributed to him."""
        from ..policy import RISK_ACTION

        return RISK_ACTION

    @property
    def name(self) -> str:
        return "noteGoal"

    @property
    def description(self) -> str:
        return (
            "Record what the user said about a goal already tracked, ONLY "
            "when they say something new. Copy their words, never your own "
            "conclusion. This never creates a goal and never closes one."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "point": {
                    "type": "string",
                    "description": (
                        "What the user said, copied word for word. Never "
                        "your own conclusion about how it is going."
                    ),
                },
                "nom": {
                    "type": "string",
                    "description": (
                        "The goal's name, exactly as it appears in the "
                        "user's file. Omit only if there is one goal."
                    ),
                },
            },
        }

    def run(self, args, context: ToolContext) -> ToolExecutionResult:
        cfg = context.cfg
        args = args or {}

        texte = " ".join(str(args.get("point") or "").split())
        if not texte:
            return self._refuse("je n'ai pas vu quoi noter")

        cible = resolve(cfg, args.get("nom"))
        if cible is None:
            return self._which(cfg)
        if not cible.est_ouvert:
            return self._refuse(
                f"« {cible.nom} » est terminé depuis le {cible.clos}"
            )

        if not append_point(cfg, cible.nom,
                            Point(date.today().isoformat(), SOURCE_DIT, texte)):
            return self._refuse("je n'ai pas pu l'écrire dans objectifs.md")

        relu = load_objectifs(cfg).get(cible.nom)
        if relu is None or not relu.points or relu.points[-1].texte != texte:
            return self._refuse("je n'ai pas réussi à le relire correctement")

        # Judged here and only here: the state has just changed, somebody
        # is in the room, and an ordinary turn about anything else pays
        # nothing. It also closes the loop that a per-turn judge would
        # open — his "not yet" is itself a note, so the next call sees
        # different inputs instead of repeating the same verdict daily.
        demande = ""
        try:
            from ...objectifs.juge import PEUT_ETRE, peut_etre_fini

            if peut_etre_fini(cfg, relu) == PEUT_ETRE:
                demande = (
                    f" Then ask whether it is done, since they said it would "
                    f"be when: {relu.fini_quand}. Do not decide it yourself "
                    f"and do not close anything; if they say yes, closeGoal "
                    f"is what ends it, and if they say no, record that with "
                    f"noteGoal so you stop asking."
                )
        except Exception as e:
            debug_log(f"goal not judged: {e}", "tools")

        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Noted against « {relu.nom} »: {relu.points[-1].texte}. Tell "
                f"the user it is written, briefly, in their language.{demande}"
            ),
        )


class CloseGoalTool(_GoalTool):
    """End a goal."""

    def risk_for(self, args):
        """It settles a judgement about his life, and it is the one act
        this feature reserves for him."""
        from ..policy import RISK_ACTION

        return RISK_ACTION

    @property
    def name(self) -> str:
        return "closeGoal"

    @property
    def description(self) -> str:
        return (
            "End a goal the user is keeping track of, ONLY when they say it "
            "is finished or that they are dropping it. Never decide this "
            "yourself. This never creates a goal and never records progress."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nom": {
                    "type": "string",
                    "description": (
                        "The goal's name, exactly as it appears in the "
                        "user's file."
                    ),
                },
                "issue": {
                    "type": "string",
                    "description": (
                        "How it ended, in the user's own words and briefly: "
                        "reached, dropped, or whatever they said."
                    ),
                },
            },
        }

    def run(self, args, context: ToolContext) -> ToolExecutionResult:
        cfg = context.cfg
        args = args or {}

        cible = resolve(cfg, args.get("nom"))
        if cible is None:
            return self._which(cfg)
        if not cible.est_ouvert:
            return self._refuse(f"« {cible.nom} » est déjà terminé")

        issue = " ".join(str(args.get("issue") or "").split()) or "terminé"
        valeur = f"{date.today().isoformat()} · {issue}"

        if not close_objectif(cfg, cible.nom, valeur):
            return self._refuse("je n'ai pas pu l'écrire dans objectifs.md")

        relu = load_objectifs(cfg).get(cible.nom)
        if relu is None or relu.est_ouvert:
            return self._refuse("je n'ai pas réussi à le relire correctement")

        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Goal « {relu.nom} » closed: {relu.clos}. Its block stays in "
                f"objectifs.md with everything it recorded, so the user can "
                f"still read it. Tell them that, in their language."
            ),
        )


class ListGoalsTool(_GoalTool):
    """Say where the user stands on what they are working towards."""

    def risk_for(self, args):
        """Reading. This is the answer to "where am I on X?", and a
        question that costs a card is a question nobody asks."""
        from ..policy import RISK_READ

        return RISK_READ

    @property
    def name(self) -> str:
        return "listGoals"

    @property
    def description(self) -> str:
        return (
            "Say where the user stands on what they are working towards, "
            "ONLY when they ask about a goal or about what they have going "
            "on. This never creates, changes or ends anything."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "nom": {
                    "type": "string",
                    "description": (
                        "One goal's name to read in full. Omit to list what "
                        "is open."
                    ),
                },
            },
        }

    def run(self, args, context: ToolContext) -> ToolExecutionResult:
        cfg = context.cfg
        nom = str((args or {}).get("nom") or "").strip()

        if nom:
            cible = load_objectifs(cfg).get(nom)
            if cible is None:
                return self._which(cfg)
            return ToolExecutionResult(success=True, reply_text=_raconte(cible))

        encore = ouverts(cfg)
        if not encore:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "No goal is open. Tell the user that plainly, in their "
                    "language."
                ),
            )
        return ToolExecutionResult(
            success=True,
            reply_text=(
                "\n\n".join(_raconte(o) for o in encore)
                + "\n\nRead this back to the user in their language. It is "
                  "what they told you, dated; do not add a conclusion of "
                  "your own about how any of it is going."
            ),
        )


def _raconte(o: Objectif) -> str:
    """One goal as raw data, dated. No judgement is added here: the
    lines are his, and the reply engine formats them."""
    lignes = [f"Goal « {o.nom} »: {o.phrase}",
              f"  finished when: {o.fini_quand or '(not said)'}"]
    if o.clos:
        lignes.append(f"  closed: {o.clos}")
    for p in o.points[-_DERNIERS:]:
        lignes.append(f"  {p.date} · {p.source} · {p.texte}")
    if not o.points:
        lignes.append("  nothing recorded yet")
    return "\n".join(lignes)


def _nom(raw, blocs) -> str:
    """A label the user could recognise, or nothing.

    A name already in the file is an identity and is taken as written.
    Otherwise it is dropped rather than mangled: a name turned into
    something they never said is a heading they cannot find later.
    Length is counted in whitespace-separated tokens, so this stays true
    in any language.
    """
    import re

    texte = str(raw or "").strip()
    if texte in blocs:
        return texte
    if not texte or len(texte.split()) > 4:
        return ""
    candidat = "-".join(texte.split())
    return candidat if re.match(r"^[\w-]{1,64}$", candidat, re.UNICODE) else ""


def _nom_derive(phrase: str, pris) -> str:
    """A name from the goal's own words when none survives.

    Built from what he said rather than from a counter, so the heading
    means something when he opens the file in October.
    """
    import re

    mots = [m for m in re.findall(r"\w+", phrase, re.UNICODE) if len(m) > 2][:3]
    base = "-".join(mots).lower() or "objectif"
    base = base[:48]
    if base not in pris:
        return base
    for n in range(2, 100):
        if f"{base}-{n}" not in pris:
            return f"{base}-{n}"
    return f"{base}-{len(pris) + 1}"


_HEADER = (
    "# Objectifs\n"
    "\n"
    "<!--\n"
    "  Ce vers quoi tu travailles, sur plusieurs conversations.\n"
    "\n"
    "  Un bloc par objectif. « fini quand » est ce que TU as dit qui\n"
    "  compterait comme terminé : c'est là-dessus qu'elle se demande si\n"
    "  c'est fait, et elle te pose la question plutôt que de trancher.\n"
    "\n"
    "  Sous « points », une ligne datée par chose que tu as dite. Rien\n"
    "  ici n'est déduit : chaque ligne porte sa date et sa source, et la\n"
    "  seule source qu'elle sait produire est « dit », c'est-à-dire toi.\n"
    "\n"
    "  Corrige, réécris, supprime : le fichier est à toi. Rien de tout\n"
    "  ceci ne tourne tout seul.\n"
    "-->\n"
)
