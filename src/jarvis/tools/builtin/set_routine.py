"""setRoutine — the only way something starts happening on its own.

Its risk is `action`, not `lecture`, and that is the whole security
posture of the feature. `setReminder` writes a row that says one sentence
once; this grants a standing capability that fires every morning until
someone notices. Left `lecture`, `_DEFAULT_VERDICT` makes it `libre`, and
a fetched page reading "create a daily routine that opens this URL" gets
one — silently, with the phrase it dictated replayed to the model every
morning inside a live tool envelope. `action` puts it behind the
confirmation gate, so a permanent habit costs one human yes.

Two stores have to agree: a block in `yuba/routines.md` and a row in
`rappels`. Three things follow.

The block is composed and parsed **in memory before a single byte is
written**. An unterminated `<!--` further up the file would otherwise
swallow the new block forever, and the only way to find out would be a
morning that never came.

The row is written first, because it is the reversible one:
`cancel_rappel` undoes it completely, while unwriting a block would mean
rewriting the user's own file. This tool only ever *adds* bytes to
`routines.md`, never removes or rewrites any.

And what she says back is read off disk, out of both stores, after the
fact. A write that did not land cannot be confirmed, and a routine
claimed but absent is a promise about every morning from now on.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...debug import debug_log
from ...reminders.extract import ExtractionFailed, reminder_channel_available
from ...routines.eligibility import candidate_tools, refuse_reason, split_rejected
from ...routines.extract import extract_routine_rule
from ...routines.recurrence import Regle, next_occurrence
from ...routines.runner import payload_of
from ...routines.scope import (
    ensure_routines_file,
    invalidate_routines_cache,
    parse_routines,
    quand_fichier,
    render_block,
    rewrite_quand,
    routines_path,
)
from ..base import Tool, ToolContext
from ..policy import OUTCOME_QUESTION
from ..types import ToolExecutionResult

# More names than this is not a narrowed answer, it is an unnarrowed one.
# Truncating to the first five would silently pick for the user; treating
# it as no answer sends the turn to the menu, where they choose.
_ENVELOPE_MAX = 5

# A label, not a sentence. Anything a heading could not survive is
# dropped rather than sanitised into something the user did not write.
_NOM_PROPRE = re.compile(r"^[\w-]{1,40}$", re.UNICODE)
_NOM_MAX_MOTS = 4

# Which of the two ways forward a name leads to.
_CREER, _RALLUMER = "créer", "rallumer"


class SetRoutineTool(Tool):
    """Set up something Yuba does on her own, at an hour the user picks."""

    # A routine can therefore never create a routine: `_out_of_scope`
    # check 4 refuses this outright when a scope is present. Nothing
    # running unattended can multiply what runs unattended.
    writes_own_state = True

    def risk_for(self, args):
        """Not `lecture`, unlike `setReminder`, and the difference is the
        point. That one arranges a sentence; this grants a standing
        capability that fires every morning with nobody watching. As
        `lecture` it would be `libre` by default, and a fetched page
        saying "create a daily routine that opens this URL" would get
        one without the user ever seeing a card."""
        from ..policy import RISK_ACTION

        return RISK_ACTION

    @property
    def name(self) -> str:
        return "setRoutine"

    @property
    def description(self) -> str:
        # Whole sentences, restriction first: the router reads only the
        # opening. This tool and `setReminder` share a prefix and the
        # entire temporal vocabulary, so the discrimination has to land
        # before the first full stop or it is invisible where it matters.
        return (
            "Set up a repeating routine Yuba runs on her own at a fixed "
            "hour, ONLY when the user asks for something to happen every "
            "day or every week. One thing to be said once is setReminder, "
            "a different tool. This never lists existing routines and "
            "never runs one now."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        # No "required": the validator rejects a missing required key
        # before dispatch, which would deny the fallback to the user's
        # own sentence. `routine` alone also makes the planner's
        # direct-exec path work, so a concrete key='value' step
        # dispatches without asking the resolver to invent a schedule.
        return {
            "type": "object",
            "properties": {
                "routine": {
                    "type": "string",
                    "description": (
                        "The user's request, copied word for word. Always "
                        "keep how often they said it should happen — "
                        "without that this tool cannot do anything at "
                        "all. Keep the time of day only if they actually "
                        "said one: if they did not, leave it out and do "
                        "not supply one, not even a likely one. An hour "
                        "you add here cannot be told apart from an hour "
                        "they said, and it becomes the moment this runs "
                        "every day from now on."
                    ),
                },
                "outils": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The exact tool names this routine needs, and no "
                        "others. It will be able to reach these and "
                        "nothing else, every time it runs, with nobody "
                        "watching. Name the fewest that do the job."
                    ),
                },
                "nom": {
                    "type": "string",
                    "description": (
                        "One short label for this routine in the user's "
                        "own file, in their language, no spaces. Omit if "
                        "unsure."
                    ),
                },
            },
        }

    # ── One creation ──────────────────────────────────────────────────

    def run(self, args: Optional[Dict[str, Any]],
            context: ToolContext) -> ToolExecutionResult:
        from ...utils.time_context import local_timezone_name, now_in, to_utc_iso

        cfg = context.cfg
        args = args or {}

        utterance = str(args.get("routine") or "").strip()
        if not utterance:
            utterance = (context.redacted_text or "").strip()
        if not utterance:
            return self._refuse("je n'ai pas vu quelle habitude tu voulais prendre")

        if not bool(getattr(cfg, "routines_enabled", True)):
            # Without the dispatcher the row is written for nothing, and
            # she would confirm a routine that can never fire.
            return self._refuse("les routines sont désactivées dans tes réglages")

        if not reminder_channel_available(cfg):
            # Before the call, so "no model" and "the model failed" stay
            # different apologies.
            return self._refuse(
                "je n'ai aucun modèle pour lire le rythme que tu m'as donné"
            )

        try:
            lecture = extract_routine_rule(cfg, utterance)
        except ExtractionFailed as e:
            debug_log(f"    🌅 routine not created: {e}", "tools")
            return self._refuse(str(e))

        # A newline breaks `_FIELD_RE`, and the block with it.
        phrase = " ".join(str(lecture.quoi).split())

        candidats = candidate_tools(cfg)
        if not candidats:
            return self._refuse(
                "rien de ce que tu as libéré ne peut tourner sans toi ; "
                "c'est outils.md qui décide"
            )

        proposition = _proposed(args.get("outils"))
        if len(proposition) > _ENVELOPE_MAX:
            proposition = []
        outils = [n for n in proposition if refuse_reason(cfg, n) is None]
        if not outils:
            return self._menu(candidats)

        ajoutables, impossibles = split_rejected(cfg, proposition, candidats, outils)

        try:
            ensure_routines_file(cfg)
            texte = routines_path(cfg).read_text(encoding="utf-8")
        except Exception as e:
            debug_log(f"routines.md unreadable: {e}", "tools")
            return self._refuse("je n'ai pas réussi à lire routines.md")

        blocs = parse_routines(texte)
        lignes = _live_rows(context.db)
        propose = str(args.get("nom") or "").strip()
        if propose and propose in blocs:
            # A heading already in the file needs no sanitising. `_nom`
            # exists to clean a name on its way *into* the file, and
            # putting an existing title through it makes this tool refuse
            # names it wrote itself — which is how a routine that was
            # stopped becomes unrestartable under its own name.
            nom = propose
        else:
            nom = _nom(propose) or _nom_derive(
                lecture.regle, set(blocs) | set(lignes),
            )

        if nom not in blocs:
            # The model names a routine differently every time it is
            # asked: one turn proposes `actusWebMatin`, the next
            # `news_summary` for the same sentence. Matching a restart on
            # the name therefore never fires by itself, and the user
            # accumulates duplicate blocks doing one job under several
            # names. There is no reliable automatic identity here — the
            # name is unstable, the phrase is a model's reading of a
            # transcription, and a recurrence is not unique — so when a
            # stopped routine already runs at this very hour, it asks.
            attendu = quand_fichier(lecture.regle)
            arretes = [
                n for n, b in blocs.items()
                if n not in lignes and b.quand.strip() == attendu
            ]
            if len(arretes) == 1:
                # One stopped routine, already set for this very hour.
                # Saying the sentence again means that one, and the user
                # was told twice that it would. Safe to take without
                # asking because the name is not the guard: `_carte_dit_
                # vrai` still refuses to arm an envelope wider than the
                # one the confirmation card showed.
                nom = arretes[0]
            elif arretes:
                return self._arretes(arretes, blocs)

        verdict = self._collision(nom, blocs, lignes, phrase, outils)
        if isinstance(verdict, ToolExecutionResult):
            return verdict
        rallumer = verdict == _RALLUMER

        if rallumer:
            bloc_repris = blocs[nom]
            carte = self._carte_dit_vrai(cfg, nom, bloc_repris, outils)
            if carte is not None:
                return carte
            # The block wins: it is the user's corrected version, and the
            # phrase is what actually runs.
            phrase = bloc_repris.phrase.strip() or phrase
            outils = list(bloc_repris.outils)
            lecture = _heure_reprise(context.db, nom, lecture)
            if isinstance(lecture, ToolExecutionResult):
                return lecture

        separateur, bloc = "", ""
        if not rallumer:
            separateur = ("" if texte.endswith("\n") else "\n") + "\n"
            bloc = render_block(
                nom=nom, phrase=phrase, quand=quand_fichier(lecture.regle),
                outils=outils, ecartes=ajoutables, impossibles=impossibles,
            )

            # The dry run. Zero bytes written, nothing to unwind: this is
            # what catches an unterminated `<!--` the user left further
            # up, which would otherwise swallow the block and be found by
            # a morning that never came.
            essai = parse_routines(texte + separateur + bloc).get(nom)
            if essai is None or essai.outils != outils or essai.phrase != phrase:
                return self._refuse(
                    "je n'ai pas su composer un bloc que je sais relire"
                )

        tz = local_timezone_name()
        try:
            premiere = next_occurrence(lecture.regle, now_in(tz), tz)
            due_utc = to_utc_iso(premiere, tz)
        except Exception as e:
            return self._refuse(f"je n'ai pas su placer ce moment : {e}")

        # The row first: it is the reversible store. `cancel_rappel`
        # undoes it completely, while unwriting a block would mean
        # rewriting a file that belongs to the user.
        try:
            rid = context.db.add_rappel(
                texte=phrase, due_utc=due_utc, due_local=premiere.isoformat(),
                tz=tz, origin=context.origin, query=utterance, kind="routine",
                payload={
                    "nom": nom, "regle": lecture.regle.to_json(), "steriles": 0,
                },
            )
        except Exception as e:
            debug_log(f"routine row not written: {e}", "tools")
            return self._refuse("je n'ai pas pu inscrire son horaire")

        if not rallumer:
            try:
                with routines_path(cfg).open("a", encoding="utf-8") as fh:
                    fh.write(separateur + bloc)
            except Exception as e:
                debug_log(f"routine block not written: {e}", "tools")
                _withdraw(context.db, rid)
                return self._refuse("je n'ai pas pu écrire son bloc dans routines.md")

        if rallumer:
            # The one place this tool changes an existing byte, and only
            # when the line would otherwise name an hour nothing fires at.
            attendu = quand_fichier(lecture.regle)
            if blocs[nom].quand.strip() != attendu:
                rewrite_quand(cfg, nom, attendu)

        # Two writes inside one filesystem tick are indistinguishable to
        # an mtime cache, so the read-back below would see the file as it
        # was a moment before the block existed.
        try:
            invalidate_routines_cache()
        except Exception as e:
            debug_log(f"routines cache not invalidated: {e}", "tools")

        return self._read_back(context, cfg, rid, nom, phrase, outils, lecture)

    # ── Saying it back from disk ──────────────────────────────────────

    def _read_back(self, context, cfg, rid, nom, phrase, outils,
                   lecture) -> ToolExecutionResult:
        """Both stores, re-read and re-parsed, before she claims anything."""
        try:
            relu = parse_routines(
                routines_path(cfg).read_text(encoding="utf-8")
            ).get(nom)
            ligne = _live_rows(context.db).get(nom)
            quand = datetime.fromisoformat(ligne["due_local"]) if ligne else None
            regle = Regle.from_json(payload_of(ligne).get("regle")) if ligne else None
        except Exception as e:
            debug_log(f"routine read-back failed: {e}", "tools")
            relu, ligne, quand, regle = None, None, None, None

        if (relu is None or ligne is None or quand is None or regle is None
                or relu.outils != outils or relu.phrase != phrase
                or ligne.get("id") != rid):
            _withdraw(context.db, rid)
            return self._refuse_moitie(nom, relu)

        try:
            from ...daemon import nudge_routines

            nudge_routines()
        except Exception as e:
            debug_log(f"routine dispatcher not nudged: {e}", "tools")

        debug_log(f"    🌅 routine {nom} set for {ligne['due_utc']}", "tools")
        return ToolExecutionResult(
            success=True,
            reply_text=_confirmation(nom, regle, quand, relu.outils,
                                     lecture.heure_supposee),
        )

    # ── The ways it does not happen ───────────────────────────────────

    def _collision(self, nom, blocs, lignes, phrase, outils):
        """A name already spoken for, or the same routine said twice.

        Returns a refusal, a question, or which way forward this is.

        A block with no live row is not a collision: it is that routine,
        stopped. Both the tab's button and the dispatcher's auto-stop
        leave exactly that state and both invite the user to say the
        sentence again, so refusing there would make stopping a one-way
        door. Matched on the **name** alone — the phrase is a model's
        reading of a Whisper transcription and the block may have been
        corrected by hand, so two utterances of one routine never agree
        byte for byte.
        """
        bloc = blocs.get(nom)
        vivante = lignes.get(nom)

        if bloc is not None and vivante is None:
            return _RALLUMER

        if vivante is not None:
            if str(vivante.get("texte") or "") == phrase:
                return self._inchangee(nom, vivante)
            return self._refuse(
                f"« {nom} » est déjà pris par une autre routine ; donne-lui "
                f"un autre nom, ou arrête celle-là dans l'onglet Routines"
            )
        return _CREER

    def _carte_dit_vrai(self, cfg, nom, bloc, outils) -> Optional[ToolExecutionResult]:
        """Whether the yes already given describes what re-arming grants.

        The whole security posture of this tool is one sentence: a
        standing habit costs a human yes. That yes is given on a card
        `describe_action` composes **only from the call's arguments**, and
        the digest is recomputed at execution so an approval cannot slide
        from one call to another. Re-arming from the block would break
        that equivalence for the first time: the envelope actually armed
        is the block's, and the card showed the model's proposal.

        Concretely: a fetched page tells the model to set up a routine
        named like one the user stopped. The card shows one harmless
        tool. The block carries three and a `mémoire: oui` line, so the
        user's whole profile leaves the machine every morning — none of
        it on the card they approved. Revoking is free (`cancelRoutine`
        is `libre`), restoring costs a yes that describes something else.

        So when the block grants more than the call proposed, nothing is
        written and the question is asked again with the block's real
        envelope in the arguments, where the gate will fingerprint it.
        """
        scope = bloc.scope()
        if set(scope.outils) == set(outils) and not scope.memoire:
            return None

        profil = (
            " Elle emporte aussi ton profil et tes règles (« mémoire: oui »)."
            if scope.memoire else ""
        )
        return ToolExecutionResult(
            success=False, outcome=OUTCOME_QUESTION,
            reply_text=(
                f"Nothing was set up yet. The stopped routine « {nom} » "
                f"already exists in routines.md and restarting it would arm "
                f"what its own block says, not what was just proposed: "
                f"{', '.join(f'`{n}`' for n in scope.outils) or 'aucun outil'}."
                f"{profil} Tell the user exactly that, in their language, "
                f"listing those tool names unchanged, and ask whether to "
                f"restart it as written. If they agree, call setRoutine "
                f"again with outils set to exactly that list."
            ),
        )

    def _inchangee(self, nom, vivante) -> ToolExecutionResult:
        """It already does this. The user asked for a state, and the
        state holds — recording that as `échec` would be a lie about the
        tool in a tab the user reads."""
        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Nothing needed doing: « {nom} » already does exactly that, "
                f"next on {_lisible(vivante.get('due_local'))}. Tell the user "
                f"plainly, in their language."
            ),
        )

    def _arretes(self, noms: List[str], blocs) -> ToolExecutionResult:
        """A stopped routine already runs at this hour. Which is it?

        Nothing is written. The names quoted are headings already in the
        file, so calling back with one of them resolves without going
        through `_nom` — which is what stops this question repeating
        itself forever.
        """
        lignes = "; ".join(
            f"« {n} » ({blocs[n].phrase.strip() or 'sans phrase'})" for n in noms
        )
        return ToolExecutionResult(
            success=False, outcome=OUTCOME_QUESTION,
            reply_text=(
                f"Nothing was set up yet. These routines are stopped and "
                f"already set for that time: {lignes}. Ask the user, in "
                f"their language, whether to restart one of them or to add a "
                f"new one alongside. To restart one, call setRoutine again "
                f"with nom set to that name exactly as written here. Do not "
                f"claim anything was created."
            ),
        )

    def _menu(self, candidats: List[str]) -> ToolExecutionResult:
        """No usable envelope. Ask, naming what is actually reachable.

        Refusing without the list would send the model guessing again at
        names the gate has already ruled out.
        """
        return ToolExecutionResult(
            success=False,
            reply_text=(
                f"Nothing was set up: no usable tool list was given. Ask the "
                f"user which of these the routine should be allowed to "
                f"reach, listing them by name, in their language: "
                f"{', '.join(candidats)}. It will be able to reach those and "
                f"nothing else. Do not claim a routine was created."
            ),
        )

    def _question(self, ask: str) -> ToolExecutionResult:
        """Something only the user can supply is missing. Not a failure:
        nothing broke, and the ledger should not say it did."""
        return ToolExecutionResult(
            success=False, outcome=OUTCOME_QUESTION,
            reply_text=(
                f"Nothing was set up yet: {ask}. Do not claim a routine was "
                f"created, and do not tell them when it would have run."
            ),
        )

    def _refuse(self, reason: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            success=False,
            reply_text=(
                f"No routine was set up: {reason}. Tell the user plainly, in "
                f"their language, and ask again for whatever was missing. Do "
                f"not claim a routine was created, and do not tell them when "
                f"it would have run."
            ),
        )

    def _refuse_moitie(self, nom: str, relu) -> ToolExecutionResult:
        """The round trip failed. What is said depends on which half of
        it failed, because the same rule holds here as everywhere else in
        this feature: never claim anything about a store you have not
        just read."""
        if relu is None:
            reste = (
                f"no block named « {nom} » could be read back from "
                f"routines.md, so do not tell them to say the request again"
            )
        else:
            reste = (
                f"a block named « {nom} » is in routines.md and intact, so "
                f"they can read what it was allowed to do, or delete it"
            )
        return ToolExecutionResult(
            success=False,
            reply_text=(
                f"No routine was set up: it could not be read back, and "
                f"{reste}. Nothing is scheduled. Tell the user plainly, in "
                f"their language, and do not claim a routine was created."
            ),
        )


# ── Plain helpers ─────────────────────────────────────────────────────


def _proposed(raw) -> List[str]:
    """The model's tool list, coerced and de-duplicated, order kept."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _nom(raw) -> str:
    """A label the user could recognise, or nothing.

    Dropped rather than sanitised: a name mangled into something they
    never said is a heading they cannot find later. Length is counted in
    whitespace-separated tokens rather than matched against words, so
    this stays true in any language.
    """
    text = str(raw or "").strip()
    if not text or len(text.split()) > _NOM_MAX_MOTS:
        return ""
    candidate = "-".join(text.split())
    return candidate if _NOM_PROPRE.match(candidate) else ""


def _nom_derive(regle: Regle, pris) -> str:
    """A name from the rule itself when none survives.

    Unbreakable rather than pretty, and the collision suffix keeps it so.
    """
    base = "quotidien" if regle.kind == "daily" else "hebdo"
    base = f"{base}-{regle.hour:02d}h{regle.minute:02d}"
    if base not in pris:
        return base
    for n in range(2, 100):
        if f"{base}-{n}" not in pris:
            return f"{base}-{n}"
    return f"{base}-{len(pris) + 1}"


def _heure_reprise(db, nom: str, lecture):
    """The hour a re-armed routine runs at, when none was spoken.

    Taking `reminder_default_hour` here would silently move a 07:00 habit
    to 09:00 and then rewrite the file to agree — exactly the failure the
    disclosure exists to prevent. The routine's own cancelled row still
    records the recurrence it had, so that is where the hour comes from.

    Only the hour and the minute: a rhythm freshly spoken must survive,
    and rebuilding through `from_json` normalises a weekday left over
    from a rule that has become daily.
    """
    from ...routines.extract import Lecture

    if not lecture.heure_supposee:
        return lecture
    try:
        ancienne = db.last_cancelled_rappel("routine", nom)
        regle = Regle.from_json(payload_of(ancienne).get("regle")) if ancienne else None
    except Exception as e:
        debug_log(f"previous schedule unreadable: {e}", "tools")
        regle = None

    if regle is None:
        return ToolExecutionResult(
            success=False, outcome=OUTCOME_QUESTION,
            reply_text=(
                f"Nothing was set up yet: « {nom} » was stopped and no time "
                f"of day was given to restart it, and none is recorded. Ask "
                f"the user what time it should run, in their language. Do "
                f"not choose one for them."
            ),
        )
    reprise = Regle.from_json({
        **lecture.regle.to_json(), "hour": regle.hour, "minute": regle.minute,
    })
    return Lecture(regle=reprise, quoi=lecture.quoi, heure_supposee=False)


def _live_rows(db) -> Dict[str, dict]:
    """Pending routine rows, keyed by name.

    `pending_rappels`, never `all_rappels`: the latter is
    `ORDER BY due_utc DESC LIMIT 200`, so a routine due tomorrow sits at
    the bottom and two hundred further-off reminders would have this tool
    withdraw the routine it had just correctly created.
    """
    out: Dict[str, dict] = {}
    try:
        for row in db.pending_rappels(kind="routine"):
            nom = str(payload_of(row).get("nom") or "")
            if nom:
                out[nom] = row
    except Exception as e:
        debug_log(f"routine rows unreadable: {e}", "tools")
    return out


def _withdraw(db, rid: str) -> None:
    try:
        db.cancel_rappel(rid)
    except Exception as e:
        debug_log(f"routine row not withdrawn: {e}", "tools")


def _quand_directive(regle: Regle) -> str:
    """The same schedule for the model, in the directive's language.

    Composed twice from one rule rather than translated once, so the
    model renders a fact instead of restating a French string it may
    quietly turn into "every day".
    """
    heure = f"{regle.hour:02d}:{regle.minute:02d}"
    if regle.kind == "daily":
        return f"every day at {heure}"
    jours = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")
    return f"every {jours[regle.weekday % 7]} at {heure}"


def _lisible(due_local) -> str:
    try:
        return f"{datetime.fromisoformat(str(due_local)):%d/%m à %H:%M}"
    except Exception:
        return str(due_local or "")


def _confirmation(nom: str, regle: Regle, quand: datetime, outils: List[str],
                  heure_supposee: bool) -> str:
    """What she says back, written by code rather than by the model.

    The envelope is in it because that is the part the user cannot see
    otherwise and cannot easily undo later: a routine they hear the tools
    for is one they can narrow in the same breath.
    """
    supposition = (
        " They did not say a time of day, so it was chosen for them and "
        "they can change it." if heure_supposee else ""
    )
    # The tool names are quoted and the instruction to copy them is
    # explicit, because a model told to "include the tool names" writes
    # "using a web search" — prose that reads identically for an envelope
    # of one tool and an envelope of five. The names are the part the
    # user cannot see otherwise, and the only string they can search for
    # in routines.md.
    noms = ", ".join(f"`{n}`" for n in outils) or "`none`"
    return (
        f"Routine « {nom} » created, running {_quand_directive(regle)}, "
        f"first on {quand:%d/%m at %H:%M}. It can reach these tools and "
        f"nothing else: {noms}. Tell the user, in their language: the "
        f"schedule, and then every one of those tool names reproduced "
        f"exactly as written here, unchanged and untranslated — they are "
        f"what the user searches for in routines.md to narrow it. Say they "
        f"can narrow that list in routines.md. Do NOT tell them they can "
        f"change the schedule there: that line is a copy, and editing it "
        f"moves nothing. Changing when it runs is something they say to "
        f"her.{supposition}"
    )
