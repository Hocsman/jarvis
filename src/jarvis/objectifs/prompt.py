"""
🧭 What he is working towards, in the turn where he mentions it.

Thin on purpose. One line per open goal — its name, what it is, and the
last thing he said about it — because the point is that she recognises
the subject when it comes up, not that she recites a history nobody
asked for. The detail is one `listGoals` call away, and only on the
turns that want it.

Every line here is his. Only points whose source is `dit` are shown, so
that the day a pass can write a line of its own, that line does not
arrive in an attended prompt looking like something he said. That is the
same defence the diary block makes, one artefact along, and it is
mechanical rather than a matter of who remembers.

Nothing here is a conclusion. She is told so explicitly, because a list
of dated facts is exactly the shape a model summarises into "il avance
bien", and that sentence would be hers arriving as his.
"""

from __future__ import annotations

from typing import List

from .page import SOURCE_DIT, Objectif, load_objectifs, ouverts

# One line of his, trimmed. Long enough to recognise the subject, short
# enough that four goals do not crowd the turn out.
_LARGEUR = 120


def _ligne(o: Objectif, au_travail=()) -> str:
    dits = [p for p in o.points if p.source == SOURCE_DIT]
    if dits:
        dernier = dits[-1]
        texte = dernier.texte[:_LARGEUR]
        queue = f" — dernier point le {dernier.date} : {texte}"
    else:
        queue = " — rien de noté pour l'instant"
    # A fact read off both stores, composed here: the routine's name and
    # its schedule line, never a word of what a pass wrote. Its write-up
    # is prose a model made of pages it fetched, and it stays in the
    # journal for him to read.
    for bloc in au_travail:
        queue += (f" — la routine « {bloc.nom} » y travaille ({bloc.quand.strip()}), "
                  f"ses comptes rendus sont dans le journal")
    return f"- {o.nom} : {o.phrase}{queue}"


# Said only when a routine is at work for one of the goals shown. Without
# it she offers to read the write-ups out, and no tool of hers reaches the
# journal: it is prose a model made of pages it fetched, kept for him.
_SES_COMPTES_RENDUS = (
    " When a routine works for one, say so, and that its write-ups are in "
    "the journal for the user to read: you have not read them and cannot, "
    "so never say what they hold and never offer to read them."
)


def format_objectifs_block(cfg, db=None) -> str:
    """The open goals as a prompt block, or empty when there are none.

    Empty is the ordinary case for most users and most days, and it
    costs one `stat()` on a file that is usually absent.

    With *db*, a goal an armed routine works for says so, with where its
    write-ups are. Without it nothing of the kind is claimed.
    """
    encore: List[Objectif] = ouverts(cfg)
    if not encore:
        return ""

    au_travail = {}
    if db is not None:
        try:
            from ..routines.objectif import routines_at_work

            au_travail = routines_at_work(cfg, db)
        except Exception:
            au_travail = {}

    return (
        "WHAT THE USER IS WORKING TOWARDS\n"
        "(goals they asked you to keep track of, in their own words, "
        "dated. These are things they said, never conclusions of yours "
        "about how any of it is going):\n"
        + "\n".join(_ligne(o, au_travail.get(o.nom, ())) for o in encore)
        + "\n\nWhen one of these comes up, say where it stands and, if the "
          "next step is obvious, offer it. Never take that step without "
          "their agreement, and never decide that a goal is finished. Call "
          "listGoals for the full history of one."
        + (_SES_COMPTES_RENDUS if any(au_travail.get(o.nom) for o in encore) else "")
    )


def format_objectif_for_routine(cfg, nom: str) -> str:
    """The one goal a routine's pass works for, or empty.

    Empty when the goal is closed or not in the file: a pass is handed
    nothing rather than something stale. Only what he said is in it, as
    in the attended block, and no other goal.
    """
    objectif = load_objectifs(cfg).get((nom or "").strip())
    if objectif is None or not objectif.est_ouvert:
        return ""

    dits = [p for p in objectif.points if p.source == SOURCE_DIT]
    points = "\n".join(f"  - {p.date} : {p.texte}" for p in dits) or "  - nothing noted yet"
    return (
        "THE GOAL THIS ROUTINE WORKS FOR\n"
        "(one goal the user asked you to keep track of, in their own words, "
        "dated. These are things they said, never conclusions of yours "
        "about how any of it is going):\n"
        f"- {objectif.nom} : {objectif.phrase}\n"
        f"- what they said would count as done: {objectif.fini_quand}\n"
        f"- what they said about it so far:\n{points}\n\n"
        "Do what this routine asks with that goal in mind, and report what "
        "you found. You cannot write to the goal, and you never decide that "
        "it is finished: the user reads your report and says so themselves."
    )
