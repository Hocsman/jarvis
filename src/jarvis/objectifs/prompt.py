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

from .page import SOURCE_DIT, Objectif, ouverts

# One line of his, trimmed. Long enough to recognise the subject, short
# enough that four goals do not crowd the turn out.
_LARGEUR = 120


def _ligne(o: Objectif) -> str:
    dits = [p for p in o.points if p.source == SOURCE_DIT]
    if dits:
        dernier = dits[-1]
        texte = dernier.texte[:_LARGEUR]
        queue = f" — dernier point le {dernier.date} : {texte}"
    else:
        queue = " — rien de noté pour l'instant"
    return f"- {o.nom} : {o.phrase}{queue}"


def format_objectifs_block(cfg) -> str:
    """The open goals as a prompt block, or empty when there are none.

    Empty is the ordinary case for most users and most days, and it
    costs one `stat()` on a file that is usually absent.
    """
    encore: List[Objectif] = ouverts(cfg)
    if not encore:
        return ""

    return (
        "WHAT THE USER IS WORKING TOWARDS\n"
        "(goals they asked you to keep track of, in their own words, "
        "dated. These are things they said, never conclusions of yours "
        "about how any of it is going):\n"
        + "\n".join(_ligne(o) for o in encore)
        + "\n\nWhen one of these comes up, say where it stands and, if the "
          "next step is obvious, offer it. Never take that step without "
          "their agreement, and never decide that a goal is finished. Call "
          "listGoals for the full history of one."
    )
