"""
🎯 A routine that works for one of his goals.

A goal has no schedule and no envelope of its own: it is remembered, never
run. What runs is a routine, and its block may name one goal with an
`objectif:` line. This module is the two questions that line raises, and
nothing else: which goal a pass is handed, and which routines are really
at work for a goal.

Both lean shut. A goal that is closed or no longer in the file is no goal
to work for, and a block with no armed row is a routine that has stopped,
whatever the file still says.

See routines.spec.md ("A routine that works for a goal").
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..debug import debug_log
from .scope import RoutineBlock, load_routines


def goal_of(cfg, scope) -> Optional["object"]:
    """The open goal this routine's block names, or None.

    None when the block names no goal, and None when it names one that is
    closed or absent: the caller tells the two apart by `scope.objectif`,
    and suspends the pass on the second.
    """
    nom = str(getattr(scope, "objectif", "") or "").strip()
    if not nom:
        return None
    try:
        from ..objectifs.page import load_objectifs

        objectif = load_objectifs(cfg).get(nom)
    except Exception as e:
        debug_log(f"routine goal unreadable, none handed over: {e}", "tools")
        return None
    if objectif is None or not objectif.est_ouvert:
        return None
    return objectif


def routines_at_work(cfg, db) -> Dict[str, List[RoutineBlock]]:
    """The armed routines, by the goal their block names.

    Armed means a pending row in `rappels`: a block alone is a routine
    that was stopped, and saying it looks after a goal would be a promise
    about every morning. Never raises, and with no table to read claims
    nothing.
    """
    par_objectif: Dict[str, List[RoutineBlock]] = {}
    if db is None:
        return par_objectif
    try:
        from .runner import payload_of

        blocs = load_routines(cfg)
        for row in db.pending_rappels(kind="routine"):
            bloc = blocs.get(str(payload_of(row).get("nom") or ""))
            if bloc is not None and bloc.objectif.strip():
                par_objectif.setdefault(bloc.objectif.strip(), []).append(bloc)
    except Exception as e:
        debug_log(f"routines at work unreadable, none claimed: {e}", "tools")
        return {}
    return par_objectif
