"""
🎯 A routine that works for one of his goals.

A goal has no schedule and no envelope of its own: it is remembered, never
run. What runs is a routine, and its block may name one goal with an
`objectif:` line. This module is the two questions that line raises, and
nothing else: which goal a pass is handed, and which routines are really
at work for a goal.

Both lean shut. A goal that is closed or not in the file is no goal to
work for, and neither is a name that could not be one of that file's
headings. A block with no armed row is a routine that has stopped,
whatever the file still says.

See routines.spec.md ("A routine that works for a goal").
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..debug import debug_log
from ..objectifs.page import Objectif, open_goal
from .runner import payload_of, quand_fichier_pour
from .scope import load_routines

# A routine's heading as `setRoutine` writes one. The file is his, and a
# heading he typed can hold anything that is not a space: one that could
# not have been written here is not repeated into a prompt.
_NOM_DE_ROUTINE = re.compile(r"[\w-]{1,64}", re.UNICODE)


def goal_of(cfg, scope) -> Optional[Objectif]:
    """The open goal this routine's block names, or None.

    None when the block names no goal, and None when it names one that no
    open goal answers to: closed, gone from the file, a name that could
    not be a heading of it, or a file that cannot be read. The caller
    tells the first case from the others by `scope.objectif`, and
    suspends the pass on them.
    """
    nom = str(getattr(scope, "objectif", "") or "").strip()
    if not nom:
        return None
    try:
        return open_goal(cfg, nom)
    except Exception as e:
        debug_log(f"routine goal unreadable, none handed over: {e}", "tools")
        return None


def routines_at_work(cfg, db) -> Dict[str, List[Tuple[str, str]]]:
    """The armed routines, by the goal their block names: each one's name
    and the schedule it runs on.

    Armed means a pending row in `rappels`: a block alone is a routine
    that was stopped, and saying it looks after a goal would be a promise
    about every morning. The schedule is composed from the row's rule,
    the one it really runs on, never read off the block's `quand:` line,
    which is a copy he can edit. Sorted by name and named once each, so
    the text made of it does not move when one of them runs. Never
    raises, and with no table to read claims nothing.
    """
    par_objectif: Dict[str, Dict[str, str]] = {}
    if db is None:
        return {}
    try:
        blocs = load_routines(cfg)
        for row in db.pending_rappels(kind="routine"):
            payload = payload_of(row)
            bloc = blocs.get(str(payload.get("nom") or ""))
            if bloc is None or not bloc.objectif.strip():
                continue
            if not _NOM_DE_ROUTINE.fullmatch(bloc.nom):
                continue
            par_objectif.setdefault(bloc.objectif.strip(), {})[bloc.nom] = (
                quand_fichier_pour(payload))
    except Exception as e:
        debug_log(f"routines at work unreadable, none claimed: {e}", "tools")
        return {}
    return {objectif: sorted(routines.items())
            for objectif, routines in par_objectif.items()}
