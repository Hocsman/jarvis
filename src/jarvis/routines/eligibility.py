"""
🚪 The gate's arithmetic, run before anything is written.

The gate at the tool funnel already decides, every morning, whether a
routine may reach a given tool. This runs the same checks at
creation time, for one reason: an envelope full of names that will be
refused at 07:00 is a routine that half-works from its first day, and
the only trace is a journal line the user reads a week later.

It is not a second gate. The gate is still the authority — the world
moves between July and October, which is the whole reason it re-checks.
This only keeps the file honest at the moment it is written.

Which is also why the rejected-tools comment is split. `render_block`
invites the user to put a name back by adding a line. That invitation is
true only for names the gate would actually let through; for the rest,
adding the line changes nothing, ever, and the comment would be lying to
someone reading it a month later with no way to test it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..debug import debug_log
from .scope import JAMAIS_EN_ROUTINE

# Tightest reason first, because a name usually fails
# several and the user should be told the one they can act on.
POURQUOI_JAMAIS = "aucune routine ne peut l'atteindre"
POURQUOI_POLITIQUE = "tu as demandé à être consulté pour cet outil"
POURQUOI_ECRIT = "il ne se contente pas de lire"
POURQUOI_MEMOIRE = "il écrit la mémoire de Yuba, et personne n'est là pour relire"
POURQUOI_HUMAIN = "il attend que quelqu'un fasse un geste, et personne n'est là"


def refuse_reason(cfg, name: str) -> Optional[str]:
    """Why this tool could never run inside a routine, or None.

    Mirrors `registry._out_of_scope` minus the envelope check, which is
    the one thing that is not yet decided when this runs.
    """
    from ..tools.policy import FREE, RISK_READ, resolve_risk
    from ..tools.registry import _known_tool, load_tool_policy

    if name in JAMAIS_EN_ROUTINE:
        return POURQUOI_JAMAIS

    tool = _known_tool(name)
    # `resolve_risk` reads a call's arguments, and there are none yet. The
    # empty-argument reading is the optimistic one — `localFiles` reads
    # today and deletes tomorrow under one name — which is exactly why the
    # gate runs again every morning with the real arguments.
    risk = resolve_risk(name, tool, {})
    if load_tool_policy(cfg).verdict(name, risk) != FREE:
        return POURQUOI_POLITIQUE
    if risk != RISK_READ:
        return POURQUOI_ECRIT
    if getattr(tool, "writes_own_state", False):
        return POURQUOI_MEMOIRE
    if getattr(tool, "needs_a_human", False):
        return POURQUOI_HUMAIN
    return None


def candidate_tools(cfg) -> List[str]:
    """Every installed tool a routine could legitimately be given.

    Read off the machine rather than hard-coded: the user's own
    `outils.md` decides half of it, and their MCP servers decide the
    rest.
    """
    from ..tools.registry import BUILTIN_TOOLS, get_cached_mcp_tools

    try:
        mcp = get_cached_mcp_tools() or {}
    except Exception as e:
        debug_log(f"routine candidates: MCP catalogue unavailable ({e})", "tools")
        mcp = {}

    names = sorted(set(BUILTIN_TOOLS) | set(mcp))
    return [n for n in names if refuse_reason(cfg, n) is None]


def split_rejected(cfg, proposed: List[str], candidates: List[str],
                   chosen: List[str]) -> Tuple[List[Tuple[str, str]],
                                               List[Tuple[str, str]]]:
    """The two halves of the rejected-tools comment.

    First: names the user could genuinely put back by adding a line.
    Second: names no line will ever admit, so the comment must not invite
    it. A file that tells someone to try something that cannot work is
    worse than one that says nothing, because they only find out at 07:00
    and only if they read the journal.
    """
    ajoutables = [
        (n, "disponible, mais pas demandé pour cette routine")
        for n in candidates if n not in chosen
    ]
    impossibles = []
    for n in proposed:
        if n in chosen:
            continue
        pourquoi = refuse_reason(cfg, n)
        if pourquoi is not None:
            impossibles.append((n, pourquoi))
    return ajoutables, impossibles
