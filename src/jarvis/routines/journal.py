"""
📓 The morning after, in a file the user can read.

A routine runs while they are asleep. There is no speech, no chat
bubble, nobody to say "hm?" to. This is the whole delivery: if it is not
written here, the routine happened to nobody.

So it carries the write-up itself, not a note that something ran. That is
the one place this diverges from the action ledger, which records what
was *done* and never what was *seen*. The ledger is an audit trail; this
is the letter on the kitchen table. It never leaves the machine — same
directory as the user's own profile — so there is nothing to redact.

One file per day, appended. A crash costs one morning rather than the
year, and Markdown means the whole history opens in anything.

Nothing here raises. By the time it runs the work is already done, and
losing the letter is bad while losing the letter *and* taking the runner
down with it is worse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from ..debug import debug_log

JOURNAL_DIRNAME = "journal"

# A dated page, and only a dated page. The folder sits in the user's own
# directory and they may well drop their own notes in it; a sweep that
# eats those is a sweep that gets the feature turned off.
_PAGE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\.md$")


@dataclass
class Entree:
    """One run, as the user will read it."""

    nom: str
    moment: datetime
    demande: str
    texte: Optional[str] = None
    outils: List[str] = field(default_factory=list)
    ecartes: List[Tuple[str, str]] = field(default_factory=list)
    duree_sec: float = 0.0
    erreur: Optional[str] = None


def journal_dir(cfg) -> Path:
    from ..memory.core import MemoryCore

    return MemoryCore.for_config(cfg).directory / JOURNAL_DIRNAME


def journal_path(cfg, jour: datetime) -> Path:
    """One page per day, named so a directory listing sorts the
    mornings."""
    return journal_dir(cfg) / f"{jour.strftime('%Y-%m-%d')}.md"


def _render(entree: Entree) -> str:
    lines = [f"## {entree.moment.strftime('%H:%M')} — {entree.nom}", ""]
    if entree.demande:
        lines.append(f"> {entree.demande}")
        lines.append("")

    if entree.texte:
        lines.append(entree.texte.strip())
        lines.append("")
    elif entree.erreur:
        lines.append(f"Rien : {entree.erreur}")
        lines.append("")
    else:
        lines.append("Rien à signaler.")
        lines.append("")

    if entree.outils:
        lines.append(f"*Outils : {', '.join(entree.outils)}.*")
    if entree.ecartes:
        # Named with the reason, because "she did nothing" and "she was
        # stopped" are different facts and only one of them means the
        # envelope wants widening.
        for nom, pourquoi in entree.ecartes:
            lines.append(f"*Écarté : {nom} — {pourquoi}.*")
    if entree.duree_sec:
        lines.append(f"*{entree.duree_sec:.1f} s.*")

    lines.append("")
    return "\n".join(lines)


def append_run(cfg, entree: Entree) -> None:
    """Add one run to its day's page. Never raises."""
    try:
        path = journal_path(cfg, entree.moment)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "" if path.exists() else f"# {entree.moment.strftime('%Y-%m-%d')}\n\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(header + _render(entree))
        debug_log(f"    📓 {entree.nom} written to {path.name}", "tools")
    except Exception as e:
        debug_log(f"journal not written for {entree.nom}: {e}", "tools")


def read_day(cfg, jour: datetime) -> str:
    """The page for one day, or empty. Never raises."""
    try:
        path = journal_path(cfg, jour)
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception as e:
        debug_log(f"journal page unreadable: {e}", "tools")
        return ""


def prune_journal(cfg, max_age_days: int = 90) -> int:
    """Drop pages past the retention window. Returns how many went.

    Only pages whose name *is* a date, and only ones that parse as one:
    everything else in the folder belongs to the user.
    """
    removed = 0
    try:
        directory = journal_dir(cfg)
        if not directory.is_dir():
            return 0
        cutoff = (datetime.now() - timedelta(days=max_age_days)).date()
        for entry in directory.iterdir():
            match = _PAGE_RE.match(entry.name)
            if not match:
                continue
            try:
                jour = datetime.strptime(match.group("date"), "%Y-%m-%d").date()
            except ValueError:
                continue
            if jour < cutoff:
                entry.unlink()
                removed += 1
    except Exception as e:
        debug_log(f"journal prune skipped: {e}", "tools")
    return removed
