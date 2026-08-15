"""
🧠 The tick is the write, and nothing else is.

No model runs on this path. A regular expression reads a checkbox and
`MemoryCore.remember` writes a line. There is no sentence any model can
emit, no page a web tool can return, and no mis-transcription that puts a
character between two brackets — which is what makes the learning step
defensible at all.

Two orderings are load-bearing, and both are about failure.

The core is written first and the page marked second. If the mark fails
after a successful write, the next pass re-remembers and the core's own
duplicate scan makes that a no-op. Marked first, a failed core write
would delete the line from his page and report success — which is the
defect shape this project keeps finding.

And the loop stops at the first write that raises. Partial work is
reported as partial: a harvest that swallowed three failures and
announced four successes would be the same lie in a quieter voice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from ..debug import debug_log
from ..memory.core import (
    SECTION_PROFILE,
    SECTION_RULES,
    SOURCE_CONFIRMED,
    MemoryCore,
)
from ..utils.redact import contains_redaction_placeholder
from .page import (
    ETAT_ATTENTE,
    ETAT_COCHEE,
    SECTION_PROFIL,
    SECTION_REGLES,
    load_appris,
    marquer_retenue,
)

# The page's own section keys, and the core sections they name. Kept here
# rather than in `page.py` so the page stays a grammar and knows nothing
# about what a line eventually becomes.
_VERS_LE_NOYAU = {
    SECTION_PROFIL: SECTION_PROFILE,
    SECTION_REGLES: SECTION_RULES,
}


@dataclass(frozen=True)
class Recolte:
    """What one harvest did, and everything it declined to do."""

    retenues: List[str] = field(default_factory=list)
    deja: int = 0            # already believed; the tick is still consumed
    masquees: int = 0        # a redaction placeholder in the ticked line
    echouees: int = 0        # the core write raised
    hors_section: int = 0    # ticked under a heading this file does not know


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def recolter_une(cfg, ligne: str) -> bool:
    """Write one named proposal into the core, now.

    The tab's door onto the same act. A tick in the file waits for his
    next ask because nothing watches the file; a click here does not.
    Everything else is identical — the same guards, the same order (core
    first, page second), and the same refusal to store a line carrying a
    redaction placeholder.

    A struck proposal is never written, whatever was clicked: refusal is
    as durable through this door as through the other, or the tab becomes
    a way for a refused belief to come back.
    """
    cible = (ligne or "").rstrip("\n")
    if not cible:
        return False

    candidate = next(
        (p for p in load_appris(cfg)
         # Waiting or already ticked, never struck: refusal is as
         # durable through this door as through the other.
         if p.ligne.rstrip("\n") == cible
         and p.etat in (ETAT_ATTENTE, ETAT_COCHEE)),
        None,
    )
    if candidate is None:
        return False

    section = _VERS_LE_NOYAU.get(candidate.section or "")
    if section is None:
        debug_log("appris: accepted under an unknown heading, nothing written", "tools")
        return False
    if contains_redaction_placeholder(candidate.texte):
        debug_log("appris: accepted line carries a redaction placeholder", "tools")
        return False

    try:
        core = MemoryCore.for_config(cfg)
        core.remember(section, candidate.texte,
                      on_date=candidate.date, source=SOURCE_CONFIRMED)
    except (OSError, ValueError) as e:
        debug_log(f"appris: core write failed from the tab: {e}", "tools")
        return False

    marquer_retenue(cfg, candidate.ligne, _today_utc())
    return True


def recolter(cfg) -> Recolte:
    """Write every ticked proposal into the core, in file order.

    Never raises. What it could not do it counts, so the caller can say
    so out loud rather than reporting a quiet success.
    """
    try:
        core = MemoryCore.for_config(cfg)
    except Exception as e:
        debug_log(f"appris harvest skipped, no core: {e}", "tools")
        return Recolte()

    aujourdhui = _today_utc()
    retenues: List[str] = []
    deja = masquees = echouees = hors_section = 0

    # Lines already dealt with this pass, so a proposal left ticked —
    # because it was refused, or because the mark could not be written —
    # is not picked up again in the same loop.
    vus: set = set()

    while True:
        candidate = None
        for p in load_appris(cfg):
            if p.etat == ETAT_COCHEE and p.ligne not in vus:
                candidate = p
                break
        if candidate is None:
            break
        vus.add(candidate.ligne)

        section = _VERS_LE_NOYAU.get(candidate.section or "")
        if section is None:
            hors_section += 1
            debug_log(
                f"appris: ticked under an unknown heading, left alone: "
                f"{candidate.texte[:60]}",
                "tools",
            )
            continue

        if contains_redaction_placeholder(candidate.texte):
            masquees += 1
            debug_log(
                "appris: ticked line carries a redaction placeholder, "
                "left for him to fix",
                "tools",
            )
            continue

        try:
            resultat = core.remember(
                section, candidate.texte,
                on_date=candidate.date, source=SOURCE_CONFIRMED,
            )
        except (OSError, ValueError) as e:
            echouees += 1
            debug_log(f"appris: core write failed, harvest stopped: {e}", "tools")
            break

        if resultat.already_known:
            deja += 1
        else:
            retenues.append(candidate.texte)

        marquer_retenue(cfg, candidate.ligne, aujourdhui)

    return Recolte(
        retenues=retenues, deja=deja, masquees=masquees,
        echouees=echouees, hors_section=hors_section,
    )
