"""
🧠 The page of things she thinks she noticed.

The fourth artefact the user owns outright, beside the core, the tool
policy, the goals and the routines envelope, and it parses like them.
She writes candidates here; he resolves each with one character.

**Nothing in this file is a belief.** No prompt reads it, so a proposal
sitting here for six months changes nothing she says or does. It becomes
a belief the moment he ticks it and not one moment before: there is no
timeout that accepts, no sweeper that tidies, no "unless you object".
The quiet state is already the safe one, which is what lets her be wrong
here without it costing anything but a line to read.

The three states are the ones his other files already use. An untouched
box waits. A struck line is refused, and refusal is as durable as
acceptance — a proposal he struck out is never offered again, so nothing
wins by attrition. A ticked line is harvested *as it currently reads*,
not as she proposed it, so he can fix a clumsy sentence, or one that
arrived in the wrong language, before agreeing to be described by it.

See appris.spec.md for the full contract.
"""

from __future__ import annotations

import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..debug import debug_log

APPRIS_FILENAME = "appris.md"

# Where a proposal came from. One word, one origin: his journal. This is
# not a core source word — it says where the idea came from, not how it
# became a belief. That second word is `confirmé`, and only the harvest
# writes it.
SOURCE_JOURNAL = "journal"

SECTION_PROFIL = "profil"
SECTION_REGLES = "regles"

ETAT_ATTENTE = "attente"
ETAT_COCHEE = "cochée"
ETAT_RAYEE = "rayée"

# A proposal must fit on one line and inside the profile block's budget.
# `_collect` stops at the first oversized entry rather than skipping it,
# so one runaway line would truncate the tail of everything he actually
# said.
TEXTE_MAX = 240
CITATION_MAX = 400

_DATE = r"\d{4}-\d{2}-\d{2}"
_HEADING_RE = re.compile(r"^\s*##\s+(?P<nom>\S+)\s*$")

# Struck first, exactly as the core reads its own files: a struck line is
# struck whatever else it looks like, including when he left the tick in
# place while crossing it out.
_RAYE_RE = re.compile(
    r"^\s*-\s+(?:\[[^\]]*\]\s+)?~~(?P<inner>.+?)~~\s*(?P<tampon>.*?)\s*$"
)
_CASE_RE = re.compile(
    rf"^\s*-\s+\[(?P<case>[^\]]*)\]\s+(?P<corps>{_DATE}\s+·\s+\S+\s*:\s*\S.*?)\s*$"
)
_CORPS_RE = re.compile(
    rf"^(?P<date>{_DATE})\s+·\s+(?P<source>\S+)\s*:\s*(?P<texte>.+?)\s*$"
)
_CITATION_RE = re.compile(r"^\s*>\s*(?P<texte>.+?)\s*$")
_COCHE_RE = re.compile(r"^\s*[xX]\s*$")
_SUR_UNE_LIGNE = re.compile(r"^[^\r\n]{1,500}$")

_COMMENT_OPEN, _COMMENT_CLOSE = "<!--", "-->"

_HEADER = (
    "# Appris\n"
    "\n"
    "<!--\n"
    "  Ce qu'elle a cru remarquer dans ton journal, et que tu ne lui as\n"
    "  jamais dit.\n"
    "\n"
    "  Rien ici n'est une croyance. Tant qu'une ligne est dans ce fichier,\n"
    "  elle ne la sait pas : elle ne la lit dans aucune de ses réponses.\n"
    "\n"
    "  Trois gestes, un caractère chacun :\n"
    "    - [ ]  en attente : elle ne fait rien, même dans six mois\n"
    "    - [x]  tu es d'accord : à ta prochaine demande, la ligne part\n"
    "           dans profil.md ou regles.md, avec la source « confirmé »\n"
    "    ~~…~~  tu n'es pas d'accord : elle ne te la reproposera jamais\n"
    "\n"
    "  Réécris la phrase avant de cocher si elle est mal dite, ou si elle\n"
    "  n'est pas dans ta langue. Ce qui part dans profil.md est la ligne\n"
    "  telle qu'elle est ici, caractère pour caractère, pas celle qu'elle\n"
    "  avait proposée.\n"
    "\n"
    "  Sous chaque proposition, la phrase du journal d'où elle vient. Si\n"
    "  cette phrase ne te dit rien, raye la ligne.\n"
    "\n"
    "  Ce fichier est à toi. Rien ne le relit tout seul.\n"
    "\n"
    "  ATTENTION : écris SOUS ce bloc, pas dedans.\n"
    "-->\n"
    "\n"
    "## Profil\n"
    "\n"
    "## Règles\n"
)


@dataclass(frozen=True)
class Proposition:
    """One thing she thinks she noticed, and what he has done about it."""

    section: Optional[str]
    date: str
    texte: str
    citation: str
    etat: str
    # The line exactly as written, so a mark can find and replace it
    # without reformatting anything he typed himself.
    ligne: str = ""
    # What follows a struck line, when anything does. A stamp means the
    # harvest took it; a bare strike means he refused it. Both are
    # `rayée`, and losing the difference would tell him he refused
    # something he agreed to.
    tampon: str = ""


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so `## Règles` and `## regles` are
    the same heading. He types these himself when he reorganises."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


_SECTIONS = {
    "profil": SECTION_PROFIL,
    "regles": SECTION_REGLES,
}


def appris_path(cfg) -> Path:
    from ..memory.core import MemoryCore

    return MemoryCore.for_config(cfg).directory / APPRIS_FILENAME


# ── Reading ───────────────────────────────────────────────────────────

_CACHE: Dict[str, Any] = {"stamp": None, "items": None}


def invalidate_appris_cache() -> None:
    """Forget the mtime stamp.

    Two writes inside one filesystem tick are indistinguishable to an
    mtime cache, so a proposal written and immediately read back would
    come back as the file was a moment before it existed.
    """
    _CACHE["stamp"], _CACHE["items"] = None, None


def parse_appris(texte: str) -> List[Proposition]:
    """Every proposal in the file, in file order.

    Forgiving by design, like the core's parser. A line the grammar does
    not recognise is not a proposal and is left entirely alone: the file
    is his, and a note he typed between two proposals must not become
    something she can act on.
    """
    out: List[Proposition] = []
    section: Optional[str] = None
    dans_commentaire = False
    lignes = (texte or "").splitlines()

    for i, ligne in enumerate(lignes):
        if dans_commentaire:
            if _COMMENT_CLOSE in ligne:
                dans_commentaire = False
            continue
        if _COMMENT_OPEN in ligne:
            if _COMMENT_CLOSE not in ligne:
                dans_commentaire = True
            continue

        titre = _HEADING_RE.match(ligne)
        if titre:
            section = _SECTIONS.get(_fold(titre.group("nom")))
            continue

        etat: Optional[str] = None
        corps = ""

        # Struck first. He may have crossed a line out while leaving its
        # tick in place, and what he crossed out is what he meant.
        tampon = ""
        raye = _RAYE_RE.match(ligne)
        if raye:
            etat, corps = ETAT_RAYEE, raye.group("inner").strip()
            tampon = (raye.group("tampon") or "").strip(" ·")
        else:
            case = _CASE_RE.match(ligne)
            if case:
                marque = case.group("case")
                etat = ETAT_COCHEE if _COCHE_RE.match(marque) else ETAT_ATTENTE
                corps = case.group("corps")

        if etat is None:
            continue

        champs = _CORPS_RE.match(corps)
        if not champs:
            continue

        citation = ""
        if i + 1 < len(lignes):
            suite = _CITATION_RE.match(lignes[i + 1])
            if suite:
                citation = suite.group("texte").strip().strip("«»").strip()

        out.append(Proposition(
            section=section,
            date=champs.group("date"),
            texte=champs.group("texte").strip(),
            citation=citation,
            etat=etat,
            ligne=ligne,
            tampon=tampon,
        ))

    return out


def load_appris(cfg) -> List[Proposition]:
    """Read the proposals, cached on the file's mtime.

    Never raises. An unreadable file yields none, which is the safe
    direction: a corrupt file must not invent proposals, and must not
    stop her working.
    """
    try:
        path = appris_path(cfg)
        stamp = path.stat().st_mtime_ns if path.exists() else None
    except Exception:
        return []

    if stamp is None:
        return []
    if _CACHE["stamp"] == stamp and _CACHE["items"] is not None:
        return _CACHE["items"]

    try:
        items = parse_appris(path.read_text(encoding="utf-8"))
    except Exception as e:
        debug_log(f"appris file unreadable, no proposal will be seen: {e}", "tools")
        return []

    _CACHE["stamp"], _CACHE["items"] = stamp, items
    return items


# ── Writing ───────────────────────────────────────────────────────────


def render_proposition(*, date: str, texte: str, citation: str) -> str:
    """One proposal as two lines, or "" when it cannot be written safely.

    What reaches here came out of a model reading his diary. A line break
    would forge a second proposal, a heading, or a tick; a `~~` would
    render as already refused. Refusing to write is always legal.
    """
    date = (date or "").strip()
    texte = (texte or "").strip()
    citation = (citation or "").strip()

    if not re.fullmatch(_DATE, date):
        return ""
    if not texte or not _SUR_UNE_LIGNE.match(texte) or len(texte) > TEXTE_MAX:
        return ""
    if "~~" in texte or "~~" in citation:
        return ""
    if citation and (not _SUR_UNE_LIGNE.match(citation) or len(citation) > CITATION_MAX):
        return ""

    ligne = f"- [ ] {date} · {SOURCE_JOURNAL} : {texte}\n"
    if citation:
        ligne += f"  > « {citation} »\n"
    return ligne


def _write_guarded(cfg, compose) -> bool:
    """Apply one composed change to the file, or refuse.

    The same discipline as the goals page, for the same reason: he may
    have this open in an editor, and the value passing through came from
    a model reading a transcription of his own days.

    Never raises. A refusal leaves the file exactly as it was, which is
    always a legal state.
    """
    tmp = None
    try:
        path = appris_path(cfg)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_HEADER, encoding="utf-8")

        avant = path.stat()
        lignes = path.read_text(encoding="utf-8").splitlines(keepends=True)

        apres = compose(lignes)
        if apres is None:
            return False

        maintenant = path.stat()
        if (maintenant.st_mtime_ns != avant.st_mtime_ns
                or maintenant.st_size != avant.st_size):
            debug_log("appris.md moved underneath, nothing written", "tools")
            return False

        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        tmp.write_text("".join(apres), encoding="utf-8")
        os.chmod(tmp, avant.st_mode & 0o777)
        os.replace(tmp, path)
        tmp = None
        invalidate_appris_cache()
        return True
    except Exception as e:
        debug_log(f"appris.md not written: {e}", "tools")
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def _fin_de_section(lignes: List[str], section: str) -> Optional[int]:
    """Where to insert, i.e. just past the last line of that section."""
    debut = None
    for i, ligne in enumerate(lignes):
        titre = _HEADING_RE.match(ligne)
        if not titre:
            continue
        if debut is None and _SECTIONS.get(_fold(titre.group("nom"))) == section:
            debut = i
        elif debut is not None:
            return i
    if debut is None:
        return None
    return len(lignes)


def ajouter_propositions(
    cfg, items: Sequence[Tuple[str, str, str, str]],
) -> bool:
    """Append proposals, each as `(section, date, texte, citation)`.

    Every proposal must render, or nothing is written: a partial batch
    would leave him agreeing to a list that is not the list she found.
    """
    rendus: List[Tuple[str, str]] = []
    for section, date, texte, citation in items:
        if section not in (SECTION_PROFIL, SECTION_REGLES):
            return False
        bloc = render_proposition(date=date, texte=texte, citation=citation)
        if not bloc:
            return False
        rendus.append((section, bloc))
    if not rendus:
        return False

    def compose(lignes: List[str]) -> Optional[List[str]]:
        out = list(lignes)
        for section, bloc in rendus:
            fin = _fin_de_section(out, section)
            if fin is None:
                return None
            # Trailing blank lines belong to the gap between sections,
            # not to the section, so step back over them.
            while fin > 0 and not out[fin - 1].strip():
                fin -= 1
            out[fin:fin] = bloc.splitlines(keepends=True)
        return out

    return _write_guarded(cfg, compose)


def marquer_refusee(cfg, ligne: str) -> bool:
    """Strike one proposal through, with no stamp.

    A struck line carrying no stamp is exactly what he writes by hand,
    so the file reads the same whichever door the refusal came through.
    The strike alone is the refusal; the stamp is bookkeeping the harvest
    adds when it takes something.

    Returns False when the line is gone, which is what happens when he
    deleted it in his editor between the page loading and the click.
    """
    cible = (ligne or "").rstrip("\n")
    if not cible:
        return False

    def compose(lignes: List[str]) -> Optional[List[str]]:
        out = list(lignes)
        for i, l in enumerate(out):
            if l.rstrip("\n") != cible:
                continue
            case = _CASE_RE.match(l)
            if not case:
                return None
            fin = "\n" if l.endswith("\n") else ""
            out[i] = f"- ~~{case.group('corps')}~~{fin}"
            return out
        return None

    return _write_guarded(cfg, compose)


def marquer_retenue(cfg, ligne: str, aujourdhui: str) -> bool:
    """Strike one harvested proposal through, stamped with today.

    The date on the proposal is the day of the journal row it came from;
    this stamp is the day he agreed. Both belong in the file whose job is
    provenance, and only one of them belongs in `profil.md`.

    Returns False when the line is gone, which is what happens when he
    deleted it by hand between the read and the write.
    """
    cible = (ligne or "").rstrip("\n")
    if not cible:
        return False

    def compose(lignes: List[str]) -> Optional[List[str]]:
        out = list(lignes)
        for i, l in enumerate(out):
            if l.rstrip("\n") != cible:
                continue
            case = _CASE_RE.match(l)
            if not case:
                return None
            corps = case.group("corps")
            fin = "\n" if l.endswith("\n") else ""
            out[i] = f"- ~~{corps}~~ · retenu le {aujourdhui}{fin}"
            return out
        return None

    return _write_guarded(cfg, compose)
