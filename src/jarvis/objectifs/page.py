"""
🎯 A goal, in a file you can read a month later.

The third artefact the user owns outright, after the core and the
routines envelope, and it inherits both their laws.

**Nothing here is ever written by deduction.** That is the rule this
whole fork rests on, and a goal is the first thing in it whose state
*wants* to be inferred — she watches the work progress, so it is tempting
to let her write down what she concludes. Every line of state therefore
carries a date and the source it came from, and the only source this
slice can produce is the user's own words.

The grammar is `routines.md`'s, deliberately: one heading per goal,
fields, and progress as dated items under `points:` — the same `- ` list
the tool envelope uses. Two files that parse alike are two files the user
learns once.

What is absent is load-bearing. There is no `cadence:` and no `outils:`,
so a goal has no unattended pass and no envelope: it is something she
remembers and brings up, never something that acts. Arming one is a
separate grant with its own card, and it does not exist yet — a field
that parsed today would be a capability nobody designed.

See objectifs.spec.md for the full contract.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from ..debug import debug_log

OBJECTIFS_FILENAME = "objectifs.md"

# Where a line of state came from. `dit` is the user's own words; the
# vocabulary exists so that a source this slice cannot produce is
# nonetheless nameable, because the day one is added it must arrive as a
# new word rather than as an unmarked line among his.
SOURCE_DIT = "dit"

_HEADING_RE = re.compile(r"^\s*##\s+(?P<name>\S+)\s*$")
_FIELD_RE = re.compile(
    r"^\s*(?P<key>phrase|fini quand|clos)\s*:\s*(?P<value>.*)$"
)
_POINTS_RE = re.compile(r"^\s*points\s*:\s*$")
_POINT_RE = re.compile(
    r"^\s*-\s+(?P<date>\d{4}-\d{2}-\d{2})\s*·\s*(?P<source>\w+)\s*·\s*(?P<texte>.+?)\s*$"
)
_COMMENT_OPEN, _COMMENT_CLOSE = "<!--", "-->"

# A heading this code would be willing to write and read back. The name
# reaches here from a model reading a Whisper transcription, so it is not
# ours: a newline in it appends a field, a heading, or a whole goal.
_NOM_PROPRE = re.compile(r"^[\w-]{1,64}$", re.UNICODE)

# One line, and one line only. What the user says arrives through the
# same two lossy layers, and a line break in it forges file structure.
_SUR_UNE_LIGNE = re.compile(r"^[^\r\n]{1,500}$")


@dataclass(frozen=True)
class Point:
    """One dated line of state, and where it came from."""

    date: str
    source: str
    texte: str


@dataclass
class Objectif:
    """One goal as written in the file."""

    nom: str
    phrase: str = ""
    fini_quand: str = ""
    clos: str = ""
    points: List[Point] = field(default_factory=list)

    @property
    def est_ouvert(self) -> bool:
        return not self.clos.strip()


def parse_objectifs(text: str) -> Dict[str, Objectif]:
    """Read the file. Anything unrecognised is skipped, never guessed.

    A bare note under `points:` is ignored rather than dated today:
    guessing a date is inventing state, which is the one thing this file
    exists to prevent.
    """
    blocks: Dict[str, Objectif] = {}
    current: Optional[Objectif] = None
    in_points = False
    in_comment = False

    for line in (text or "").splitlines():
        if in_comment:
            if _COMMENT_CLOSE in line:
                in_comment = False
            continue
        if _COMMENT_OPEN in line:
            if _COMMENT_CLOSE not in line:
                in_comment = True
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            nom = heading.group("name").strip()
            # First wins, unlike the routines file. A goal's identity is
            # its name: a second block of the same name would silently
            # shadow the one the user has been correcting by hand.
            current = blocks.get(nom)
            if current is None:
                current = Objectif(nom=nom)
                blocks[nom] = current
            in_points = False
            continue

        if current is None:
            continue

        if _POINTS_RE.match(line):
            in_points = True
            continue

        field_match = _FIELD_RE.match(line)
        if field_match:
            in_points = False
            key = field_match.group("key")
            value = field_match.group("value").strip()
            if key == "phrase":
                current.phrase = value
            elif key == "clos":
                current.clos = value
            else:
                current.fini_quand = value
            continue

        if in_points:
            point = _POINT_RE.match(line)
            if point:
                current.points.append(Point(
                    date=point.group("date"), source=point.group("source"),
                    texte=point.group("texte").strip(),
                ))
            elif line.lstrip().startswith("-"):
                # An item that is not a dated point: the user typing a
                # bare note. Skipped rather than dated today — guessing a
                # date is inventing state — but it does not end the list,
                # or one hand-written line would hide every point below
                # it.
                continue
            elif line.strip():
                in_points = False

    return blocks


def render_objectif(*, nom: str, phrase: str, fini_quand: str,
                    points: Optional[List[Point]] = None,
                    clos: str = "") -> str:
    """Write one block, or nothing when it could not be read back.

    Filtered here rather than at the call sites, so no future caller can
    route round it: the name and the notes arrive from a model reading a
    Whisper transcription, and a line break in either forges structure.
    """
    if not _NOM_PROPRE.match(nom or ""):
        debug_log(f"objectif not rendered: unusable name {nom!r}", "tools")
        return ""
    if not _SUR_UNE_LIGNE.match(phrase or "") or not _SUR_UNE_LIGNE.match(
        fini_quand or ""
    ):
        debug_log(f"objectif {nom} not rendered: unusable field", "tools")
        return ""

    lines = [f"## {nom}", f"phrase: {phrase}", f"fini quand: {fini_quand}"]
    if clos:
        lines.append(f"clos: {clos}")

    gardes = [
        p for p in (points or [])
        if _SUR_UNE_LIGNE.match(p.texte or "") and _NOM_PROPRE.match(p.source or "")
    ]
    if gardes:
        lines.append("points:")
        lines.extend(f"- {p.date} · {p.source} · {p.texte}" for p in gardes)
    return "\n".join(lines) + "\n"


def render_point(texte: str, *, source: str = SOURCE_DIT,
                 jour: Optional[str] = None) -> str:
    """One dated line, or empty when it could not be read back."""
    if not _SUR_UNE_LIGNE.match(texte or ""):
        return ""
    return f"- {jour or date.today().isoformat()} · {source} · {texte.strip()}\n"


_CACHE: Dict[str, Any] = {"stamp": None, "blocks": None}


def objectifs_path(cfg):
    from ..memory.core import MemoryCore

    return MemoryCore.for_config(cfg).directory / OBJECTIFS_FILENAME


def invalidate_objectifs_cache() -> None:
    """Forget the mtime stamp.

    Two writes inside one filesystem tick are indistinguishable to an
    mtime cache, so a goal written and immediately read back would come
    back as the file was a moment before it existed.
    """
    _CACHE["stamp"], _CACHE["blocks"] = None, None


def load_objectifs(cfg) -> Dict[str, Objectif]:
    """Read the goals, cached on the file's mtime.

    Never raises. An unreadable file yields no goals, which is the safe
    direction here for the same reason as everywhere else in this fork: a
    corrupt file must not invent state, and must not stop her working.
    """
    try:
        path = objectifs_path(cfg)
        stamp = path.stat().st_mtime_ns if path.exists() else None
    except Exception:
        return {}

    if stamp is None:
        return {}
    if _CACHE["stamp"] == stamp and _CACHE["blocks"] is not None:
        return _CACHE["blocks"]

    try:
        blocks = parse_objectifs(path.read_text(encoding="utf-8"))
    except Exception as e:
        debug_log(f"objectifs file unreadable, none will be seen: {e}", "tools")
        return {}

    _CACHE["stamp"], _CACHE["blocks"] = stamp, blocks
    return blocks


def _span(lignes: List[str], nom: str):
    """Where one block starts and ends, ignoring comments.

    The file's own header explains the grammar using the same words the
    fields use, so a scan that does not track comment state would edit
    the documentation — and a parse-based check could not see it, since
    a comment never becomes a block.
    """
    in_comment = False
    debut = None
    for i, ligne in enumerate(lignes):
        if in_comment:
            if _COMMENT_CLOSE in ligne:
                in_comment = False
            continue
        if _COMMENT_OPEN in ligne:
            if _COMMENT_CLOSE not in ligne:
                in_comment = True
            continue
        heading = _HEADING_RE.match(ligne)
        if not heading:
            continue
        if debut is not None:
            return debut, i
        if heading.group("name").strip() == nom:
            debut = i
    return (debut, len(lignes)) if debut is not None else (None, None)


def _compose_point(lignes: List[str], nom: str, point: Point):
    """The file with one dated line added to a block, or None."""
    debut, fin = _span(lignes, nom)
    if debut is None:
        return None

    rendu = render_point(point.texte, source=point.source, jour=point.date)
    if not rendu:
        return None

    dernier = None
    liste = None
    in_comment = False
    for i in range(debut + 1, fin):
        ligne = lignes[i]
        if in_comment:
            if _COMMENT_CLOSE in ligne:
                in_comment = False
            continue
        if _COMMENT_OPEN in ligne:
            if _COMMENT_CLOSE not in ligne:
                in_comment = True
            continue
        if _POINTS_RE.match(ligne):
            liste = i
        elif _POINT_RE.match(ligne):
            dernier = i

    out = list(lignes)
    if dernier is not None:
        out.insert(dernier + 1, rendu)
    elif liste is not None:
        out.insert(liste + 1, rendu)
    else:
        # No list yet. Added at the end of the block's own span so the
        # heading and the fields stay where the user expects them.
        queue = fin
        while queue > debut + 1 and not lignes[queue - 1].strip():
            queue -= 1
        out.insert(queue, "points:\n")
        out.insert(queue + 1, rendu)
    return out


def _compose_clos(lignes: List[str], nom: str, valeur: str):
    """The file with a block's closing line added, or None.

    Only ever added. A goal already closed is left alone: reopening one
    by overwriting the line is a different act, and it is his.
    """
    debut, fin = _span(lignes, nom)
    if debut is None or not _SUR_UNE_LIGNE.match(valeur or ""):
        return None

    apres = debut + 1
    in_comment = False
    for i in range(debut + 1, fin):
        ligne = lignes[i]
        if in_comment:
            if _COMMENT_CLOSE in ligne:
                in_comment = False
            continue
        if _COMMENT_OPEN in ligne:
            if _COMMENT_CLOSE not in ligne:
                in_comment = True
            continue
        champ = _FIELD_RE.match(ligne)
        if champ:
            if champ.group("key") == "clos":
                return None
            apres = i + 1

    out = list(lignes)
    out.insert(apres, f"clos: {valeur}\n")
    return out


def _write_guarded(cfg, compose) -> bool:
    """Apply one composed change to the file, or refuse.

    The same discipline as `rewrite_quand`, for the same reason: the user
    may have this open in an editor, and the value passing through came
    from a model reading a transcription.

    Never raises. A refusal leaves the file exactly as it was, which is
    always a legal state.
    """
    tmp = None
    try:
        path = objectifs_path(cfg)
        avant_stat = path.stat()
        lignes = path.read_text(encoding="utf-8").splitlines(keepends=True)

        apres = compose(lignes)
        if apres is None:
            return False

        # Byte for byte, not parse-alike: two files can parse the same
        # and differ in a paragraph the user wrote. One or two lines
        # added, everything else identical and merely shifted down.
        ajout = len(apres) - len(lignes)
        if ajout not in (1, 2):
            return False
        idx = next((i for i, (a, b) in enumerate(zip(lignes, apres)) if a != b),
                   len(lignes))
        if lignes[idx:] != apres[idx + ajout:]:
            return False

        maintenant = path.stat()
        if (maintenant.st_mtime_ns != avant_stat.st_mtime_ns
                or maintenant.st_size != avant_stat.st_size):
            debug_log("objectifs.md moved underneath, nothing written", "tools")
            return False

        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
        tmp.write_text("".join(apres), encoding="utf-8")
        os.chmod(tmp, avant_stat.st_mode & 0o777)
        os.replace(tmp, path)
        tmp = None
        invalidate_objectifs_cache()
        return True
    except Exception as e:
        debug_log(f"objectifs.md not written: {e}", "tools")
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def append_point(cfg, nom: str, point: Point) -> bool:
    """Add one dated line of state to a goal. Returns whether it did."""
    fait = _write_guarded(cfg, lambda lignes: _compose_point(lignes, nom, point))
    if fait:
        debug_log(f"    🎯 {nom}: {point.source} · {point.texte[:60]}", "tools")
    return fait


def close_objectif(cfg, nom: str, valeur: str) -> bool:
    """Write the line that ends a goal. Returns whether it did."""
    fait = _write_guarded(cfg, lambda lignes: _compose_clos(lignes, nom, valeur))
    if fait:
        debug_log(f"    🎯 {nom} closed: {valeur}", "tools")
    return fait


def ouverts(cfg) -> List[Objectif]:
    """The goals still going, in the order the file lists them."""
    return [o for o in load_objectifs(cfg).values() if o.est_ouvert]


def resolve(cfg, nom: str) -> Optional[Objectif]:
    """The goal a name refers to, or None.

    A name already in the file is an identity and is taken as written —
    the sanitiser exists to clean a name on its way *into* the file, and
    putting an existing heading through it makes a tool refuse names it
    wrote itself.

    With no name given and exactly one goal open, that is the one meant.
    With several, nothing is guessed: picking the wrong goal writes a
    fact about the wrong thing, under his name.
    """
    blocs = load_objectifs(cfg)
    nom = (nom or "").strip()
    if nom:
        return blocs.get(nom)
    encore = ouverts(cfg)
    return encore[0] if len(encore) == 1 else None
