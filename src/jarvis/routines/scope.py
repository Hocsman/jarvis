"""
🧾 The envelope: what one routine may reach, in a file you can edit.

Everything else about a routine is invisible — it runs at 07:00 and the
user is asleep. This file is the one place they can answer "what are my
routines allowed to do?" a month later without recalling a sentence they
said in July. The same information inside a JSON blob in a database row
would answer nothing: nothing to read, nothing to correct, and tightening
a routine would cost as much as deleting it.

Every default leans shut. An empty tool list is an empty envelope, never
"all". A routine with no block is suspended, never unrestricted. A
malformed file yields the blocks that parse and ignores the rest: it must
not raise, because a syntax error in a file the user edits would stop
every routine at once, and it must not open anything up.

There are no wildcards, unlike `outils.md`. That file has them because
the user is present to see the result; here nobody is, and a server that
gains a tool overnight would gain it inside the envelope too.

See routines.spec.md for the full contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..debug import debug_log

ROUTINES_FILENAME = "routines.md"

_HEADING_RE = re.compile(r"^\s*##\s+(?P<name>\S+)\s*$")
_FIELD_RE = re.compile(r"^\s*(?P<key>phrase|quand|mémoire|memoire)\s*:\s*(?P<value>.*)$")
_TOOLS_RE = re.compile(r"^\s*outils\s*:\s*$")
_ITEM_RE = re.compile(r"^\s*-\s+(?P<name>\S+)\s*$")
_COMMENT_OPEN, _COMMENT_CLOSE = "<!--", "-->"


# Three names an envelope can never contain, whatever the file says.
#
# `toolSearchTool` appends any name in the registry to the running turn's
# allow-list and regenerates the schema — an envelope that can widen
# itself is not an envelope. `refreshMCPTools` rediscovers servers
# mid-run, so the catalogue would change underneath it. `stop` ends a
# conversation, and a routine is not one.
JAMAIS_EN_ROUTINE = frozenset({"toolSearchTool", "refreshMCPTools", "stop"})

# The one value that turns the profile on. Anything else — absent,
# empty, negated, a word from another language, a typo — leaves it off.
# Failing shut in every direction beats a list of affirmatives that has
# to be right in every language the user might type.
_MEMOIRE_OUI = "oui"


@dataclass(frozen=True)
class RoutineScope:
    """The tools one routine may reach, and nothing else.

    ``memoire`` is off unless the block asks for it. The warm profile is
    the user's own life, and it does not need to leave the machine at 7am
    for a routine to summarise their mail.
    """

    nom: str
    outils: List[str]
    memoire: bool = False

    def allows(self, tool_name: str) -> bool:
        """Exact names only.

        No prefixes and no wildcards: a line reading
        `chrome-devtools__take` must not admit
        `chrome-devtools__take_heapsnapshot`, and a server that grows a
        tool overnight must not grow the envelope with it.
        """
        if tool_name in JAMAIS_EN_ROUTINE:
            return False
        return tool_name in self.outils


@dataclass
class RoutineBlock:
    """One routine as written in the file."""

    nom: str
    phrase: str = ""
    quand: str = ""
    memoire: str = ""
    outils: List[str] = field(default_factory=list)

    def scope(self) -> RoutineScope:
        return RoutineScope(
            nom=self.nom,
            outils=list(self.outils),
            memoire=self.memoire.strip().lower() == _MEMOIRE_OUI,
        )


def parse_routines(text: str) -> Dict[str, RoutineBlock]:
    """Read the file. Anything unrecognised is skipped, never guessed."""
    blocks: Dict[str, RoutineBlock] = {}
    current: Optional[RoutineBlock] = None
    in_tools = False
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
            current = RoutineBlock(nom=heading.group("name").strip())
            blocks[current.nom] = current
            in_tools = False
            continue

        if current is None:
            continue

        if _TOOLS_RE.match(line):
            in_tools = True
            continue

        field_match = _FIELD_RE.match(line)
        if field_match:
            in_tools = False
            key = field_match.group("key")
            value = field_match.group("value").strip()
            if key == "phrase":
                current.phrase = value
            elif key == "quand":
                current.quand = value
            else:
                current.memoire = value
            continue

        if in_tools:
            item = _ITEM_RE.match(line)
            if item:
                current.outils.append(item.group("name").strip())
            elif line.strip():
                in_tools = False

    return blocks


_CACHE: Dict[str, Any] = {"stamp": None, "blocks": None}


def routines_path(cfg):
    from ..memory.core import MemoryCore

    return MemoryCore.for_config(cfg).directory / ROUTINES_FILENAME


def load_routines(cfg) -> Dict[str, RoutineBlock]:
    """Read the envelopes, cached on the file's mtime.

    Re-read when it changes, so an edit takes effect on the next run
    rather than the next restart: the file is the control surface, and a
    control surface with a restart delay is one people stop trusting.

    Never raises. An unreadable file yields no routines, which suspends
    them all — the safe direction, and the same class of failure the
    policy loader already handles for a Windows editor saving as ANSI.
    """
    try:
        path = routines_path(cfg)
        stamp = path.stat().st_mtime_ns if path.exists() else None
    except Exception:
        return {}

    if stamp is None:
        return {}
    if _CACHE["stamp"] == stamp and _CACHE["blocks"] is not None:
        return _CACHE["blocks"]

    try:
        blocks = parse_routines(path.read_text(encoding="utf-8"))
    except Exception as e:
        debug_log(f"routines file unreadable, none will run: {e}", "tools")
        return {}

    _CACHE["stamp"], _CACHE["blocks"] = stamp, blocks
    return blocks


def scope_for(cfg, nom: str) -> Optional[RoutineScope]:
    """The envelope for one routine, or None if it has no block.

    None means suspended, never unrestricted. That is what makes
    deleting the block an off switch a user can reach with a text editor.
    """
    block = load_routines(cfg).get(nom)
    return block.scope() if block is not None else None


_HEADER = (
    "# Routines\n"
    "\n"
    "<!--\n"
    "  Ce que Yuba fait toute seule, à heure fixe, sans que tu sois là.\n"
    "\n"
    "  Un bloc par routine. La liste « outils » est son périmètre : elle\n"
    "  ne peut atteindre QUE ces outils-là, et seulement pour lire.\n"
    "\n"
    "  Retire une ligne d'outil et la routine est resserrée.\n"
    "  Supprime le bloc et la routine est suspendue.\n"
    "  Une liste vide n'est pas « tous » : c'est aucun.\n"
    "\n"
    "  Par défaut une routine ne sait rien de toi : ton profil et tes\n"
    "  règles ne partent pas avec elle. Écris « mémoire: oui » dans un\n"
    "  bloc si cette routine-là en a besoin. Tout autre texte vaut non.\n"
    "\n"
    "  Pas de joker ici, contrairement à outils.md : là-bas tu es présent\n"
    "  pour voir le résultat, ici non, et un serveur qui gagne un outil\n"
    "  pendant la nuit le gagnerait aussi dans le périmètre.\n"
    "-->\n"
)


def render_block(*, nom: str, phrase: str, quand: str, outils: List[str],
                 ecartes: Optional[List[Tuple[str, str]]] = None) -> str:
    """Write one block, including what was left out and why.

    The rejected tools are named as a comment, so the user can put one
    back knowingly rather than discovering the gap when the routine
    quietly does nothing for a week. A comment, because a rejected tool
    that parsed back as an allowed one would be the worst possible
    outcome of explaining yourself.
    """
    lines = [f"## {nom}", f"phrase: {phrase}", f"quand: {quand}", "outils:"]
    lines.extend(f"- {name}" for name in outils)
    if not outils:
        lines.append("<!-- vide : cette routine ne peut rien atteindre -->")
    if ecartes:
        lines.append("")
        lines.append("<!--")
        lines.append("  Écartés du périmètre, avec la raison. Ajoute une ligne")
        lines.append("  « - nom » ci-dessus pour en autoriser un.")
        for name, why in ecartes:
            lines.append(f"    {name} : {why}")
        lines.append("-->")
    return "\n".join(lines) + "\n"


def ensure_routines_file(cfg) -> None:
    """Create the file with its header, once."""
    try:
        path = routines_path(cfg)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_HEADER, encoding="utf-8")
    except Exception as e:
        debug_log(f"routines file not created: {e}", "tools")
