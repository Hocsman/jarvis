"""
🚪 Tool policy — what the assistant may do, and what it must ask about.

Every tool call in the process funnels through ``run_tool_with_retries``.
This module decides, deterministically, what happens there: a risk level
for the tool about to run, and a verdict the user controls from a file
they can read.

Deterministic on purpose. Asking a model "is this dangerous?" would be
more flexible and would cover tools nobody annotated, but a gate whose
answer varies between two identical calls is not a gate.

The asymmetry behind every default: refusing a harmless tool costs a
turn, allowing a destructive one costs something that may not come back.

See policy.spec.md for the full contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

# What a tool does to the world.
RISK_READ = "lecture"
RISK_ACTION = "action"
RISK_DESTRUCTIVE = "destructif"

# What the gate does about it.
FREE = "libre"
ASK = "demande"
NEVER = "jamais"

# How a call ended, as written to the action ledger. `refusé` means the
# policy said no and nothing was asked of anyone; `décliné` means the
# user was asked and said no. Keeping those apart is the difference
# between "she would not" and "I would not".
OUTCOME_OK = "ok"
OUTCOME_FAILED = "échec"
OUTCOME_REFUSED = "refusé"
OUTCOME_ASKED = "demandé"
OUTCOME_DECLINED = "décliné"
OUTCOME_EXPIRED = "expiré"
# A routine reached past its own envelope. Kept apart from `refusé`
# because the two point at different files: `refusé` is the user's policy
# saying no to everyone, `hors-cadre` is one routine's own scope, which
# they widen by adding a line to `routines.md` rather than by loosening
# anything for every origin.
OUTCOME_OUT_OF_SCOPE = "hors-cadre"

_SECTIONS = {
    "libre": FREE,
    "demande": ASK,
    "jamais": NEVER,
}

# Risk to default verdict, used when the user's file says nothing. Acting
# and destroying both land on ASK: the difference between them is what
# the confirmation prompt shows, not whether one is needed.
_DEFAULT_VERDICT = {
    RISK_READ: FREE,
    RISK_ACTION: ASK,
    RISK_DESTRUCTIVE: ASK,
}

_HEADING_RE = re.compile(r"^\s*##\s+(?P<name>\S+)\s*$", re.IGNORECASE)
_ENTRY_RE = re.compile(r"^\s*-\s+(?P<name>\S+)\s*$")
_COMMENT_OPEN, _COMMENT_CLOSE = "<!--", "-->"


def resolve_risk(name: str, tool: Any, args: Optional[Dict[str, Any]]) -> str:
    """What the tool about to run does to the world.

    ``tool`` is a builtin ``Tool``, an MCP ``ToolSpec``, or None when the
    catalogue has never been told about this name.

    An unknown tool is destructive. That is not caution for its own sake:
    the MCP branch of the funnel fires on the server name alone and never
    consults the tool cache, so a cold cache or a failed discovery would
    otherwise walk a completely unclassified tool straight through.
    """
    if tool is None:
        return RISK_DESTRUCTIVE

    # Builtins declare their own, and may read their arguments to do it:
    # localFiles reads, writes and deletes through one name.
    risk_for = getattr(tool, "risk_for", None)
    if callable(risk_for):
        try:
            declared = risk_for(args or {})
        except Exception:
            return RISK_DESTRUCTIVE
        if declared in _DEFAULT_VERDICT:
            return declared
        return RISK_DESTRUCTIVE

    # MCP servers ship this in a standard field. Reading it is the whole
    # point: the servers already tell us which of their tools destroy
    # things, and the catalogue used to throw that away.
    annotations = getattr(tool, "annotations", None)
    if isinstance(annotations, dict) and annotations:
        if annotations.get("destructiveHint") is True:
            return RISK_DESTRUCTIVE
        if annotations.get("readOnlyHint") is True:
            return RISK_READ
        return RISK_ACTION

    # A tool we hold a spec for but no annotation is still a tool nobody
    # classified. Same reasoning as above.
    return RISK_DESTRUCTIVE


@dataclass(frozen=True)
class ToolPolicy:
    """The user's verdicts, parsed from their own file."""

    exact: Dict[str, str]
    wildcards: Dict[str, str]

    @classmethod
    def empty(cls) -> "ToolPolicy":
        return cls(exact={}, wildcards={})

    @classmethod
    def parse(cls, text: str) -> "ToolPolicy":
        """Read a policy file. Anything unrecognised is simply skipped.

        A malformed file yields the defaults rather than an exception or
        a blanket allow: a corrupt file must not silently unlock the
        machine, and it must not stop the assistant working either.
        """
        exact: Dict[str, str] = {}
        wildcards: Dict[str, str] = {}
        verdict: Optional[str] = None
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
                verdict = _SECTIONS.get(heading.group("name").strip().lower())
                continue

            entry = _ENTRY_RE.match(line)
            if entry and verdict:
                name = entry.group("name").strip()
                if name.endswith("*"):
                    wildcards[name[:-1]] = verdict
                else:
                    exact[name] = verdict

        return cls(exact=exact, wildcards=wildcards)

    def verdict(self, name: str, risk: str) -> str:
        """What to do about ``name`` running at ``risk``.

        Resolution order: the exact name the user wrote, then the widest
        matching wildcard, then the default for that risk. An exact entry
        beats a wildcard so a single tool can be carved out of a server
        the user otherwise trusts.
        """
        if name in self.exact:
            return self.exact[name]

        match: Optional[str] = None
        longest = -1
        for prefix, decision in self.wildcards.items():
            if name.startswith(prefix) and len(prefix) > longest:
                match, longest = decision, len(prefix)
        if match is not None:
            return match

        return _DEFAULT_VERDICT.get(risk, ASK)


_HEADER = (
    "# Outils\n"
    "\n"
    "<!--\n"
    "  Ce que Yuba a le droit de faire, et ce qu'elle doit te demander.\n"
    "\n"
    "  Trois sections, et tu déplaces les lignes d'une à l'autre :\n"
    "    ## Libre    elle le fait sans rien demander\n"
    "    ## Demande  elle te montre ce qu'elle veut faire, tu valides\n"
    "    ## Jamais   refusé quoi qu'il arrive, aucune validation possible\n"
    "\n"
    "  Un joker couvre tout un serveur : « - chrome-devtools__* ».\n"
    "  Un nom exact l'emporte sur un joker, donc tu peux libérer un\n"
    "  serveur entier et en retirer un seul outil.\n"
    "\n"
    "  Ce fichier a été généré depuis tes outils réels au premier\n"
    "  lancement, et il est à toi : rien ne le réécrit ensuite. Un outil\n"
    "  apparu depuis et absent d'ici est traité comme destructif, donc\n"
    "  il te sera demandé.\n"
    "-->\n"
)


def render_policy_file(
    builtin_risks: Dict[str, str], mcp_risks: Dict[str, str],
) -> str:
    """Write the starting policy from the tools actually installed.

    Generated rather than shipped as a default, so the user opens it and
    sees their own catalogue — their 29 Chrome tools by name — instead of
    a list that may not match what they have.
    """
    buckets: Dict[str, list] = {FREE: [], ASK: [], NEVER: []}
    for name, risk in sorted({**builtin_risks, **mcp_risks}.items()):
        buckets[_DEFAULT_VERDICT.get(risk, ASK)].append(name)

    out = [_HEADER]
    for verdict, heading in ((FREE, "Libre"), (ASK, "Demande"), (NEVER, "Jamais")):
        out.append(f"\n## {heading}\n")
        if buckets[verdict]:
            out.extend(f"- {n}\n" for n in buckets[verdict])
        elif verdict == NEVER:
            out.append("\n<!-- vide : ajoute ici ce que tu veux interdire pour de bon -->\n")
    return "".join(out)
