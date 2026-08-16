"""Where a world fact came from, and how the line says so.

The core files are strict about the origin of a belief: every line of
``profil.md`` carries its source, and an ``appris.md`` proposal becomes a
belief only when he ticks it. The ``world`` branch of the graph had none
of that, and it is the branch fed by pages the assistant did not write.

This module holds the vocabulary and the line format. The rule that uses
them — a window in which no tool ran holds no lookup, so it yields no
facts and the model is not asked — lives in ``graph_ops``. Both the store
and the ops import from here, which is why it is its own module rather
than living in either.

See ``provenance.spec.md``.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

# One word per fact, recorded at write time, never inferred later.
#
# There is deliberately no word for "the model said it". A fact with no
# tool behind it does not get a weaker label, it does not get written.
SOURCE_WEB = "web"          # an untrusted page was fetched in the window
SOURCE_TOOL = "outil"       # some other tool ran: weather, MCP, builtin
SOURCE_UNKNOWN = "inconnu"  # written before this existed, or unestablishable
SOURCES = (SOURCE_WEB, SOURCE_TOOL, SOURCE_UNKNOWN)

# Tools whose output is a page the assistant did not write.
_WEB_TOOLS = frozenset({"webSearch", "fetchWebPage"})

# The source travels on the fact's own line, the way `profil.md` writes
# `- il habite à Genève · dit`. A fact is a line inside a node's `data`
# blob and one node accumulates facts from many windows on many days, so
# a column would describe the node rather than the fact.
#
# Recognised only when the suffix matches the vocabulary and an ISO date
# exactly, so a fact whose own text uses a middle dot survives whole. A
# line with no suffix is `inconnu` by construction — which is why there
# is no migration: nothing written before this needs marking.
_SEP = " · "
_SUFFIX_RE = re.compile(
    r"\s*·\s*(" + "|".join(SOURCES) + r")(?:\s*·\s*(\d{4}-\d{2}-\d{2}))?\s*$"
)


def source_for_tools(tools_used: Optional[Sequence[str]]) -> str:
    """The one word describing what a window's facts rest on.

    ``None`` means the caller could not establish the tools (importing a
    stored summary, say) and is distinct from ``[]``, which means it
    established that none ran.

    A window that touched the web is ``web`` even when a trusted tool ran
    beside it: the weaker guarantee is the one that has to be reported.
    """
    if not tools_used:
        return SOURCE_UNKNOWN
    if any(nom in _WEB_TOOLS for nom in tools_used):
        return SOURCE_WEB
    return SOURCE_TOOL


def fact_line(fact: str, source: str, date_utc: Optional[str] = None) -> str:
    """Compose the line as it is stored."""
    morceaux = [fact.strip(), source]
    if date_utc:
        morceaux.append(date_utc)
    return _SEP.join(morceaux)


def fact_text(line: str) -> str:
    """The claim, without its provenance suffix.

    What dedupe compares and what a reader is shown. The daily summary is
    cumulative and re-seeds the same facts on every flush, so comparing
    whole lines would stop recognising a fact re-extracted on a later date
    and append a copy once per flush, for ever.
    """
    return _SUFFIX_RE.sub("", line).strip()


def fact_source(line: str) -> str:
    """What the line rests on, or ``inconnu`` when it does not say."""
    m = _SUFFIX_RE.search(line)
    return m.group(1) if m else SOURCE_UNKNOWN
