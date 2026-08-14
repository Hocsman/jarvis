"""What a tool name is allowed to look like, in one place.

An MCP tool name is `f"{server}__{tool}"`, built from whatever a server
put in its `tools/list` reply. `mcp.types.Tool.name` is a bare `str`:
no character class, no length bound, no normalisation between the wire
and here. Everything downstream is line-based — the confirmation card's
heading, the routine envelope's list, `outils.md`'s sections, and the
one-tool-per-line catalogues the router and planner read — so a newline
in a name writes a line nobody intended.

The class binds what this codebase *writes*. It never binds what it
reads back from the user's own files: a line he typed by hand means what
it says.
"""

from __future__ import annotations

import re

# Deliberately narrow. Every real tool name in the catalogue passes it,
# and the failure it produces leans the safe way: a name outside the
# class is unclassifiable, therefore destructive, therefore asked about.
PLAIN_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def is_plain_name(name: str) -> bool:
    """Whether this name can be written on a line without forging one."""
    return bool(PLAIN_NAME.match(name or ""))


def one_line(text: str, limit: int = 0) -> str:
    """Collapse third-party prose to a single line.

    For a description rather than a name: a description is legitimately
    prose and refusing it would drop honest tools, but it is read from a
    list where one entry is one line, so its own line breaks would open
    an entry of its own.
    """
    flat = " ".join((text or "").split())
    return flat[:limit] if limit and len(flat) > limit else flat
